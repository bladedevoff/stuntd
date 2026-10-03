import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time

import pytest
import uvicorn

from stuntd.cli import main
from stuntd.store.db import Capture, Decision, Store
from stuntd.store.redact import Redactor
from stuntd.train.artifacts import HEAD_FILE
from stuntd.train.layout import Layout
from stuntd.train.run import TrainedHead


def test_status_without_db(data_dir, capsys):
    assert main(["status"]) == 0
    assert "no captures yet" in capsys.readouterr().out


def test_status_lists_sites(data_dir, capsys):
    data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(data_dir / "captures.sqlite", Redactor())
    store.record(Capture("mod", "{}", "choice", "user: x", "block", "gpt-x", 100, 1, 1))
    store.close()
    assert main(["status", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)["sites"]
    assert rows[0]["site"] == "mod" and rows[0]["captures"] == 1
    assert main(["status"]) == 0
    assert "mod" in capsys.readouterr().out


def test_status_json_lists_no_sites_without_db(data_dir, capsys):
    assert main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"learning": True, "sites": []}


def test_status_lists_the_field_sites_of_a_multi_field_decision(data_dir, capsys):
    data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(data_dir / "captures.sqlite", Redactor())
    for field in ("category", "urgent"):
        store.record(Capture(f"triage.{field}", "{}", "choice", "user: x", "a", "gpt-x", 100, 1, 1))
    store.close()
    assert main(["status", "--json"]) == 0
    sites = json.loads(capsys.readouterr().out)["sites"]
    assert [row["site"] for row in sites] == ["triage.category", "triage.urgent"]


def test_status_columns_stay_separated(data_dir, capsys):
    data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(data_dir / "captures.sqlite", Redactor())
    store.record(Capture("a" * 30, "{}", "classification", "user: x", "block", "gpt-x", 100, 1, 1))
    store.close()
    assert main(["status"]) == 0
    row = capsys.readouterr().out.splitlines()[1]
    assert row.split()[:3] == ["a" * 30, "collect", "1"]


def test_config_init_writes_template_once(data_dir, capsys):
    assert main(["config", "init"]) == 0
    path = data_dir / "stuntd.toml"
    assert path.exists() and "[redaction]" in path.read_text(encoding="utf-8")
    assert str(path) in capsys.readouterr().out
    assert main(["config", "init"]) == 1
    assert "exists" in capsys.readouterr().out
    assert main(["config", "init", "--force"]) == 0


def test_config_show_prints_dotted_keys(data_dir, capsys):
    assert main(["config", "show"]) == 0
    out = capsys.readouterr().out
    assert "training.epochs = 3  (default)" in out
    assert "serving.window = 100  (default)" in out
    assert "storage.max_rows = 100000  (default)" in out
    assert "redaction.enabled = True  (default)" in out
    assert "jev.upstream =   (default)" in out
    assert "learn = True  (default)" in out


def test_config_show_reports_sources(data_dir, capsys):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "stuntd.toml").write_text("port = 9000\n", encoding="utf-8")
    assert main(["config", "show", "--upstream", "http://flag"]) == 0
    out = capsys.readouterr().out
    assert "upstream = http://flag  (flag)" in out
    assert "port = 9000  (file)" in out
    assert "host = 127.0.0.1  (default)" in out
    assert "storage.models_dir =   (default)" in out
    assert "training.min_examples = 300  (default)" in out
    assert "training.holdout = 0.2  (default)" in out
    assert "training.target_agreement = 0.99  (default)" in out
    assert "training.base_model = convaiinnovations/laya  (default)" in out
    assert "training.epochs = 3  (default)" in out
    assert "training.device = auto  (default)" in out
    assert "serving.check_share = 0.02  (default)" in out
    assert "serving.window = 100  (default)" in out
    assert "serving.min_window = 20  (default)" in out
    assert "serving.auto_promote = False  (default)" in out
    assert "serving.auto_promote_after_hours = 24  (default)" in out
    assert "serving.cache_size = 1000  (default)" in out
    assert "learn = True  (default)" in out
    assert "jev.upstream =   (default)" in out
    assert "jev.require_key = False  (default)" in out
    assert "jev.model_name = stuntd  (default)" in out


def test_config_init_creates_the_parent_directory(data_dir, tmp_path, capsys):
    path = tmp_path / "missing" / "dir" / "stuntd.toml"
    assert main(["config", "init", "--config", str(path)]) == 0
    assert "[redaction]" in path.read_text(encoding="utf-8")
    assert f"wrote {path}" in capsys.readouterr().out


def test_config_show_reports_a_port_flag(data_dir, capsys):
    assert main(["config", "show", "--port", "1234"]) == 0
    out = capsys.readouterr().out
    assert "port = 1234  (flag)" in out
    assert "redaction.patterns =   (default)" in out


def test_config_show_joins_redaction_patterns(data_dir, capsys):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "stuntd.toml").write_text(
        "[redaction]\npatterns = ['A-\\d', 'B-\\d']\n", encoding="utf-8"
    )
    assert main(["config", "show"]) == 0
    assert "redaction.patterns = A-\\d, B-\\d  (file)" in capsys.readouterr().out


def test_unusable_config_path_is_reported(data_dir, tmp_path, capsys):
    missing = tmp_path / "missing.toml"
    assert main(["status", "--config", str(missing)]) == 2
    assert f"stuntd: {missing} does not exist" in capsys.readouterr().err
    assert main(["status", "--config", str(tmp_path)]) == 2
    assert f"stuntd: {tmp_path} is not a file" in capsys.readouterr().err


def test_serve_without_an_upstream_serves_only_jev(data_dir, capsys, monkeypatch):
    recorded = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: recorded.update(app=app))
    monkeypatch.setattr("stuntd.cli._make_decider", lambda settings: WarmingDecider())
    assert main(["serve"]) == 0
    recorded["app"].state.store.close()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "stuntd listening on http://127.0.0.1:8787"
    assert lines[1] == "endpoints: jev local"


def test_config_init_under_a_file_is_reported(data_dir, tmp_path, capsys):
    blocking_file = tmp_path / "afile"
    blocking_file.write_text("x", encoding="utf-8")
    assert main(["config", "init", "--config", str(blocking_file / "stuntd.toml")]) == 2
    assert capsys.readouterr().err.startswith("stuntd: ")


def test_unreadable_database_is_reported(data_dir, capsys):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "captures.sqlite").write_bytes(b"not a database at all")
    assert main(["status"]) == 2
    assert capsys.readouterr().err.startswith("stuntd: ")


def test_serve_leaves_the_date_and_server_headers_to_the_provider(data_dir, monkeypatch):
    recorded = {}

    def record(app, **kwargs):
        recorded["app"] = app
        recorded.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", record)
    monkeypatch.setattr("stuntd.cli._make_decider", lambda settings: WarmingDecider())
    assert main(["serve", "--upstream", "http://127.0.0.1:9"]) == 0
    recorded["app"].state.store.close()
    assert recorded["server_header"] is False
    assert recorded["date_header"] is False


REMOTE_JEV = """[jev]
upstream = "https://jev.example"
"""


def write_config(data_dir, body):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "stuntd.toml").write_text(body, encoding="utf-8")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_serve_starts_and_terminates_cleanly(data_dir):
    write_config(data_dir, REMOTE_JEV)
    port = free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "stuntd",
            "serve",
            "--upstream",
            "http://127.0.0.1:9",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "STUNTD_DATA_DIR": str(data_dir)},
    )
    with proc:
        try:
            deadline = time.time() + 15
            while time.time() < deadline:
                with socket.socket() as s:
                    if s.connect_ex(("127.0.0.1", port)) == 0:
                        break
                time.sleep(0.2)
            else:
                proc.terminate()
                raise AssertionError("serve did not open the port; output:\n" + proc.stdout.read())
        finally:
            proc.terminate()
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
    assert proc.returncode is not None


@pytest.mark.parametrize(
    ("argv", "config", "endpoints"),
    [
        (["--upstream", "http://127.0.0.1:9"], "", "openai, anthropic, jev local"),
        ([], "", "jev local"),
        (["--upstream", "http://127.0.0.1:9"], REMOTE_JEV, "openai, anthropic, jev proxy"),
        ([], REMOTE_JEV, "jev proxy"),
    ],
    ids=["upstream", "no-upstream", "upstream-and-jev-provider", "jev-provider-only"],
)
def test_serve_names_the_endpoints_it_serves(
    data_dir, capsys, monkeypatch, argv, config, endpoints
):
    write_config(data_dir, config)
    recorded = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: recorded.update(app=app))
    monkeypatch.setattr("stuntd.cli._make_decider", lambda settings: WarmingDecider())
    assert main(["serve", *argv]) == 0
    recorded["app"].state.store.close()
    assert f"endpoints: {endpoints}" in capsys.readouterr().out.splitlines()


def test_serve_records_its_pid_while_it_runs(data_dir, monkeypatch):
    seen = {}

    def run(app, **kwargs):
        seen["pid"] = (data_dir / "stuntd.pid").read_text(encoding="utf-8")
        seen["app"] = app

    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setattr("stuntd.cli._make_decider", lambda settings: WarmingDecider())
    assert main(["serve"]) == 0
    seen["app"].state.store.close()
    assert seen["pid"] == str(os.getpid())
    assert not (data_dir / "stuntd.pid").exists()


def test_stop_ends_the_daemon_named_by_the_pid_file(data_dir, capsys, monkeypatch):
    killed = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    data_dir.mkdir(parents=True)
    (data_dir / "stuntd.pid").write_text("4242", encoding="utf-8")
    assert main(["stop"]) == 0
    assert killed == [(4242, signal.SIGTERM)]
    assert not (data_dir / "stuntd.pid").exists()
    assert "stopped stuntd (pid 4242)" in capsys.readouterr().out


def test_stop_without_a_pid_file_says_nothing_is_running(data_dir, capsys):
    assert main(["stop"]) == 1
    assert "stuntd is not running" in capsys.readouterr().err


def test_stop_removes_the_pid_file_of_a_daemon_that_is_gone(data_dir, capsys, monkeypatch):
    def gone(pid, sig):
        raise ProcessLookupError(pid)

    monkeypatch.setattr(os, "kill", gone)
    data_dir.mkdir(parents=True)
    (data_dir / "stuntd.pid").write_text("4242", encoding="utf-8")
    assert main(["stop"]) == 1
    assert not (data_dir / "stuntd.pid").exists()
    assert "stuntd is not running" in capsys.readouterr().err


def test_stop_removes_the_pid_file_when_the_os_reports_a_dead_pid_as_oserror(
    data_dir, capsys, monkeypatch
):
    def gone(pid, sig):
        raise OSError(22, "The parameter is incorrect")

    monkeypatch.setattr(os, "kill", gone)
    data_dir.mkdir(parents=True)
    (data_dir / "stuntd.pid").write_text("4242", encoding="utf-8")
    assert main(["stop"]) == 1
    assert not (data_dir / "stuntd.pid").exists()
    assert "stuntd is not running" in capsys.readouterr().err


def test_stop_ends_a_running_daemon(data_dir):
    write_config(data_dir, REMOTE_JEV)
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "stuntd", "serve", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        env={**os.environ, "STUNTD_DATA_DIR": str(data_dir)},
    )
    try:
        deadline = time.time() + 15
        while time.time() < deadline:
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.2)
        assert main(["stop"]) == 0
        assert proc.wait(timeout=10) is not None
    finally:
        proc.kill()
        proc.wait()
    assert not (data_dir / "stuntd.pid").exists()


SPAM = '{"properties":{"spam":{"type":"boolean"}},"type":"object"}'


def seed_captures(data_dir, n=10, site="s1"):
    data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(data_dir / "captures.sqlite", Redactor())
    for i in range(n):
        store.record(
            Capture(
                site, SPAM, "boolean", f"user: m{i}", "true" if i % 2 else "false", "m", 1, 1, 1
            )
        )
    store.close()


def fake_trainer_factory(settings):
    def trainer(dataset, head_path):
        head_path.write_bytes(b"h")
        return TrainedHead(
            [[2.0, 0.0] if item.label == 0 else [0.0, 2.0] for item in dataset.holdout],
            Layout(512, 192, spaced_labels=True),
        )

    return trainer


def warning_trainer_factory(settings):
    trained = fake_trainer_factory(settings)

    def trainer(dataset, head_path):
        logging.getLogger("stuntd.train.trainer").warning("%s: trains uncached", dataset.site)
        return trained(dataset, head_path)

    return trainer


def barely_covering_trainer_factory(settings):
    def trainer(dataset, head_path):
        head_path.write_bytes(b"h")
        sure = [0.0, 10.0] if dataset.holdout[0].label else [10.0, 0.0]
        logits = [sure] + [[0.001, 0.0] for _ in dataset.holdout[1:]]
        return TrainedHead(logits, Layout(512, 192, spaced_labels=True))

    return trainer


def write_training_config(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "stuntd.toml").write_text(
        "[training]\nmin_examples = 4\nholdout = 0.3\n", encoding="utf-8"
    )


def test_train_without_captures(data_dir, capsys):
    assert main(["train"]) == 0
    assert "no captures yet" in capsys.readouterr().out


def test_train_reports_each_site(data_dir, capsys, monkeypatch):
    seed_captures(data_dir)
    write_training_config(data_dir)
    monkeypatch.setattr("stuntd.cli._make_trainer", fake_trainer_factory)
    assert main(["train"]) == 0
    out = capsys.readouterr().out
    assert "base model: convaiinnovations/laya" in out
    assert "s1  trained  holdout agreement 1.000  coverage 1.00 at agreement 1.000" in out
    assert (data_dir / "models" / "s1" / "meta.json").exists()


def test_train_warnings_go_to_stderr(data_dir, capsys, monkeypatch):
    seed_captures(data_dir)
    write_training_config(data_dir)
    monkeypatch.setattr("stuntd.cli._make_trainer", warning_trainer_factory)
    assert main(["train"]) == 0
    assert capsys.readouterr().err == "stuntd: s1: trains uncached\n"
    assert logging.getLogger("stuntd").handlers == []


def progress_trainer_factory(settings):
    trained = fake_trainer_factory(settings)

    def trainer(dataset, head_path):
        for epoch in (1, 2):
            logging.getLogger("stuntd.train.trainer").info(
                "%s epoch %d/2 loss %.4f", dataset.site, epoch, 1 / epoch
            )
        return trained(dataset, head_path)

    return trainer


def test_train_prints_a_line_per_epoch_on_stderr(data_dir, capsys, monkeypatch):
    seed_captures(data_dir)
    write_training_config(data_dir)
    monkeypatch.setattr("stuntd.cli._make_trainer", progress_trainer_factory)
    assert main(["train"]) == 0
    assert capsys.readouterr().err == (
        "stuntd: s1 epoch 1/2 loss 1.0000\nstuntd: s1 epoch 2/2 loss 0.5000\n"
    )
    assert logging.getLogger("stuntd").level == logging.NOTSET


def test_the_trainer_is_built_from_the_training_settings(monkeypatch):
    import types

    from stuntd.cli import _make_trainer
    from stuntd.settings import Settings

    built = {}

    class FakeTrainer:
        def __init__(
            self, base_model, device, epochs, cache_encoder, cache_max_bytes, max_option_tokens
        ):
            built.update(
                epochs=epochs,
                cache_encoder=cache_encoder,
                cache_max_bytes=cache_max_bytes,
                max_option_tokens=max_option_tokens,
            )

    module = types.ModuleType("stuntd.train.trainer")
    module.LayaTrainer = FakeTrainer
    monkeypatch.setitem(sys.modules, "stuntd.train.trainer", module)
    _make_trainer(Settings(epochs=24, cache_encoder=False, cache_max_mb=512, max_option_tokens=640))
    assert built == {
        "epochs": 24,
        "cache_encoder": False,
        "cache_max_bytes": 512 * 2**20,
        "max_option_tokens": 640,
    }


def test_train_without_torch_explains_the_extra(data_dir, capsys, monkeypatch):
    seed_captures(data_dir)
    write_training_config(data_dir)

    def missing(settings):
        raise ImportError("No module named 'laya'")

    monkeypatch.setattr("stuntd.cli._make_trainer", missing)
    assert main(["train"]) == 2
    assert 'pip install "stuntd[train]"' in capsys.readouterr().err


def test_report_without_models(data_dir, capsys):
    assert main(["report"]) == 0
    assert "no trained sites yet" in capsys.readouterr().out
    assert main(["report", "--json"]) == 0
    assert capsys.readouterr().out.strip() == "[]"


def test_report_after_training(data_dir, capsys, monkeypatch):
    seed_captures(data_dir)
    write_training_config(data_dir)
    monkeypatch.setattr("stuntd.cli._make_trainer", fake_trainer_factory)
    main(["train"])
    capsys.readouterr()
    assert main(["report", "s1", "--curve"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("site s1  boolean spam") and "threshold  coverage  agreement" in out
    assert main(["report", "ghost"]) == 2
    assert "no model for ghost" in capsys.readouterr().err


def test_train_names_a_local_checkpoint_without_a_download_note(
    data_dir, tmp_path, capsys, monkeypatch
):
    seed_captures(data_dir)
    checkpoint = tmp_path / "laya-checkpoint"
    checkpoint.mkdir()
    (data_dir / "stuntd.toml").write_text(
        f"[training]\nmin_examples = 4\nholdout = 0.3\nbase_model = '{checkpoint}'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("stuntd.cli._make_trainer", fake_trainer_factory)
    assert main(["train"]) == 0
    out = capsys.readouterr().out
    assert f"base model: {checkpoint}" in out and "downloaded on first use" not in out


def test_train_says_a_named_base_model_is_downloaded(data_dir, capsys, monkeypatch):
    seed_captures(data_dir)
    write_training_config(data_dir)
    monkeypatch.setattr("stuntd.cli._make_trainer", fake_trainer_factory)
    assert main(["train"]) == 0
    out = capsys.readouterr().out
    assert "base model: convaiinnovations/laya (downloaded on first use)" in out


def test_train_skips_an_unknown_site_without_building_the_trainer(data_dir, capsys, monkeypatch):
    seed_captures(data_dir)
    write_training_config(data_dir)
    built = []

    def record(settings):
        built.append(settings.base_model)
        return fake_trainer_factory(settings)

    monkeypatch.setattr("stuntd.cli._make_trainer", record)
    assert main(["train", "ghost"]) == 0
    assert capsys.readouterr().out.strip() == "ghost  skipped  no captures for this site"
    assert built == []


def train_site(data_dir, monkeypatch, capsys):
    seed_captures(data_dir)
    write_training_config(data_dir)
    monkeypatch.setattr("stuntd.cli._make_trainer", fake_trainer_factory)
    assert main(["train"]) == 0
    capsys.readouterr()


def skip_extra_check(monkeypatch):
    monkeypatch.setattr("stuntd.cli._require_serving", lambda: None)


def set_coverage(data_dir, coverage):
    meta = data_dir / "models" / "s1" / "meta.json"
    raw = json.loads(meta.read_text(encoding="utf-8"))
    raw["coverage"] = coverage
    meta.write_text(json.dumps(raw), encoding="utf-8")


def mode_of(data_dir, site="s1"):
    path = data_dir / "models" / site / "mode.json"
    return json.loads(path.read_text(encoding="utf-8"))["mode"]


def test_train_on_an_empty_database(data_dir, capsys):
    data_dir.mkdir(parents=True, exist_ok=True)
    Store(data_dir / "captures.sqlite", Redactor()).close()
    assert main(["train"]) == 0
    assert "no captures yet" in capsys.readouterr().out


def test_enable_without_a_model_is_reported(data_dir, capsys):
    assert main(["enable", "s1"]) == 2
    assert "stuntd: no model for s1" in capsys.readouterr().err


def test_disable_without_a_model_is_reported(data_dir, capsys):
    assert main(["disable", "s1"]) == 2
    assert "stuntd: no model for s1" in capsys.readouterr().err


def test_enable_without_a_threshold_asks_for_more_examples(data_dir, capsys, monkeypatch):
    train_site(data_dir, monkeypatch, capsys)
    meta = data_dir / "models" / "s1" / "meta.json"
    raw = json.loads(meta.read_text(encoding="utf-8"))
    raw["threshold"] = None
    meta.write_text(json.dumps(raw), encoding="utf-8")
    assert main(["enable", "s1"]) == 1
    assert "stuntd: s1 never reached the target agreement on its holdout" in capsys.readouterr().err
    assert mode_of(data_dir) == "shadow"


@pytest.mark.parametrize(
    ("factory", "code", "mode"),
    [(barely_covering_trainer_factory, 1, "shadow"), (fake_trainer_factory, 0, "live")],
    ids=["covers-nothing", "covers-the-holdout"],
)
def test_enable_needs_a_model_that_covers_the_holdout(
    data_dir, capsys, monkeypatch, factory, code, mode
):
    seed_captures(data_dir, n=1000)
    write_training_config(data_dir)
    monkeypatch.setattr("stuntd.cli._make_trainer", factory)
    assert main(["train"]) == 0
    trained = capsys.readouterr().out
    skip_extra_check(monkeypatch)
    assert main(["enable", "s1"]) == code
    captured = capsys.readouterr()
    assert mode_of(data_dir) == mode
    if code == 1:
        assert "coverage 0.00" in trained
        assert captured.err.strip() == (
            "stuntd: s1 covers 0% of the holdout at target 0.99: nothing will be answered"
            " locally; lower training.target_agreement or retrain"
        )


@pytest.mark.parametrize(
    ("coverage", "warning"),
    [
        (
            0.05,
            "stuntd: s1 covers 5% of the holdout at target 0.99;"
            " most answers will still go to the provider",
        ),
        (0.10, ""),
    ],
    ids=["covers-little", "covers-enough"],
)
def test_enable_warns_about_a_head_that_covers_little(
    data_dir, capsys, monkeypatch, coverage, warning
):
    train_site(data_dir, monkeypatch, capsys)
    (data_dir / "stuntd.toml").write_text('upstream = "http://up"\n', encoding="utf-8")
    set_coverage(data_dir, coverage)
    skip_extra_check(monkeypatch)
    assert main(["enable", "s1"]) == 0
    captured = capsys.readouterr()
    assert captured.err.strip() == warning
    assert captured.out.strip() == "s1: live"
    assert mode_of(data_dir) == "live"


@pytest.mark.parametrize(
    ("coverage", "fallback", "warning"),
    [
        (
            0.4,
            "zeroshot",
            "stuntd: s1 covers 40% of the holdout at target 0.99;"
            " the rest is answered zero-shot by the base checkpoint",
        ),
        (0.5, "zeroshot", ""),
        (0.4, "head", ""),
    ],
    ids=["covers-little", "covers-enough", "head-fallback"],
)
def test_enable_in_local_mode_warns_that_the_rest_is_answered_zero_shot(
    data_dir, capsys, monkeypatch, coverage, fallback, warning
):
    train_site(data_dir, monkeypatch, capsys)
    (data_dir / "stuntd.toml").write_text(
        f'[serving]\nlocal_fallback = "{fallback}"\n', encoding="utf-8"
    )
    set_coverage(data_dir, coverage)
    skip_extra_check(monkeypatch)
    assert main(["enable", "s1"]) == 0
    assert capsys.readouterr().err.strip() == warning


@pytest.mark.parametrize(
    ("command", "printed"),
    [
        ("train", "moderation.verdict  trained"),
        ("report", "site moderation.verdict"),
        ("enable", "moderation.verdict: live"),
        ("disable", "moderation.verdict: shadow"),
    ],
    ids=["train", "report", "enable", "disable"],
)
def test_a_colon_in_a_site_name_reaches_its_folder(data_dir, capsys, monkeypatch, command, printed):
    seed_captures(data_dir, site="moderation.verdict")
    write_training_config(data_dir)
    monkeypatch.setattr("stuntd.cli._make_trainer", fake_trainer_factory)
    skip_extra_check(monkeypatch)
    assert main(["train", "moderation:verdict"]) == 0
    capsys.readouterr()
    assert main([command, "moderation:verdict"]) == 0
    assert printed in capsys.readouterr().out


def test_enable_and_disable_switch_the_mode(data_dir, capsys, monkeypatch):
    train_site(data_dir, monkeypatch, capsys)
    skip_extra_check(monkeypatch)
    assert main(["enable", "s1"]) == 0
    assert capsys.readouterr().out.strip() == "s1: live"
    assert mode_of(data_dir) == "live"
    assert main(["disable", "s1"]) == 0
    assert capsys.readouterr().out.strip() == "s1: shadow"
    assert mode_of(data_dir) == "shadow"


def train_sites(data_dir, monkeypatch, capsys, *sites):
    for site in sites:
        seed_captures(data_dir, site=site)
    write_training_config(data_dir)
    monkeypatch.setattr("stuntd.cli._make_trainer", fake_trainer_factory)
    skip_extra_check(monkeypatch)
    assert main(["train"]) == 0
    capsys.readouterr()


def test_enable_and_disable_act_on_every_field_of_a_parent(data_dir, capsys, monkeypatch):
    train_sites(data_dir, monkeypatch, capsys, "triage.category", "triage.urgent")
    assert main(["enable", "triage"]) == 0
    assert capsys.readouterr().out.splitlines() == ["triage.category: live", "triage.urgent: live"]
    assert mode_of(data_dir, "triage.category") == mode_of(data_dir, "triage.urgent") == "live"
    assert main(["disable", "triage"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "triage.category: shadow",
        "triage.urgent: shadow",
    ]
    assert mode_of(data_dir, "triage.category") == mode_of(data_dir, "triage.urgent") == "shadow"


def test_enable_a_site_that_is_also_a_parent_leaves_its_fields_alone(data_dir, capsys, monkeypatch):
    train_sites(data_dir, monkeypatch, capsys, "triage", "triage.urgent")
    assert main(["enable", "triage"]) == 0
    assert capsys.readouterr().out.splitlines() == ["triage: live"]
    assert mode_of(data_dir, "triage.urgent") == "shadow"


def test_enable_a_parent_without_fields_is_reported(data_dir, capsys, monkeypatch):
    train_sites(data_dir, monkeypatch, capsys, "other")
    assert main(["enable", "triage"]) == 2
    assert "stuntd: no model for triage" in capsys.readouterr().err


def test_enable_without_torch_explains_the_extra(data_dir, capsys, monkeypatch):
    train_site(data_dir, monkeypatch, capsys)
    monkeypatch.setitem(sys.modules, "stuntd.serve.decider", None)
    assert main(["enable", "s1"]) == 2
    assert 'stuntd: serving needs the train extra: pip install "stuntd[train]"' in (
        capsys.readouterr().err
    )
    assert mode_of(data_dir) == "shadow"


def test_status_shows_a_site_without_a_model_as_collecting(data_dir, capsys):
    seed_captures(data_dir)
    assert main(["status"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == [
        "site",
        "mode",
        "captures",
        "shadow",
        "live",
        "agreement",
        "retrain",
    ]
    assert lines[1].split() == ["s1", "collect", "10", "-", "-", "-", "-"]
    assert main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "learning": True,
        "sites": [
            {
                "site": "s1",
                "name": None,
                "mode": "collect",
                "captures": 10,
                "shadow": 0,
                "live": 0,
                "agreement": None,
                "retrain": None,
            }
        ],
    }


def test_status_shows_modes_counts_and_agreement(data_dir, capsys, monkeypatch):
    train_site(data_dir, monkeypatch, capsys)
    skip_extra_check(monkeypatch)
    assert main(["enable", "s1"]) == 0
    capsys.readouterr()
    store = Store(data_dir / "captures.sqlite", Redactor())
    store.record_decision(Decision("s1", "shadow", "true", 1.0, True, 5, 0.0))
    store.record_decision(Decision("s1", "check", "true", 1.0, False, 5, 0.0))
    store.record_decision(Decision("s1", "live", "true", 1.0, None, 5, 0.0))
    store.close()
    assert main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "learning": True,
        "sites": [
            {
                "site": "s1",
                "name": None,
                "mode": "live",
                "captures": 10,
                "shadow": 2,
                "live": 1,
                "agreement": 0.5,
                "retrain": None,
            }
        ],
    }
    assert main(["status"]) == 0
    assert capsys.readouterr().out.splitlines()[1].split() == [
        "s1",
        "live",
        "10",
        "2",
        "1",
        "0.500",
        "-",
    ]


def test_status_shows_a_site_retrained_over_a_live_one(data_dir, capsys, monkeypatch):
    train_site(data_dir, monkeypatch, capsys)
    skip_extra_check(monkeypatch)
    assert main(["enable", "s1"]) == 0
    assert main(["train"]) == 0
    capsys.readouterr()
    assert main(["status"]) == 0
    assert capsys.readouterr().out.splitlines()[1].split()[1:5] == [
        "shadow",
        "(retrained,",
        "was",
        "live)",
    ]


def test_status_shows_how_the_last_automatic_training_ended(data_dir, capsys):
    data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(data_dir / "captures.sqlite", Redactor())
    store.record(Capture("s1", "{}", "choice", "user: x", "block", "gpt-x", 100, 1, 1))
    store.close()
    (data_dir / "logs").mkdir()
    (data_dir / "logs" / "train-s1.result").write_text("failed (exit 1)", encoding="utf-8")
    assert main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["sites"][0]["retrain"] == "failed (exit 1)"
    assert main(["status"]) == 0
    assert capsys.readouterr().out.splitlines()[1].split()[-3:] == ["failed", "(exit", "1)"]


def test_serve_builds_a_decider_for_a_serving_site(data_dir, capsys, monkeypatch):
    train_site(data_dir, monkeypatch, capsys)
    skip_extra_check(monkeypatch)
    assert main(["enable", "s1"]) == 0
    capsys.readouterr()
    decider = WarmingDecider()
    recorded = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: recorded.update(app=app))
    monkeypatch.setattr("stuntd.cli._make_decider", lambda settings: decider)
    assert main(["serve", "--upstream", "http://127.0.0.1:9"]) == 0
    recorded["app"].state.store.close()
    assert recorded["app"].state.runtime._decider is decider
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("stuntd listening on http://127.0.0.1:")
    assert lines[2] == "serving 1 site(s) with convaiinnovations/laya"


class WarmingDecider:
    """Stands in for the Decider, which serve reaches only to warm it up."""

    def __init__(self):
        self.warmed = []
        self.warmed_base = False

    def decide(self, model, head_path, text):
        raise AssertionError("serve decided nothing while it started")

    def warm(self, model, head_path):
        self.warmed.append((model.site, head_path))

    def warm_base(self):
        self.warmed_base = True


def test_serve_warms_the_first_serving_site(data_dir, capsys, monkeypatch):
    train_site(data_dir, monkeypatch, capsys)
    skip_extra_check(monkeypatch)
    assert main(["enable", "s1"]) == 0
    capsys.readouterr()
    decider = WarmingDecider()
    recorded = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: recorded.update(app=app))
    monkeypatch.setattr("stuntd.cli._make_decider", lambda settings: decider)
    assert main(["serve", "--upstream", "http://127.0.0.1:9"]) == 0
    recorded["app"].state.store.close()
    assert decider.warmed == [("s1", data_dir / "models" / "s1" / HEAD_FILE)]
    assert not decider.warmed_base


def test_serve_warms_nothing_when_every_site_collects(data_dir, capsys, monkeypatch):
    seed_captures(data_dir)
    decider = WarmingDecider()
    recorded = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: recorded.update(app=app))
    monkeypatch.setattr("stuntd.cli._make_decider", lambda settings: decider)
    assert main(["serve", "--upstream", "http://127.0.0.1:9"]) == 0
    recorded["app"].state.store.close()
    assert decider.warmed == []
    assert decider.warmed_base
    assert "serving jev locally with convaiinnovations/laya" in capsys.readouterr().out


@pytest.mark.parametrize("via_flag", [True, False], ids=["flag", "config"])
def test_serve_skips_warming_and_says_the_model_loads_on_first_use(
    data_dir, capsys, monkeypatch, via_flag
):
    seed_captures(data_dir)
    if not via_flag:
        write_config(data_dir, "[serving]\nlazy_load = true\n")
    decider = WarmingDecider()
    recorded = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: recorded.update(app=app))
    monkeypatch.setattr("stuntd.cli._make_decider", lambda settings: decider)
    argv = ["serve", "--upstream", "http://127.0.0.1:9"]
    assert main([*argv, "--lazy"] if via_flag else argv) == 0
    recorded["app"].state.store.close()
    assert decider.warmed == []
    assert not decider.warmed_base
    assert (
        "serving jev locally with convaiinnovations/laya, loading on first use"
        in capsys.readouterr().out
    )


def test_serve_without_torch_reports_serving_disabled(data_dir, capsys, monkeypatch):
    train_site(data_dir, monkeypatch, capsys)
    recorded = {}

    def missing(settings):
        raise ImportError("No module named 'laya'")

    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: recorded.update(app=app))
    monkeypatch.setattr("stuntd.cli._make_decider", missing)
    assert main(["serve", "--upstream", "http://127.0.0.1:9"]) == 0
    recorded["app"].state.store.close()
    assert recorded["app"].state.runtime._decider is None
    captured = capsys.readouterr()
    assert 'stuntd: serving disabled: pip install "stuntd[train]"' in captured.err
    assert "serving 1 site(s)" not in captured.out


def test_serve_builds_no_decider_without_a_trained_site_or_a_local_jev(
    data_dir, capsys, monkeypatch
):
    write_config(data_dir, REMOTE_JEV)
    built = []
    recorded = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: recorded.update(app=app))
    monkeypatch.setattr("stuntd.cli._make_decider", lambda settings: built.append(settings))
    assert main(["serve", "--upstream", "http://127.0.0.1:9"]) == 0
    recorded["app"].state.store.close()
    assert built == []
    assert "serving" not in capsys.readouterr().out


def test_serve_says_nothing_about_listening_when_the_base_model_fails(
    data_dir, capsys, monkeypatch
):
    train_site(data_dir, monkeypatch, capsys)
    ran = []

    def unloadable(settings):
        raise RuntimeError("base model failed to load: no checkpoint")

    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: ran.append(app))
    monkeypatch.setattr("stuntd.cli._make_decider", unloadable)
    assert main(["serve", "--upstream", "http://127.0.0.1:9"]) == 2
    captured = capsys.readouterr()
    assert "stuntd: base model failed to load: no checkpoint" in captured.err
    assert captured.out == ""
    assert ran == []


def test_status_shows_a_trained_site_without_a_database(data_dir, capsys, monkeypatch):
    train_site(data_dir, monkeypatch, capsys)
    (data_dir / "captures.sqlite").unlink()
    assert main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "learning": True,
        "sites": [
            {
                "site": "s1",
                "name": None,
                "mode": "shadow",
                "captures": 0,
                "shadow": 0,
                "live": 0,
                "agreement": None,
                "retrain": None,
            }
        ],
    }
    assert main(["status"]) == 0
    assert capsys.readouterr().out.splitlines()[1].split() == [
        "s1",
        "shadow",
        "0",
        "0",
        "0",
        "-",
        "-",
    ]
    assert not (data_dir / "captures.sqlite").exists()


def test_read_only_commands_never_import_torch(data_dir):
    seed_captures(data_dir)
    script = (
        "import sys\n"
        "from stuntd.cli import main\n"
        "for command in (['status'], ['report'], ['config', 'show']):\n"
        "    assert main(command) == 0\n"
        "assert 'torch' not in sys.modules, sorted(sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "STUNTD_DATA_DIR": str(data_dir)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
