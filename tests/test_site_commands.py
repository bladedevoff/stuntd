import json

import pytest

from stuntd.cli import main
from stuntd.serve.modes import MODE_LIVE, MODE_SHADOW, read_mode, write_mode
from stuntd.store.db import Capture, Decision, Store
from stuntd.store.redact import Redactor
from stuntd.train.artifacts import SiteModel, load_model, save_model, site_dir
from stuntd.train.metrics import ClassStats

SPAM = '{"properties":{"spam":{"type":"boolean"}},"type":"object"}'
HASH_SITE = "0123456789abcdef"


def seed(data_dir, site, name=None, count=1):
    data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(data_dir / "captures.sqlite", Redactor())
    for i in range(count):
        store.record(
            Capture(site, SPAM, "boolean", f"user: m{i}", "true", "m", 1, 1, 1, schema_name=name)
        )
        store.record_decision(Decision(site, "shadow", "true", 1.0, True, 1, 0.0))
    store.close()


def publish(data_dir, site, mode=MODE_SHADOW):
    folder = save_model(
        data_dir / "models",
        SiteModel(
            site=site,
            kind="boolean",
            field="spam",
            labels=["false", "true"],
            base_model="b",
            temperature=1.0,
            threshold=0.6,
            target_agreement=0.99,
            trained_at=100.0,
            n_train=8,
            n_holdout=2,
            agreement=1.0,
            coverage=1.0,
            covered_agreement=1.0,
            ece=0.0,
            per_class={"false": ClassStats(1, 1.0), "true": ClassStats(1, 1.0)},
            confident_errors=[],
            curve=[],
        ),
    )
    (folder / "head.safetensors").write_bytes(b"h")
    write_mode(folder, mode, 1.0)


def captured_sites(data_dir):
    store = Store(data_dir / "captures.sqlite", Redactor())
    try:
        return sorted(info.site for info in store.sites())
    finally:
        store.close()


def decided_sites(data_dir):
    store = Store(data_dir / "captures.sqlite", Redactor())
    try:
        return sorted(store.decision_counts())
    finally:
        store.close()


def test_rm_removes_captures_decisions_and_the_model(data_dir, capsys):
    seed(data_dir, "spam")
    seed(data_dir, "other")
    publish(data_dir, "spam")
    publish(data_dir, "other")
    assert main(["site", "rm", "spam", "--yes"]) == 0
    assert capsys.readouterr().out == "removed spam\n"
    assert captured_sites(data_dir) == decided_sites(data_dir) == ["other"]
    assert not (data_dir / "models" / "spam").exists()
    assert (data_dir / "models" / "other").is_dir()


def test_rm_removes_the_field_sites_of_a_parent(data_dir, capsys):
    for site in ("triage.category", "triage.urgent", "triage2"):
        seed(data_dir, site)
        publish(data_dir, site)
    assert main(["site", "rm", "triage", "--yes"]) == 0
    assert capsys.readouterr().out == "removed triage.category, triage.urgent\n"
    assert captured_sites(data_dir) == ["triage2"]
    assert sorted(p.name for p in (data_dir / "models").iterdir()) == ["triage2"]


@pytest.mark.parametrize(
    ("reply", "removed"),
    [("y", True), ("yes", True), ("n", False), ("", False)],
    ids=["y", "yes", "n", "empty"],
)
def test_rm_asks_before_removing(data_dir, monkeypatch, reply, removed):
    seed(data_dir, "spam")
    monkeypatch.setattr("builtins.input", lambda prompt: reply)
    assert main(["site", "rm", "spam"]) == (0 if removed else 1)
    assert captured_sites(data_dir) == ([] if removed else ["spam"])


def test_rm_without_an_answer_keeps_the_site(data_dir, monkeypatch):
    seed(data_dir, "spam")

    def closed(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", closed)
    assert main(["site", "rm", "spam"]) == 1
    assert captured_sites(data_dir) == ["spam"]


def test_rm_refuses_a_served_site_without_force(data_dir, capsys):
    seed(data_dir, "spam")
    publish(data_dir, "spam", MODE_LIVE)
    assert main(["site", "rm", "spam", "--yes"]) == 1
    assert "stuntd: spam is served live; disable it or use --force" in capsys.readouterr().err
    assert captured_sites(data_dir) == ["spam"]
    assert main(["site", "rm", "spam", "--yes", "--force"]) == 0
    assert captured_sites(data_dir) == []
    assert not (data_dir / "models" / "spam").exists()


def test_rm_of_an_unknown_site_is_reported(data_dir, capsys):
    seed(data_dir, "spam")
    assert main(["site", "rm", "ghost", "--yes"]) == 2
    assert "stuntd: no site ghost" in capsys.readouterr().err


def test_rename_moves_captures_decisions_and_the_model(data_dir, capsys):
    seed(data_dir, "old")
    publish(data_dir, "old", MODE_LIVE)
    assert main(["site", "rename", "old", "new"]) == 0
    assert capsys.readouterr().out == "renamed old to new\n"
    assert captured_sites(data_dir) == decided_sites(data_dir) == ["new"]
    models = data_dir / "models"
    assert load_model(models, "new").site == "new"
    assert read_mode(site_dir(models, "new"))[0] == MODE_LIVE
    assert (site_dir(models, "new") / "head.safetensors").is_file()
    assert not site_dir(models, "old").exists()


def test_rename_puts_the_model_back_when_the_database_move_fails(data_dir, monkeypatch, capsys):
    seed(data_dir, "old")
    publish(data_dir, "old", MODE_LIVE)

    def fail(self, old, new):
        raise RuntimeError("disk full")

    monkeypatch.setattr(Store, "rename_site", fail)
    assert main(["site", "rename", "old", "new"]) == 2
    models = data_dir / "models"
    assert "stuntd: disk full" in capsys.readouterr().err
    assert load_model(models, "old").site == "old"
    assert not site_dir(models, "new").exists()


def test_rename_moves_the_field_sites_of_a_parent(data_dir):
    for site in ("old.category", "old.urgent"):
        seed(data_dir, site)
        publish(data_dir, site)
    assert main(["site", "rename", "old", "new"]) == 0
    assert captured_sites(data_dir) == ["new.category", "new.urgent"]
    assert sorted(p.name for p in (data_dir / "models").iterdir()) == [
        "new.category",
        "new.urgent",
    ]
    assert load_model(data_dir / "models", "new.urgent").site == "new.urgent"


def test_rename_a_site_that_has_no_model_yet(data_dir):
    seed(data_dir, "old")
    assert main(["site", "rename", "old", "new"]) == 0
    assert captured_sites(data_dir) == ["new"]


def test_rename_onto_an_existing_site_is_refused(data_dir, capsys):
    seed(data_dir, "old")
    seed(data_dir, "new")
    assert main(["site", "rename", "old", "new"]) == 2
    assert "stuntd: site new already exists" in capsys.readouterr().err
    assert captured_sites(data_dir) == ["new", "old"]


def test_rename_onto_an_existing_model_is_refused(data_dir, capsys):
    seed(data_dir, "old")
    publish(data_dir, "new")
    assert main(["site", "rename", "old", "new"]) == 2
    assert "stuntd: site new already exists" in capsys.readouterr().err


def test_rename_of_an_unknown_site_is_reported(data_dir, capsys):
    seed(data_dir, "spam")
    assert main(["site", "rename", "ghost", "new"]) == 2
    assert "stuntd: no site ghost" in capsys.readouterr().err


def test_rename_to_an_invalid_name_is_refused(data_dir, capsys):
    seed(data_dir, "old")
    assert main(["site", "rename", "old", "../escape"]) == 2
    assert "stuntd: invalid site name '../escape'" in capsys.readouterr().err
    assert captured_sites(data_dir) == ["old"]


def test_a_colon_in_a_site_name_reaches_its_folder(data_dir):
    seed(data_dir, "ns.question")
    publish(data_dir, "ns.question")
    assert main(["site", "rename", "ns:question", "ns:answer"]) == 0
    assert captured_sites(data_dir) == ["ns.answer"]


def test_status_shows_the_schema_name_next_to_a_hash_site(data_dir, capsys):
    seed(data_dir, HASH_SITE, name="triage")
    assert main(["status"]) == 0
    assert capsys.readouterr().out.splitlines()[1].split()[:4] == [
        HASH_SITE,
        "(triage)",
        "collect",
        "1",
    ]
    assert main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["sites"][0]["name"] == "triage"


def test_status_groups_field_sites_under_their_parent(data_dir, capsys):
    for field in ("category", "urgent"):
        seed(data_dir, f"{HASH_SITE}.{field}", name="triage")
    seed(data_dir, "other")
    assert main(["status"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].split() == [HASH_SITE, "(triage)"]
    assert lines[2].split()[:3] == [".category", "collect", "1"]
    assert lines[3].split()[:3] == [".urgent", "collect", "1"]
    assert lines[2].startswith("  .category")
    assert lines[4].split() == ["other", "collect", "1", "-", "-", "-", "-"]


def test_status_shows_a_parent_that_is_also_a_site_with_its_fields_below(data_dir, capsys):
    seed(data_dir, "triage", count=3)
    seed(data_dir, "triage.urgent")
    assert main(["status"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].split()[:3] == ["triage", "collect", "3"]
    assert lines[2].split()[:3] == [".urgent", "collect", "1"]
