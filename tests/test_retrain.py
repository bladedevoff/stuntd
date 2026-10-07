import itertools
import json
import logging
import subprocess
import sys
import threading
from types import SimpleNamespace

import anyio
import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from stuntd.proxy.app import build_app
from stuntd.serve import retrain
from stuntd.serve.retrain import Retrainer, last_result
from stuntd.serve.runtime import Runtime
from stuntd.settings import Settings, models_path
from stuntd.store.db import Capture, Store
from stuntd.store.redact import Redactor
from stuntd.train.artifacts import SiteModel, save_model
from stuntd.train.metrics import ClassStats

pytestmark = pytest.mark.anyio

SITE = "mod"
UPSTREAM_BODY = json.dumps(
    {"choices": [{"message": {"role": "assistant", "content": '{"verdict": "block"}'}}]}
).encode()
REQUEST = {
    "model": "gpt-x",
    "messages": [{"role": "user", "content": "buy pills"}],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "mod",
            "schema": {
                "type": "object",
                "properties": {"verdict": {"type": "string", "enum": ["allow", "block"]}},
            },
        },
    },
}


class FakeTraining:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.commands = []
        self.release = threading.Event()

    def __call__(self, command, stdout, stderr):
        self.commands.append(command)
        stdout.write(b"training output")
        assert stderr == subprocess.STDOUT
        assert self.release.wait(10)
        return SimpleNamespace(returncode=self.returncode)


@pytest.fixture
def training(monkeypatch):
    fake = FakeTraining()
    monkeypatch.setattr(subprocess, "run", fake)
    yield fake
    fake.release.set()


async def until(condition):
    for _ in range(500):
        if condition():
            return
        await anyio.sleep(0.01)
    raise AssertionError("condition not reached")


def trained(site, trained_at):
    return SiteModel(
        site=site,
        kind="choice",
        field="verdict",
        labels=["allow", "block"],
        base_model="b",
        temperature=1.0,
        threshold=0.6,
        target_agreement=0.9,
        trained_at=trained_at,
        n_train=8,
        n_holdout=2,
        agreement=1.0,
        coverage=1.0,
        covered_agreement=1.0,
        ece=0.0,
        per_class={name: ClassStats(1, 1.0) for name in ("allow", "block")},
        confident_errors=[],
        curve=[],
    )


@pytest.fixture
def retraining(data_dir, tmp_path):
    made = []

    def make(config=None, now=None, **overrides):
        settings = Settings(
            **{"auto_retrain": 2, "min_examples": 2, "auto_retrain_min_minutes": 0, **overrides}
        )
        store = Store(tmp_path / "captures.sqlite", Redactor())
        made.append(store)
        retrainer = Retrainer(
            settings, store, Runtime(settings, store, None), config, *([now] if now else [])
        )
        texts = itertools.count()

        def capture(site=SITE, text=None):
            text = f"user: {next(texts)}" if text is None else text
            store.record(Capture(site, "{}", "choice", text, "block", "gpt-x", 1, 1, 1))
            retrainer.record(site, text)

        return settings, capture

    yield make
    for store in made:
        store.close()


async def test_retrainer_trains_the_site_once_enough_captures_arrived(retraining, training):
    _, capture = retraining()
    capture()
    assert training.commands == []
    capture()
    await until(lambda: training.commands)
    training.release.set()
    await until(lambda: last_result(SITE))
    assert last_result(SITE) == "ok"


async def test_retrainer_runs_the_train_command_with_the_config(retraining, training, data_dir):
    _, capture = retraining(config=data_dir / "stuntd.toml")
    capture()
    capture()
    await until(lambda: training.commands)
    training.release.set()
    await until(lambda: last_result(SITE))
    assert training.commands == [
        [sys.executable, "-m", "stuntd", "train", SITE, "--config", str(data_dir / "stuntd.toml")]
    ]
    assert (data_dir / "logs" / f"train-{SITE}.log").read_bytes() == b"training output"


async def test_retrainer_waits_for_min_examples_when_the_site_has_no_model(retraining, training):
    _, capture = retraining(min_examples=4)
    for _ in range(3):
        capture()
    assert training.commands == []
    capture()
    await until(lambda: training.commands)


async def test_retrainer_counts_captures_from_when_the_model_was_trained(retraining, training):
    settings, capture = retraining()
    save_model(models_path(settings), trained(SITE, 4_000_000_000.0))
    capture()
    capture()
    await anyio.sleep(0.05)
    assert training.commands == []


async def test_retrainer_starts_one_run_when_captures_keep_crossing_the_threshold(
    retraining, training
):
    _, capture = retraining()
    for _ in range(5):
        capture()
    await until(lambda: training.commands)
    capture("other")
    capture("other")
    await anyio.sleep(0.05)
    assert len(training.commands) == 1


async def test_retrainer_starts_the_next_site_once_the_run_has_finished(retraining, training):
    _, capture = retraining()
    capture()
    capture()
    capture("other")
    capture("other")
    await until(lambda: training.commands)
    training.release.set()
    await until(lambda: last_result(SITE))
    capture("other")
    await until(lambda: len(training.commands) == 2)
    assert training.commands[1][4] == "other"


async def test_retrainer_logs_a_failed_run_and_waits_for_more_captures(
    retraining, monkeypatch, caplog, data_dir
):
    failing = FakeTraining(returncode=3)
    failing.release.set()
    monkeypatch.setattr(subprocess, "run", failing)
    _, capture = retraining()
    with caplog.at_level(logging.ERROR):
        capture()
        capture()
        await until(lambda: last_result(SITE))
    assert last_result(SITE) == "failed (exit 3)"
    message = caplog.records[0].getMessage()
    assert "exited with 3" in message
    assert str(data_dir / "logs" / f"train-{SITE}.log") in message
    capture()
    await anyio.sleep(0.05)
    assert len(failing.commands) == 1
    capture()
    await until(lambda: len(failing.commands) == 2)


async def test_retrainer_reports_a_command_that_cannot_start(retraining, monkeypatch, caplog):
    def missing(command, stdout, stderr):
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(subprocess, "run", missing)
    _, capture = retraining()
    with caplog.at_level(logging.ERROR):
        capture()
        capture()
        await until(lambda: last_result(SITE))
    assert last_result(SITE) == "failed (not started)"
    assert "could not start" in caplog.records[0].getMessage()


def test_last_result_is_none_before_any_run(data_dir):
    assert last_result(SITE) is None


async def test_proxy_serves_requests_while_a_run_is_in_progress(data_dir, training):
    async def provider(request: Request):
        return Response(content=UPSTREAM_BODY, media_type="application/json")

    upstream = Starlette(routes=[Route("/{path:path}", provider, methods=["POST"])])
    app = build_app(
        Settings(upstream="http://upstream", auto_retrain=2, min_examples=2),
        transport=httpx.ASGITransport(app=upstream),
        config=data_dir / "stuntd.toml",
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        for index in range(4):
            request = {**REQUEST, "messages": [{"role": "user", "content": f"buy pills {index}"}]}
            response = await client.post(
                "/v1/chat/completions", json=request, headers={"X-Stuntd-Site": SITE}
            )
            assert response.content == UPSTREAM_BODY
    await until(lambda: training.commands)
    assert len(training.commands) == 1
    training.release.set()
    await until(lambda: last_result(SITE))
    await app.state.proxy.aclose()


async def test_retrainer_counts_again_from_zero_after_a_finished_run(retraining, training):
    _, capture = retraining()
    capture()
    capture()
    await until(lambda: training.commands)
    training.release.set()
    await until(lambda: last_result(SITE))
    capture()
    await anyio.sleep(0.05)
    assert len(training.commands) == 1
    capture()
    await until(lambda: len(training.commands) == 2)


async def test_retrainer_trains_again_after_a_failure_to_prepare_the_logs(
    retraining, training, monkeypatch
):
    prepare = retrain.private_dir
    attempts = []

    def failing_once(path):
        prepare(path)
        attempts.append(path)
        if len(attempts) == 1:
            raise PermissionError(path)
        return path

    monkeypatch.setattr(retrain, "private_dir", failing_once)
    _, capture = retraining()
    capture()
    capture()
    await until(lambda: last_result(SITE))
    assert last_result(SITE) == "failed (not started)"
    capture()
    capture()
    await until(lambda: training.commands)


async def test_retrainer_does_not_count_a_repeated_text_twice(retraining, training):
    _, capture = retraining()
    for _ in range(5):
        capture(text="user: same")
    await anyio.sleep(0.05)
    assert training.commands == []
    capture(text="user: other")
    await until(lambda: training.commands)


async def test_retrainer_waits_the_minimum_minutes_between_runs_of_a_site(retraining, training):
    clock = [1000.0]
    _, capture = retraining(now=lambda: clock[0], auto_retrain_min_minutes=30)
    capture()
    capture()
    await until(lambda: training.commands)
    training.release.set()
    await until(lambda: last_result(SITE))
    capture()
    capture()
    await anyio.sleep(0.05)
    assert len(training.commands) == 1
    clock[0] += 29 * 60
    capture()
    await anyio.sleep(0.05)
    assert len(training.commands) == 1
    clock[0] += 60
    capture()
    await until(lambda: len(training.commands) == 2)


async def test_retrainer_trains_a_new_site_at_once_when_the_wait_is_for_another(
    retraining, training
):
    _, capture = retraining(now=lambda: 1000.0, auto_retrain_min_minutes=30)
    capture()
    capture()
    await until(lambda: training.commands)
    training.release.set()
    await until(lambda: last_result(SITE))
    capture("other")
    capture("other")
    await until(lambda: len(training.commands) == 2)


JEV_TONE = {"type": "choice", "instructions": "Tone?", "criteria": {"calm": None, "angry": None}}
JEV_ANSWER = json.dumps(
    {
        "model": "jev-1",
        "answers": {
            "tone": {
                "type": "choice",
                "choice": "calm",
                "confidence": 0.9,
                "probabilities": {"calm": 0.9, "angry": 0.1},
            }
        },
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
).encode()


@pytest.fixture
async def jev_client(data_dir):
    async def provider(request: Request):
        return Response(content=JEV_ANSWER, media_type="application/json")

    upstream = Starlette(routes=[Route("/{path:path}", provider, methods=["POST"])])
    app = build_app(
        Settings(
            upstream="http://upstream",
            jev_upstream="http://jev-provider",
            auto_retrain=2,
            min_examples=2,
        ),
        transport=httpx.ASGITransport(app=upstream),
        config=data_dir / "stuntd.toml",
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:

        async def ask(state):
            body = {"state": state, "model": "jev-latest", "questions": {"tone": JEV_TONE}}
            response = await client.post("/v1/systemone", json=body)
            assert response.content == JEV_ANSWER

        yield app, ask
    await app.state.proxy.aclose()


async def test_a_jev_capture_is_counted_toward_auto_retrain(jev_client, monkeypatch):
    app, ask = jev_client
    counted = []
    monkeypatch.setattr(app.state.proxy.retrainer, "record", lambda *args: counted.append(args))
    await ask("I was charged twice.")
    assert counted == [("tone", "I was charged twice.")]


async def test_a_repeated_jev_text_is_not_counted_twice(jev_client, training):
    _, ask = jev_client
    for _ in range(4):
        await ask("I was charged twice.")
    await anyio.sleep(0.05)
    assert training.commands == []
    await ask("Where is my invoice?")
    await until(lambda: training.commands)


async def test_a_jev_site_retrains_in_the_background(jev_client, training):
    _, ask = jev_client
    for index in range(4):
        await ask(f"Question {index}")
    await until(lambda: training.commands)
    assert len(training.commands) == 1
    assert training.commands[0][4] == "tone"
    training.release.set()
    await until(lambda: last_result("tone"))
    assert last_result("tone") == "ok"
