import gzip
import json
import logging
from dataclasses import dataclass

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from stuntd.proxy.app import build_app
from stuntd.serve.modes import MODE_FILE, MODE_LIVE, MODE_SHADOW, read_mode, write_mode
from stuntd.settings import Settings, models_path
from stuntd.train.artifacts import SiteModel, save_model
from stuntd.train.metrics import ClassStats

pytestmark = pytest.mark.anyio

SITE = "mod"

MOD_REQUEST = {
    "model": "gpt-x",
    "messages": [
        {"role": "system", "content": "Moderate."},
        {"role": "user", "content": "buy pills"},
    ],
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

NUMBER_REQUEST = {
    "model": "gpt-x",
    "messages": [
        {"role": "system", "content": "Moderate."},
        {"role": "user", "content": "buy pills"},
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "score",
            "schema": {"type": "object", "properties": {"score": {"type": "number"}}},
        },
    },
}

TOOL_REQUEST = {
    "model": "gpt-x",
    "messages": [
        {"role": "system", "content": "Route."},
        {"role": "user", "content": "buy pills"},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "route",
                "parameters": {"type": "object", "properties": {"spam": {"type": "boolean"}}},
            },
        }
    ],
}

UPSTREAM_BODY = json.dumps(
    {
        "id": "x",
        "model": "gpt-x",
        "choices": [{"message": {"role": "assistant", "content": '{"verdict": "block"}'}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }
).encode()


GZIPPED_BODY = gzip.compress(UPSTREAM_BODY, mtime=0)


@dataclass(frozen=True)
class FakeVerdict:
    label: int
    confidence: float
    latency_ms: int
    probabilities: tuple[float, ...] = ()


SURE = FakeVerdict(1, 0.9, 7)


class FakeDecider:
    def __init__(self, verdict=SURE, error=None):
        self.verdict = verdict
        self.error = error
        self.calls = []

    def decide(self, model, head_path, text):
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return self.verdict


class Provider:
    """Fake upstream that counts the requests that reached it."""

    def __init__(self, body=UPSTREAM_BODY, headers=None):
        self.body = body
        self.headers = {"x-up": "1", **(headers or {})}
        self.calls = 0

    async def handle(self, request: Request):
        self.calls += 1
        return Response(content=self.body, media_type="application/json", headers=self.headers)


def model(site=SITE, kind="choice", field="verdict", labels=("allow", "block"), threshold=0.6):
    return SiteModel(
        site=site,
        kind=kind,
        field=field,
        labels=list(labels),
        base_model="b",
        temperature=1.0,
        threshold=threshold,
        target_agreement=0.9,
        trained_at=100.0,
        n_train=8,
        n_holdout=2,
        agreement=1.0,
        coverage=1.0,
        covered_agreement=1.0,
        ece=0.0,
        per_class={name: ClassStats(1, 1.0) for name in labels},
        confident_errors=[],
        curve=[],
    )


@pytest.fixture
async def serving(data_dir):
    apps = []

    def make(provider, *, site_model=None, mode=None, decider=None, **overrides):
        settings = Settings(upstream="http://upstream", **overrides)
        if site_model is not None:
            folder = save_model(models_path(settings), site_model)
            if mode is not None:
                write_mode(folder, mode, now=0.0)
        upstream = Starlette(routes=[Route("/{path:path}", provider.handle, methods=["POST"])])
        app = build_app(settings, transport=httpx.ASGITransport(app=upstream), decider=decider)
        apps.append(app)
        return app

    yield make
    for app in apps:
        await app.state.proxy.aclose()


async def post(app, payload=MOD_REQUEST):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        return await client.post(
            "/v1/chat/completions", json=payload, headers={"X-Stuntd-Site": SITE}
        )


async def stream_post(app, payload=MOD_REQUEST):
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as client,
        client.stream(
            "POST", "/v1/chat/completions", json=payload, headers={"X-Stuntd-Site": SITE}
        ) as response,
    ):
        raw = b"".join([chunk async for chunk in response.aiter_raw()])
    return response, raw


async def test_site_without_a_model_still_collects(serving):
    provider = Provider()
    app = serving(provider)
    response = await post(app)
    assert response.content == UPSTREAM_BODY
    assert response.headers["x-stuntd"] == f"collect; site={SITE}"
    assert [(row["site"], row["count"]) for row in app.state.store.stats()] == [(SITE, 1)]


@pytest.mark.parametrize(
    ("verdict", "answer", "agree"),
    [(SURE, "block", True), (FakeVerdict(0, 0.9, 7), "allow", False)],
    ids=["agrees", "differs"],
)
async def test_shadow_site_records_how_the_model_compared(serving, verdict, answer, agree):
    provider = Provider()
    app = serving(
        provider, site_model=model(), mode=MODE_SHADOW, decider=FakeDecider(verdict=verdict)
    )
    response = await post(app)
    assert response.content == UPSTREAM_BODY
    assert response.headers["x-up"] == "1"
    assert response.headers["x-stuntd"] == f"shadow; site={SITE}"
    assert [(row["site"], row["count"]) for row in app.state.store.stats()] == [(SITE, 1)]
    rows = app.state.store.decisions(SITE, 10)
    assert [(row.mode, row.answer, row.agree) for row in rows] == [("shadow", answer, agree)]


async def test_shadow_decision_failure_leaves_the_response_untouched(serving, caplog):
    provider = Provider()
    app = serving(
        provider,
        site_model=model(),
        mode=MODE_SHADOW,
        decider=FakeDecider(error=RuntimeError("boom")),
    )
    with caplog.at_level(logging.ERROR):
        response = await post(app)
    assert response.status_code == 200
    assert response.content == UPSTREAM_BODY
    assert app.state.store.decisions(SITE, 10) == []
    assert "shadow decision failed" in caplog.text


async def test_site_whose_state_cannot_be_read_still_collects(serving, caplog):
    provider = Provider()
    app = serving(provider, site_model=model(), mode=MODE_LIVE, decider=FakeDecider())
    (models_path(app.state.settings) / SITE / MODE_FILE).write_bytes(b"")
    with caplog.at_level(logging.ERROR):
        response = await post(app)
    assert response.status_code == 200
    assert response.content == UPSTREAM_BODY
    assert response.headers["x-stuntd"] == f"collect; site={SITE}; reason=state-error"
    assert f"{SITE} state unreadable" in caplog.text


async def test_live_site_answers_without_the_provider(serving):
    provider = Provider()
    app = serving(provider, site_model=model(), mode=MODE_LIVE, decider=FakeDecider())
    response = await post(app)
    body = json.loads(response.content)
    assert provider.calls == 0
    assert response.status_code == 200
    assert body["object"] == "chat.completion"
    assert body["model"] == "gpt-x"
    assert body["id"].startswith("chatcmpl-stuntd-")
    assert json.loads(body["choices"][0]["message"]["content"]) == {"verdict": "block"}
    assert response.headers["x-stuntd"] == f"live; site={SITE}; confidence=0.90"
    assert response.headers["content-type"] == "application/json"
    assert int(response.headers["content-length"]) == len(response.content)
    assert app.state.store.stats() == []
    rows = app.state.store.decisions(SITE, 10)
    assert [(row.mode, row.answer, row.agree) for row in rows] == [("live", "block", None)]


async def test_live_site_answers_a_tool_request_with_tool_calls(serving):
    provider = Provider()
    app = serving(
        provider,
        site_model=model(kind="boolean", field="spam", labels=("false", "true")),
        mode=MODE_LIVE,
        decider=FakeDecider(),
    )
    response = await post(app, TOOL_REQUEST)
    call = json.loads(response.content)["choices"][0]["message"]["tool_calls"][0]
    assert provider.calls == 0
    assert call["function"]["name"] == "route"
    assert json.loads(call["function"]["arguments"]) == {"spam": True}
    assert response.headers["x-stuntd"] == f"live; site={SITE}; confidence=0.90"


async def test_live_site_asks_the_provider_when_the_model_is_unsure(serving):
    provider = Provider()
    app = serving(
        provider,
        site_model=model(),
        mode=MODE_LIVE,
        decider=FakeDecider(verdict=FakeVerdict(1, 0.4, 7)),
    )
    response = await post(app)
    assert provider.calls == 1
    assert response.content == UPSTREAM_BODY
    assert response.headers["x-stuntd"] == f"collect; site={SITE}; reason=low-confidence"
    assert [(row["site"], row["count"]) for row in app.state.store.stats()] == [(SITE, 1)]


async def test_live_site_checks_a_sampled_answer_against_the_provider(serving):
    provider = Provider()
    app = serving(
        provider,
        site_model=model(),
        mode=MODE_LIVE,
        decider=FakeDecider(verdict=FakeVerdict(0, 0.9, 7)),
        check_share=1.0,
    )
    response = await post(app)
    assert provider.calls == 1
    assert response.content == UPSTREAM_BODY
    assert response.headers["x-stuntd"] == f"collect; site={SITE}; reason=check"
    rows = app.state.store.decisions(SITE, 10)
    assert [(row.mode, row.answer, row.agree) for row in rows] == [("check", "allow", False)]


async def test_live_site_without_a_decider_asks_the_provider(serving):
    provider = Provider()
    app = serving(provider, site_model=model(), mode=MODE_LIVE)
    response = await post(app)
    assert provider.calls == 1
    assert response.content == UPSTREAM_BODY
    assert response.headers["x-stuntd"] == f"collect; site={SITE}; reason=no-runtime"


async def test_live_site_falls_back_to_shadow_after_failed_checks(serving):
    provider = Provider()
    app = serving(
        provider,
        site_model=model(),
        mode=MODE_LIVE,
        decider=FakeDecider(verdict=FakeVerdict(0, 0.9, 7)),
        check_share=1.0,
        min_window=2,
        target_agreement=0.99,
    )
    for _ in range(2):
        checked = await post(app)
        assert checked.headers["x-stuntd"] == f"collect; site={SITE}; reason=check"
    folder = models_path(app.state.settings) / SITE
    assert read_mode(folder)[0] == MODE_SHADOW
    response = await post(app)
    assert response.headers["x-stuntd"] == f"shadow; site={SITE}"


async def test_streaming_request_to_a_live_site_passes_through(serving):
    provider = Provider()
    app = serving(provider, site_model=model(), mode=MODE_LIVE, decider=FakeDecider())
    response = await post(app, dict(MOD_REQUEST, stream=True))
    assert provider.calls == 1
    assert response.headers["x-stuntd"] == "passthrough; reason=streaming"
    assert app.state.store.decisions(SITE, 10) == []


async def test_shadow_site_relays_a_compressed_body_and_still_decides(serving):
    provider = Provider(body=GZIPPED_BODY, headers={"content-encoding": "gzip"})
    app = serving(provider, site_model=model(), mode=MODE_SHADOW, decider=FakeDecider())
    response, raw = await stream_post(app)
    assert response.headers["content-encoding"] == "gzip"
    assert raw == GZIPPED_BODY
    assert response.headers["x-stuntd"] == f"shadow; site={SITE}"
    rows = app.state.store.decisions(SITE, 10)
    assert [(row.mode, row.answer, row.agree) for row in rows] == [("shadow", "block", True)]


async def test_check_path_survives_a_store_failure(serving, caplog):
    provider = Provider()
    app = serving(
        provider,
        site_model=model(),
        mode=MODE_LIVE,
        decider=FakeDecider(verdict=FakeVerdict(0, 0.9, 7)),
        check_share=1.0,
    )
    app.state.store.close()
    with caplog.at_level(logging.ERROR):
        response = await post(app)
    assert response.status_code == 200
    assert response.content == UPSTREAM_BODY
    assert response.headers["x-stuntd"] == f"collect; site={SITE}; reason=check"
    assert "check decision not recorded" in caplog.text


async def test_live_answer_survives_a_store_failure(serving, caplog):
    provider = Provider()
    app = serving(provider, site_model=model(), mode=MODE_LIVE, decider=FakeDecider())
    app.state.store.close()
    with caplog.at_level(logging.ERROR):
        response = await post(app)
    assert provider.calls == 0
    assert json.loads(response.content)["object"] == "chat.completion"
    assert response.headers["x-stuntd"] == f"live; site={SITE}; confidence=0.90"
    assert "live decision not recorded" in caplog.text


async def test_live_site_asks_the_provider_when_its_labels_do_not_fit_the_schema(serving, caplog):
    provider = Provider()
    app = serving(provider, site_model=model(), mode=MODE_LIVE, decider=FakeDecider())
    with caplog.at_level(logging.ERROR):
        response = await post(app, NUMBER_REQUEST)
    assert provider.calls == 1
    assert response.content == UPSTREAM_BODY
    assert response.headers["x-stuntd"] == f"collect; site={SITE}; reason=model-error"
    assert "live answer not built" in caplog.text
