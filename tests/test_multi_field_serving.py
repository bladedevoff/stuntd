import json
from dataclasses import dataclass

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from stuntd.proxy.app import build_app
from stuntd.serve.modes import MODE_LIVE, MODE_SHADOW, read_mode, write_mode
from stuntd.settings import Settings, models_path
from stuntd.train.artifacts import SiteModel, save_model
from stuntd.train.metrics import ClassStats

pytestmark = pytest.mark.anyio

SITE = "triage"
PROPERTIES = {
    "category": {"type": "string", "enum": ["billing", "tech"]},
    "urgent": {"type": "boolean"},
}
MESSAGES = [
    {"role": "system", "content": "Triage."},
    {"role": "user", "content": "card declined"},
]
REQUEST = {
    "model": "gpt-x",
    "messages": MESSAGES,
    "response_format": {
        "type": "json_schema",
        "json_schema": {"name": "triage", "schema": {"type": "object", "properties": PROPERTIES}},
    },
}
TOOL_REQUEST = {
    "model": "gpt-x",
    "messages": MESSAGES,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "triage",
                "parameters": {"type": "object", "properties": PROPERTIES},
            },
        }
    ],
}
UPSTREAM_BODY = json.dumps(
    {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"category": "tech", "urgent": False}),
                }
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }
).encode()


@dataclass(frozen=True)
class FakeVerdict:
    label: int
    confidence: float
    latency_ms: int = 7
    probabilities: tuple[float, ...] = ()
    novelty: float | None = None


class FakeDecider:
    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.calls = []

    def decide(self, model, head_path, text):
        self.calls.append(model.site)
        return self.verdicts[model.site]


class Provider:
    def __init__(self):
        self.calls = 0

    async def handle(self, request: Request):
        self.calls += 1
        return Response(content=UPSTREAM_BODY, media_type="application/json")


def model(field, kind, labels):
    return SiteModel(
        site=f"{SITE}.{field}",
        kind=kind,
        field=field,
        labels=list(labels),
        base_model="b",
        temperature=1.0,
        threshold=0.6,
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


CATEGORY = model("category", "choice", ("billing", "tech"))
URGENT = model("urgent", "boolean", ("false", "true"))
SURE = {f"{SITE}.category": FakeVerdict(1, 0.9), f"{SITE}.urgent": FakeVerdict(1, 0.8)}


@pytest.fixture
async def serving(data_dir):
    apps = []

    def make(provider, verdicts, modes=(MODE_LIVE, MODE_LIVE), **overrides):
        settings = Settings(upstream="http://upstream", **overrides)
        for site_model, mode in zip((CATEGORY, URGENT), modes, strict=True):
            write_mode(save_model(models_path(settings), site_model), mode, now=0.0)
        upstream = Starlette(routes=[Route("/{path:path}", provider.handle, methods=["POST"])])
        app = build_app(
            settings,
            transport=httpx.ASGITransport(app=upstream),
            decider=FakeDecider(verdicts),
        )
        apps.append(app)
        return app

    yield make
    for app in apps:
        await app.state.proxy.aclose()


async def post(app, payload=REQUEST):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        return await client.post(
            "/v1/chat/completions", json=payload, headers={"X-Stuntd-Site": SITE}
        )


def captured(app):
    return {row["site"]: row["count"] for row in app.state.store.stats()}


def decisions(app, field):
    return [
        (row.mode, row.answer, row.agree)
        for row in app.state.store.decisions(f"{SITE}.{field}", 10)
    ]


async def test_multi_field_answers_locally_when_every_field_is_confident(serving):
    provider = Provider()
    app = serving(provider, SURE)
    response = await post(app)
    assert provider.calls == 0
    assert (
        response.json()["choices"][0]["message"]["content"] == '{"category":"tech","urgent":true}'
    )
    assert response.headers["x-stuntd"] == f"live; site={SITE}; confidence=0.80"
    assert captured(app) == {}


async def test_multi_field_local_answer_fills_a_tool_call(serving):
    app = serving(Provider(), SURE)
    response = await post(app, TOOL_REQUEST)
    call = response.json()["choices"][0]["message"]["tool_calls"][0]["function"]
    assert call["name"] == "triage"
    assert call["arguments"] == '{"category":"tech","urgent":true}'


async def test_multi_field_local_answer_records_a_decision_per_field_site(serving):
    app = serving(Provider(), SURE)
    await post(app)
    assert decisions(app, "category") == [("live", "tech", None)]
    assert decisions(app, "urgent") == [("live", "true", None)]


async def test_multi_field_unsure_field_sends_the_request_to_the_provider(serving):
    provider = Provider()
    app = serving(provider, {**SURE, f"{SITE}.urgent": FakeVerdict(1, 0.4)})
    response = await post(app)
    assert provider.calls == 1
    assert response.content == UPSTREAM_BODY
    assert response.headers["x-stuntd"] == f"collect; site={SITE}; reason=low-confidence:urgent"
    assert captured(app) == {f"{SITE}.category": 1, f"{SITE}.urgent": 1}


async def test_multi_field_stops_at_the_first_unsure_field(serving):
    app = serving(Provider(), {**SURE, f"{SITE}.category": FakeVerdict(1, 0.4)})
    await post(app)
    assert app.state.proxy.decider.calls == [f"{SITE}.category"]


@pytest.mark.parametrize(
    ("modes", "field"),
    [((MODE_SHADOW, MODE_LIVE), "category"), ((MODE_LIVE, MODE_SHADOW), "urgent")],
    ids=["first", "second"],
)
async def test_multi_field_field_not_live_sends_the_request_to_the_provider(serving, modes, field):
    provider = Provider()
    app = serving(provider, SURE, modes=modes)
    response = await post(app)
    assert provider.calls == 1
    assert response.headers["x-stuntd"].endswith(f"reason=not-live:{field}")
    assert captured(app) == {f"{SITE}.category": 1, f"{SITE}.urgent": 1}


async def test_multi_field_heads_trained_on_another_encoder_send_the_request_to_the_provider(
    serving,
):
    provider = Provider()
    app = serving(provider, SURE, encoder="other")
    response = await post(app)
    assert provider.calls == 1
    assert response.headers["x-stuntd"] == f"collect; site={SITE}; reason=encoder-mismatch:category"
    assert app.state.proxy.decider.calls == []


async def test_multi_field_shadow_fields_record_how_the_model_compared(serving):
    app = serving(Provider(), SURE, modes=(MODE_SHADOW, MODE_SHADOW))
    response = await post(app)
    assert response.headers["x-stuntd"] == f"shadow; site={SITE}; reason=not-live:category"
    assert decisions(app, "category") == [("shadow", "tech", True)]
    assert decisions(app, "urgent") == [("shadow", "true", False)]


async def test_multi_field_live_field_beside_a_shadow_field_records_a_comparison(serving):
    app = serving(Provider(), SURE, modes=(MODE_LIVE, MODE_SHADOW))
    await post(app)
    assert decisions(app, "category") == [("check", "tech", True)]
    assert decisions(app, "urgent") == [("shadow", "true", False)]


async def test_multi_field_live_field_beside_a_shadow_field_can_be_demoted(serving):
    verdicts = {**SURE, f"{SITE}.category": FakeVerdict(0, 0.9)}
    app = serving(
        Provider(), verdicts, modes=(MODE_LIVE, MODE_SHADOW), min_window=1, target_agreement=0.9
    )
    await post(app)
    assert read_mode(models_path(app.state.settings) / f"{SITE}.category")[0] == MODE_SHADOW


async def test_multi_field_check_compares_every_field(serving):
    provider = Provider()
    app = serving(provider, SURE, check_share=1.0)
    response = await post(app)
    assert provider.calls == 1
    assert response.headers["x-stuntd"] == f"collect; site={SITE}; reason=check:urgent"
    assert decisions(app, "category") == [("check", "tech", True)]
    assert decisions(app, "urgent") == [("check", "true", False)]


async def test_multi_field_demotion_touches_only_the_failing_site(serving):
    app = serving(Provider(), SURE, check_share=1.0, min_window=1, target_agreement=0.9)
    await post(app)
    models = models_path(app.state.settings)
    assert read_mode(models / f"{SITE}.category")[0] == MODE_LIVE
    assert read_mode(models / f"{SITE}.urgent")[0] == MODE_SHADOW
