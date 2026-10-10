import json
import logging
from dataclasses import replace

import httpx
import pytest
from decisions_helpers import (
    CHOICE,
    CHOICE_ANSWER,
    PREDICATE,
    PREDICATE_ANSWER,
    PROVIDER_MODEL,
    REFUSAL_ANSWER,
    SCORE,
    SCORE_ANSWER,
    TEXT,
    FakeVerdict,
    decision_body,
    site_model,
)
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from stuntd.proxy.app import build_app
from stuntd.serve.modes import MODE_LIVE, write_mode
from stuntd.settings import Settings, models_path
from stuntd.train.artifacts import save_model

pytestmark = pytest.mark.anyio

INPUT_TOKENS = 120
OUTPUT_TOKENS = 3

THREE = [PREDICATE, CHOICE, SCORE]
THREE_ANSWERS = [PREDICATE_ANSWER, CHOICE_ANSWER, SCORE_ANSWER]
PROVIDER_BODY = decision_body(THREE_ANSWERS, INPUT_TOKENS, OUTPUT_TOKENS)

DAMAGED_CANONICAL = (
    '{"criteria":null,"instructions":"Is the product visibly damaged?","type":"noul"}'
)
ROUTE_CANONICAL = (
    '{"criteria":{"billing":"Payment questions","other":null,"shipping":null},'
    '"instructions":"Which team should handle this?","type":"choice"}'
)
URGENCY_CANONICAL = (
    '{"criteria":["low","medium: Needs a reply today","high"],'
    '"instructions":"How urgent is this?","type":"score"}'
)


class Provider:
    """Fake Decisions API that counts its calls and keeps what reached it."""

    def __init__(self, body=PROVIDER_BODY, status=200):
        self.body = body
        self.status = status
        self.calls = 0
        self.seen = []

    async def handle(self, request: Request):
        self.calls += 1
        self.seen.append(
            {
                "path": request.url.path,
                "headers": dict(request.headers),
                "body": await request.body(),
            }
        )
        return Response(
            content=self.body,
            status_code=self.status,
            media_type="application/json",
            headers={"x-request-id": "req_1"},
        )


SURE_BILLING = FakeVerdict(0, 0.9, 7, (0.75, 0.15, 0.1))


class FakeDecider:
    def __init__(self, verdict=SURE_BILLING):
        self.verdict = verdict

    def decide(self, model, head_path, text):
        return self.verdict


class RetrainSpy:
    def __init__(self):
        self.recorded = []

    def record(self, site, text):
        self.recorded.append((site, text))


def transport_for(provider):
    app = Starlette(routes=[Route("/{path:path}", provider.handle, methods=["GET", "POST"])])
    return httpx.ASGITransport(app=app)


@pytest.fixture
async def proxying(data_dir):
    apps = []

    def make(provider, *, model=None, mode=None, decider=None, **overrides):
        settings = Settings(upstream="http://openai", **overrides)
        if model is not None:
            write_mode(save_model(models_path(settings), model), mode, now=0.0)
        app = build_app(settings, transport=transport_for(provider), decider=decider)
        apps.append(app)
        return app

    yield make
    for app in apps:
        await app.state.proxy.aclose()


def body_of(questions, *, text=TEXT):
    return json.dumps({"model": "gpt-6-luna", "input": text, "questions": questions}).encode()


async def post(app, questions=None, *, content=None, headers=None):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        return await client.post(
            "/v1/decisions",
            content=body_of(questions) if content is None else content,
            headers={"content-type": "application/json", **(headers or {})},
        )


def captures(app, columns="site, answer"):
    return app.state.store._conn.execute(f"select {columns} from captures order by id").fetchall()


async def test_the_request_and_the_answer_cross_byte_for_byte(proxying):
    provider = Provider()
    app = proxying(provider)
    body = (
        b'{"questions":[{"type":"predicate","instructions":"Damaged?"}],'
        b'"zeta":1,"input":"Hi","model":"gpt-6-luna"}'
    )
    response = await post(app, content=body, headers={"authorization": "Bearer k"})
    assert response.status_code == 200
    assert response.content == PROVIDER_BODY
    assert provider.seen[0]["body"] == body
    assert provider.seen[0]["path"] == "/v1/decisions"
    assert provider.seen[0]["headers"]["host"] == "openai"
    assert provider.seen[0]["headers"]["authorization"] == "Bearer k"
    assert response.headers["x-request-id"] == "req_1"
    assert response.headers["x-stuntd"] == (
        "jev; mode=proxy; questions=1; live=0; shadow=0; check=0"
    )


async def test_the_caller_never_sends_its_own_stuntd_headers_upstream(proxying):
    provider = Provider()
    await post(proxying(provider), THREE, headers={"x-stuntd-site": "tone"})
    assert [name for name in provider.seen[0]["headers"] if name.startswith("x-stuntd")] == []


async def test_provider_answers_become_captures_under_the_translated_canonical(proxying):
    app = proxying(Provider())
    response = await post(app, THREE)
    assert response.status_code == 200
    assert captures(
        app,
        "site, schema_canonical, kind, input_text, answer, model, prompt_tokens, completion_tokens",
    ) == [
        ("damaged", DAMAGED_CANONICAL, "boolean", TEXT, "true", PROVIDER_MODEL, 120, 3),
        ("route", ROUTE_CANONICAL, "choice", TEXT, "billing", PROVIDER_MODEL, 120, 3),
        ("urgency", URGENCY_CANONICAL, "number", TEXT, "2", PROVIDER_MODEL, 120, 3),
    ]


async def test_a_refusal_is_relayed_and_never_recorded(proxying):
    provider = Provider(decision_body([REFUSAL_ANSWER, CHOICE_ANSWER]))
    app = proxying(provider)
    response = await post(app, [PREDICATE, CHOICE])
    assert response.content == provider.body
    assert captures(app) == [("route", "billing")]


async def test_a_choice_the_question_never_offered_is_not_learned_from(proxying, caplog):
    stray = dict(CHOICE_ANSWER, choice="refunds")
    app = proxying(Provider(decision_body([stray])))
    with caplog.at_level(logging.WARNING):
        await post(app, [CHOICE])
    assert captures(app) == []
    assert "no usable provider answer for question 'route'" in caplog.text


async def test_an_answer_list_of_the_wrong_length_records_nothing(proxying, caplog):
    app = proxying(Provider(decision_body([PREDICATE_ANSWER])))
    with caplog.at_level(logging.ERROR):
        response = await post(app, [PREDICATE, CHOICE])
    assert response.status_code == 200
    assert captures(app) == []
    assert "the provider answered without answers" in caplog.text


IMAGE_INPUT = [
    {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ],
    }
]


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "gpt-6-luna", "input": IMAGE_INPUT, "questions": [PREDICATE]},
        {
            "model": "gpt-6-luna",
            "input": TEXT,
            "questions": [dict(CHOICE, choices=[{"value": "true"}, {"value": True}])],
        },
        {
            "model": "gpt-6-luna",
            "input": TEXT,
            "questions": [dict(SCORE, levels=[{"label": str(n)} for n in range(11)])],
        },
        {"model": "gpt-6-luna", "input": TEXT, "questions": [PREDICATE, PREDICATE]},
        {"model": "gpt-6-luna", "input": TEXT, "questions": [dict(PREDICATE, type="ranking")]},
        {"model": "gpt-6-luna", "input": "a\ud800b", "questions": [PREDICATE]},
        {
            "model": "gpt-6-luna",
            "input": TEXT,
            "questions": [dict(CHOICE, choices=[{"value": "a\ud800"}, {"value": "b"}])],
        },
        {
            "model": "gpt-6-luna",
            "input": TEXT,
            "questions": [dict(CHOICE, choices=[{"value": "billing"}])],
        },
    ],
    ids=[
        "image",
        "colliding-values",
        "eleven-levels",
        "duplicate-names",
        "unknown-type",
        "lone-surrogate-input",
        "lone-surrogate-choice",
        "one-choice",
    ],
)
async def test_a_request_stuntd_cannot_learn_from_is_relayed_and_not_recorded(proxying, payload):
    provider = Provider(decision_body([PREDICATE_ANSWER]))
    app = proxying(provider)
    content = json.dumps(payload).encode()
    response = await post(app, content=content)
    assert response.content == provider.body
    assert provider.seen[0]["body"] == content
    assert response.headers["x-stuntd"] == "jev; mode=proxy; reason=not-parsed"
    assert captures(app) == []


async def test_a_deeply_nested_body_is_relayed_and_not_recorded(proxying):
    provider = Provider(b'{"error":{"message":"bad"}}', status=400)
    app = proxying(provider)
    content = b"[" * 200_000
    response = await post(app, content=content)
    assert response.status_code == 400
    assert provider.seen[0]["body"] == content
    assert captures(app) == []


async def test_junk_is_relayed_to_the_provider(proxying):
    provider = Provider(b'{"error":{"message":"bad"}}', status=400)
    response = await post(proxying(provider), content=b"junk")
    assert response.status_code == 400
    assert provider.seen[0]["body"] == b"junk"


async def test_a_provider_error_is_relayed_without_a_capture(proxying):
    provider = Provider(b'{"error":{"message":"rate"}}', status=429)
    app = proxying(provider)
    response = await post(app, THREE)
    assert (response.status_code, response.content) == (429, provider.body)
    assert captures(app) == []


async def test_confident_heads_answer_the_whole_request_without_the_provider(proxying):
    provider = Provider()
    app = proxying(provider, model=site_model("route"), mode=MODE_LIVE, decider=FakeDecider())
    response = await post(app, [CHOICE])
    assert provider.calls == 0
    answer = json.loads(response.content)["answers"][0]
    assert answer["choice"] == "billing"
    assert [item["value"] for item in answer["probabilities"]] == ["billing", "shipping", "other"]
    assert response.headers["x-stuntd"] == (
        "jev; mode=proxy; questions=1; live=1; shadow=0; check=0"
    )
    assert captures(app) == []
    rows = app.state.store.decisions("route", 10)
    assert [(row.mode, row.answer) for row in rows] == [("live", "billing")]


async def test_one_question_the_heads_cannot_answer_sends_the_whole_request_out(proxying):
    provider = Provider(decision_body([CHOICE_ANSWER, PREDICATE_ANSWER]))
    app = proxying(provider, model=site_model("route"), mode=MODE_LIVE, decider=FakeDecider())
    response = await post(app, [CHOICE, PREDICATE])
    assert provider.calls == 1
    assert response.content == provider.body
    assert captures(app) == [("route", "billing"), ("damaged", "true")]


async def test_a_head_trained_on_another_encoder_sends_the_request_out(proxying):
    provider = Provider(decision_body([CHOICE_ANSWER]))
    app = proxying(
        provider,
        model=replace(site_model("route"), encoder="other"),
        mode=MODE_LIVE,
        decider=FakeDecider(),
    )
    response = await post(app, [CHOICE])
    assert provider.calls == 1
    assert response.headers["x-stuntd"] == (
        "jev; mode=proxy; questions=1; live=0; shadow=0; check=0; reason=encoder-mismatch"
    )
    assert captures(app) == [("route", "billing")]
    assert app.state.store.decisions("route", 10) == []


async def test_a_head_below_its_threshold_sends_the_request_out(proxying):
    provider = Provider()
    app = proxying(
        provider,
        model=site_model("route"),
        mode=MODE_LIVE,
        decider=FakeDecider(FakeVerdict(0, 0.4, 7, (0.4, 0.3, 0.3))),
    )
    await post(app, [CHOICE])
    assert provider.calls == 1


async def test_decisions_captures_count_toward_auto_retrain(proxying):
    app = proxying(Provider())
    spy = RetrainSpy()
    app.state.proxy.retrainer = spy
    await post(app, THREE)
    assert spy.recorded == [("damaged", TEXT), ("route", TEXT), ("urgency", TEXT)]


async def test_learning_turned_off_relays_and_records_nothing(proxying):
    provider = Provider()
    app = proxying(provider, learn=False)
    response = await post(app, THREE)
    assert response.content == PROVIDER_BODY
    assert response.headers["x-stuntd"] == (
        "jev; mode=proxy; questions=3; live=0; shadow=0; check=0; learn=off"
    )
    assert app.state.store is None
