import json
import logging
from dataclasses import dataclass

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from stuntd.cli import main
from stuntd.proxy.app import build_app
from stuntd.serve.modes import MODE_LIVE, MODE_SHADOW, read_mode, write_mode
from stuntd.settings import Settings, config_path, load_settings, models_path
from stuntd.train.artifacts import SiteModel, load_model, save_model, site_dir
from stuntd.train.layout import Layout
from stuntd.train.metrics import ClassStats
from stuntd.train.run import TrainedHead

pytestmark = pytest.mark.anyio

UPSTREAM = "http://jev-provider"

STATE = "Hi, I was double charged."

TONE = {
    "type": "choice",
    "instructions": "What is the tone?",
    "criteria": {"calm": None, "angry": None},
}
BILLING = {"type": "noul"}
URGENCY = {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]}

TONE_CANONICAL = (
    '{"criteria":{"angry":null,"calm":null},"instructions":"What is the tone?","type":"choice"}'
)
BILLING_CANONICAL = '{"criteria":null,"instructions":null,"type":"noul"}'
URGENCY_CANONICAL = '{"criteria":["low","mid","high"],"instructions":"How urgent?","type":"score"}'

CHOICE_ANSWER = {
    "type": "choice",
    "choice": "angry",
    "confidence": 0.7,
    "probabilities": {"calm": 0.3, "angry": 0.7},
}
NOUL_ANSWER = {"type": "noul", "noul": 0.8}
SCORE_ANSWER = {
    "type": "score",
    "score": 1.4,
    "confidence": 0.5,
    "legend": {"0": "low", "1": "mid", "2": "high"},
    "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
}

PROVIDER_MODEL = "jev-1"
INPUT_TOKENS = 120
OUTPUT_TOKENS = 12


def provider_body(answers):
    body = {
        "model": PROVIDER_MODEL,
        "answers": answers,
        "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS},
    }
    return json.dumps(body).encode()


MODELS_BODY = json.dumps(
    {
        "models": [
            {"name": "jev-1", "description": "The provider's own.", "release_date": "2026-09-15"}
        ]
    }
).encode()


PROVIDER_BODY = provider_body(
    {"tone": CHOICE_ANSWER, "billing": NOUL_ANSWER, "urgency": SCORE_ANSWER}
)


@dataclass(frozen=True)
class FakeVerdict:
    label: int
    confidence: float
    latency_ms: int
    probabilities: tuple[float, ...]


SURE_ANGRY = FakeVerdict(1, 0.9, 7, (0.25, 0.75))


class FakeDecider:
    def __init__(self, verdict=SURE_ANGRY):
        self.verdict = verdict

    def decide(self, model, head_path, text):
        return self.verdict


class Provider:
    """Fake Jev API that counts its calls and keeps what reached it."""

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
        response = Response(
            content=self.body,
            status_code=self.status,
            media_type="application/json",
            headers={"x-jev": "1"},
        )
        response.raw_headers.append((b"set-cookie", b"a=1"))
        response.raw_headers.append((b"set-cookie", b"b=2"))
        return response


class Unreachable(httpx.AsyncBaseTransport):
    """A provider the daemon cannot connect to."""

    async def handle_async_request(self, request):
        raise httpx.ConnectError("no route to the provider")


def transport_for(provider):
    app = Starlette(routes=[Route("/{path:path}", provider.handle, methods=["GET", "POST"])])
    return httpx.ASGITransport(app=app)


def site_model(site, kind="choice", field="tone", labels=("calm", "angry"), threshold=0.6):
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
async def proxying(data_dir):
    apps = []

    def make(transport, *, model=None, mode=None, decider=None, **overrides):
        overrides.setdefault("jev_upstream", UPSTREAM)
        settings = Settings(upstream="http://upstream", **overrides)
        if model is not None:
            write_mode(save_model(models_path(settings), model), mode, now=0.0)
        app = build_app(settings, transport=transport, decider=decider)
        apps.append(app)
        return app

    yield make
    for app in apps:
        await app.state.proxy.aclose()


async def models(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        return await client.get("/v1/models")


async def post(app, questions=None, *, state=STATE, headers=None, content=None):
    if content is None:
        payload = {"state": state, "model": "jev-latest", "questions": questions}
        content = json.dumps(payload).encode()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        return await client.post(
            "/v1/systemone",
            content=content,
            headers={"content-type": "application/json", **(headers or {})},
        )


def captures(
    app,
    columns="site, schema_canonical, kind, input_text, answer, model,"
    " prompt_tokens, completion_tokens",
):
    return app.state.store._conn.execute(f"select {columns} from captures order by id").fetchall()


async def test_the_request_and_the_answer_cross_byte_for_byte(proxying):
    provider = Provider()
    app = proxying(transport_for(provider))
    body = b'{"model":"jev-latest","zeta":1,"state":"Hi","questions":{"billing":{"type":"noul"}}}'
    response = await post(
        app, content=body, headers={"authorization": "Bearer k", "x-stuntd-site": "tone"}
    )
    assert response.status_code == 200
    assert response.content == PROVIDER_BODY
    assert provider.seen[0]["body"] == body
    assert provider.seen[0]["path"] == "/v1/systemone"
    assert provider.seen[0]["headers"]["host"] == "jev-provider"
    assert provider.seen[0]["headers"]["authorization"] == "Bearer k"
    assert [name for name in provider.seen[0]["headers"] if name.startswith("x-stuntd")] == []
    assert response.headers["x-jev"] == "1"
    assert response.headers.get_list("set-cookie") == ["a=1", "b=2"]
    assert response.headers["x-stuntd"] == (
        "jev; mode=proxy; questions=1; live=0; shadow=0; check=0"
    )


@pytest.mark.parametrize(
    ("name", "question", "answer", "canonical", "kind", "label"),
    [
        ("tone", TONE, CHOICE_ANSWER, TONE_CANONICAL, "choice", "angry"),
        ("billing", BILLING, NOUL_ANSWER, BILLING_CANONICAL, "boolean", "true"),
        ("urgency", URGENCY, SCORE_ANSWER, URGENCY_CANONICAL, "number", "2"),
    ],
    ids=["choice", "noul", "score"],
)
async def test_a_provider_answer_becomes_a_capture(
    proxying, name, question, answer, canonical, kind, label
):
    provider = Provider(body=provider_body({name: answer}))
    app = proxying(transport_for(provider))
    response = await post(app, {name: question})
    assert response.status_code == 200
    assert captures(app) == [
        (name, canonical, kind, STATE, label, PROVIDER_MODEL, INPUT_TOKENS, OUTPUT_TOKENS)
    ]


async def test_confident_heads_answer_the_whole_request_without_the_provider(proxying):
    provider = Provider()
    app = proxying(
        transport_for(provider), model=site_model("tone"), mode=MODE_LIVE, decider=FakeDecider()
    )
    response = await post(app, {"tone": TONE})
    assert provider.calls == 0
    assert json.loads(response.content) == {
        "model": "stuntd",
        "answers": {
            "tone": {
                "type": "choice",
                "choice": "angry",
                "confidence": 0.9,
                "probabilities": {"calm": 0.25, "angry": 0.75},
            }
        },
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    assert response.headers["x-stuntd"] == (
        "jev; mode=proxy; questions=1; live=1; shadow=0; check=0"
    )
    assert len(response.headers["x-typesafe-request-id"]) == 12
    rows = app.state.store.decisions("tone", 10)
    assert [(row.mode, row.answer, row.agree) for row in rows] == [("live", "angry", None)]
    assert captures(app) == []


async def test_one_question_the_heads_cannot_answer_sends_the_whole_request_out(proxying):
    provider = Provider()
    app = proxying(
        transport_for(provider),
        model=site_model("tone"),
        mode=MODE_LIVE,
        decider=FakeDecider(),
        check_share=0.9999,
    )
    response = await post(app, {"tone": TONE, "billing": BILLING})
    assert provider.calls == 1
    assert response.content == PROVIDER_BODY
    assert response.headers["x-stuntd"] == (
        "jev; mode=proxy; questions=2; live=0; shadow=0; check=1"
    )
    rows = app.state.store.decisions("tone", 10)
    assert [(row.mode, row.answer, row.agree) for row in rows] == [("check", "angry", True)]
    assert captures(app, "site, answer") == [("tone", "angry"), ("billing", "true")]


async def test_a_shadow_site_compares_its_head_with_the_provider_answer(proxying):
    provider = Provider()
    app = proxying(
        transport_for(provider), model=site_model("tone"), mode=MODE_SHADOW, decider=FakeDecider()
    )
    response = await post(app, {"tone": TONE})
    assert provider.calls == 1
    assert response.content == PROVIDER_BODY
    assert response.headers["x-stuntd"] == (
        "jev; mode=proxy; questions=1; live=0; shadow=1; check=0"
    )
    rows = app.state.store.decisions("tone", 10)
    assert [(row.mode, row.answer, row.agree) for row in rows] == [("shadow", "angry", True)]
    assert captures(app, "site, answer") == [("tone", "angry")]


async def test_an_unreachable_provider_is_reported(proxying):
    app = proxying(Unreachable())
    response = await post(app, {"billing": BILLING})
    assert response.status_code == 502
    assert json.loads(response.content) == {"error": {"message": "upstream unreachable"}}
    assert response.headers["x-stuntd"] == "jev; mode=proxy; reason=upstream-error"
    assert captures(app) == []


async def test_a_non_200_answer_is_relayed_without_a_capture(proxying):
    body = provider_body({"billing": NOUL_ANSWER})
    provider = Provider(body=body, status=429)
    app = proxying(transport_for(provider))
    response = await post(app, {"billing": BILLING})
    assert response.status_code == 429
    assert response.content == body
    assert response.headers["x-stuntd"] == (
        "jev; mode=proxy; questions=1; live=0; shadow=0; check=0"
    )
    assert captures(app) == []


async def test_learning_turned_off_relays_and_records_nothing(proxying):
    provider = Provider()
    app = proxying(transport_for(provider), learn=False)
    response = await post(app, {"billing": BILLING})
    assert provider.calls == 1
    assert response.content == PROVIDER_BODY
    assert app.state.store is None


@pytest.mark.parametrize(
    "answers",
    [{"billing": NOUL_ANSWER}, {"billing": NOUL_ANSWER, "tone": {"type": "choice"}}],
    ids=["missing", "malformed"],
)
async def test_an_answer_the_provider_did_not_give_skips_only_that_question(
    proxying, caplog, answers
):
    provider = Provider(body=provider_body(answers))
    app = proxying(transport_for(provider))
    with caplog.at_level(logging.WARNING):
        response = await post(app, {"tone": TONE, "billing": BILLING})
    assert response.status_code == 200
    assert captures(app, "site, answer") == [("billing", "true")]
    assert "no usable provider answer for question 'tone'" in caplog.text


async def test_a_body_the_daemon_cannot_read_is_still_relayed(proxying, caplog):
    provider = Provider(body=b'{"detail":[]}', status=422)
    app = proxying(transport_for(provider))
    with caplog.at_level(logging.WARNING):
        response = await post(app, content=b"{}")
    assert response.status_code == 422
    assert provider.seen[0]["body"] == b"{}"
    assert response.headers["x-stuntd"] == "jev; mode=proxy; reason=not-parsed"
    assert "the request was not parsed: request must contain state" in caplog.text
    assert captures(app) == []


async def test_a_choice_the_question_never_offered_is_not_learned_from(proxying, caplog):
    answer = dict(CHOICE_ANSWER, choice="furious")
    provider = Provider(body=provider_body({"tone": answer, "billing": NOUL_ANSWER}))
    app = proxying(
        transport_for(provider),
        model=site_model("tone"),
        mode=MODE_LIVE,
        decider=FakeDecider(),
        check_share=0.9999,
    )
    with caplog.at_level(logging.WARNING):
        response = await post(app, {"tone": TONE, "billing": BILLING})
    assert response.status_code == 200
    assert captures(app, "site, answer") == [("billing", "true")]
    assert app.state.store.decisions("tone", 10) == []
    assert "no usable provider answer for question 'tone'" in caplog.text


async def test_a_required_key_is_left_to_the_provider(proxying):
    provider = Provider()
    app = proxying(transport_for(provider), jev_require_key=True)
    response = await post(app, {"billing": BILLING})
    assert response.status_code == 200
    assert provider.calls == 1


async def test_the_model_list_is_relayed_to_the_provider(proxying):
    provider = Provider(body=MODELS_BODY)
    app = proxying(transport_for(provider))
    response = await models(app)
    assert response.status_code == 200
    assert response.content == MODELS_BODY
    assert provider.seen[0]["path"] == "/v1/models"
    assert provider.seen[0]["headers"]["host"] == "jev-provider"
    assert response.headers["x-jev"] == "1"
    assert response.headers["x-stuntd"] == "jev; mode=proxy"


async def test_the_model_list_stays_local_without_a_provider(proxying):
    provider = Provider(body=MODELS_BODY)
    app = proxying(transport_for(provider), jev_upstream="")
    response = await models(app)
    assert response.status_code == 200
    assert provider.calls == 0
    assert [model["name"] for model in json.loads(response.content)["models"]] == ["stuntd"]


COLLECTED = 40


def write_config(data_dir, checkpoint=None):
    data_dir.mkdir(parents=True, exist_ok=True)
    training = "[training]\nmin_examples = 10\nholdout = 0.25\n"
    if checkpoint is not None:
        training += (
            f'base_model = "{checkpoint.replace(chr(92), "/")}"\ndevice = "cpu"\nepochs = 1\n'
        )
    (data_dir / "stuntd.toml").write_text(
        'upstream = "http://upstream"\n'
        + training
        + "[serving]\ncheck_share = 0.0\n"
        + f'[jev]\nupstream = "{UPSTREAM}"\n',
        encoding="utf-8",
    )


class ToneProvider:
    """Fake Jev API whose tone answer follows the state it was given."""

    def __init__(self):
        self.calls = 0

    async def handle(self, request: Request):
        self.calls += 1
        payload = json.loads(await request.body())
        angry = "charged twice" in payload["state"]
        answer = {
            "type": "choice",
            "choice": "angry" if angry else "calm",
            "confidence": 0.9,
            "probabilities": {"calm": 0.1, "angry": 0.9} if angry else {"calm": 0.9, "angry": 0.1},
        }
        return Response(content=provider_body({"tone": answer}), media_type="application/json")


def perfect_trainer_factory(settings):
    def trainer(dataset, head_path):
        head_path.write_bytes(b"head")
        return TrainedHead(
            [[3.0, 0.0] if item.label == 0 else [0.0, 3.0] for item in dataset.holdout],
            Layout(512, 192, spaced_labels=True),
        )

    return trainer


def app_for(provider, decider=None):
    return build_app(
        load_settings(config_path()), transport=transport_for(provider), decider=decider
    )


def collected_state(index):
    invoice = 1000 + index
    if index % 2:
        return f"I was charged twice on invoice {invoice}."
    return f"Where do I find invoice {invoice}?"


async def collect(provider):
    collecting = app_for(provider)
    for index in range(COLLECTED):
        relayed = await post(collecting, {"tone": TONE}, state=collected_state(index))
        assert relayed.headers["x-stuntd"] == (
            "jev; mode=proxy; questions=1; live=0; shadow=0; check=0"
        )
    assert provider.calls == COLLECTED
    await collecting.state.proxy.aclose()


async def test_proxied_answers_train_a_site_that_then_answers_locally(
    data_dir, monkeypatch, capsys
):
    write_config(data_dir)
    provider = ToneProvider()
    await collect(provider)

    monkeypatch.setattr("stuntd.cli._make_trainer", perfect_trainer_factory)
    assert main(["train"]) == 0
    assert "tone  trained" in capsys.readouterr().out
    models = models_path(load_settings(config_path()))
    model = load_model(models, "tone")
    assert model.labels == ["angry", "calm"] and model.threshold is not None
    assert read_mode(site_dir(models, "tone"))[0] == MODE_SHADOW

    monkeypatch.setattr("stuntd.cli._require_serving", lambda: None)
    assert main(["enable", "tone"]) == 0
    assert capsys.readouterr().out.strip() == f"tone: {MODE_LIVE}"

    serving = app_for(provider, FakeDecider(FakeVerdict(0, 1.0, 5, (1.0, 0.0))))
    try:
        answered = await post(serving, {"tone": TONE}, state=collected_state(41))
        assert provider.calls == COLLECTED
        assert answered.headers["x-stuntd"] == (
            "jev; mode=proxy; questions=1; live=1; shadow=0; check=0"
        )
        assert json.loads(answered.content)["answers"]["tone"]["choice"] == "angry"
        rows = serving.state.store.decisions("tone", 10)
        assert [(row.mode, row.answer) for row in rows] == [("live", "angry")]
    finally:
        await serving.state.proxy.aclose()


@pytest.mark.slow
async def test_a_jev_site_trains_on_the_real_checkpoint_and_answers_live(
    data_dir, laya_checkpoint, capsys
):
    from stuntd.serve.decider import Decider

    write_config(data_dir, checkpoint=laya_checkpoint)
    provider = ToneProvider()
    await collect(provider)

    assert main(["train"]) == 0
    assert "tone  trained" in capsys.readouterr().out
    models = models_path(load_settings(config_path()))
    assert load_model(models, "tone").labels == ["angry", "calm"]
    assert main(["enable", "tone"]) == 0
    assert capsys.readouterr().out.strip() == f"tone: {MODE_LIVE}"

    serving = app_for(provider, Decider(laya_checkpoint, device="cpu"))
    try:
        answered = await post(serving, {"tone": TONE}, state=collected_state(41))
        assert provider.calls == COLLECTED
        assert answered.headers["x-stuntd"] == (
            "jev; mode=proxy; questions=1; live=1; shadow=0; check=0"
        )
        assert json.loads(answered.content)["answers"]["tone"]["choice"] == "angry"
    finally:
        await serving.state.proxy.aclose()
