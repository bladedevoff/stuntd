import json
from dataclasses import replace

import httpx
import pytest
from decisions_helpers import (
    CHOICE,
    CHOICE_ANSWER,
    PREDICATE,
    SCORE,
    TEXT,
    UNNAMED_FLAG,
    FakeVerdict,
    site_model,
    usage,
)

from stuntd.proxy.app import build_app
from stuntd.serve.modes import MODE_LIVE, write_mode
from stuntd.settings import Settings, models_path
from stuntd.train.artifacts import save_model

pytestmark = pytest.mark.anyio

INPUT_TOKENS = 11

ZERO_SHOT = {
    "#0": {"type": "noul", "noul": 0.8},
    "damaged": {"type": "noul", "noul": 0.8},
    "route": {
        "type": "choice",
        "choice": "billing",
        "confidence": 0.6,
        "probabilities": {"billing": 0.6, "shipping": 0.3, "other": 0.1},
    },
    "urgency": {
        "type": "score",
        "score": 1.4,
        "confidence": 0.5,
        "legend": {"0": "low", "1": "medium: Needs a reply today", "2": "high"},
        "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
    },
    "flag": {
        "type": "choice",
        "choice": "true",
        "confidence": 0.7,
        "probabilities": {"true": 0.7, "false": 0.3},
    },
}

UNNAMED_PREDICATE = {"type": "predicate", "instructions": "Is the product visibly damaged?"}
FLAG = dict(UNNAMED_FLAG, name="flag")


SURE_TRUE = FakeVerdict(1, 0.9, 7, (0.25, 0.75))


class FakeDecider:
    def __init__(self, verdict=SURE_TRUE, error=None):
        self.verdict = verdict
        self.error = error
        self.batches = []

    def decide(self, model, head_path, text):
        return self.verdict

    def answer(self, state, questions):
        self.batches.append((state, questions))
        if self.error is not None:
            raise self.error
        return {
            "answers": {name: ZERO_SHOT[name] for name in questions},
            "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": 0},
        }


@pytest.fixture
async def serving(data_dir):
    apps = []

    def make(decider=None, *, zero_shot=None, model=None, mode=None, **overrides):
        settings = Settings(**overrides)
        if model is not None:
            write_mode(save_model(models_path(settings), model), mode, now=0.0)
        app = build_app(settings, decider=decider, zero_shot=zero_shot or decider)
        apps.append(app)
        return app

    yield make
    for app in apps:
        await app.state.proxy.aclose()


async def post(app, questions, *, text=TEXT, headers=None):
    payload = {"model": "gpt-6-luna", "input": text, "questions": questions}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        return await client.post("/v1/decisions", json=payload, headers=headers)


async def test_every_question_type_is_answered_zero_shot_in_order(serving):
    decider = FakeDecider()
    app = serving(decider)
    response = await post(app, [UNNAMED_PREDICATE, FLAG, SCORE, CHOICE])
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert json.loads(response.content) == {
        "answers": [
            {"type": "predicate", "name": None, "probability": 0.8},
            {
                "type": "choice",
                "name": "flag",
                "choice": True,
                "confidence": 0.7,
                "probabilities": [
                    {"value": True, "probability": 0.7},
                    {"value": False, "probability": 0.3},
                ],
            },
            {
                "type": "score",
                "name": "urgency",
                "score": 1.4,
                "confidence": 0.5,
                "probabilities": [
                    {"value": 0, "label": "low", "probability": 0.1},
                    {"value": 1, "label": "medium", "probability": 0.4},
                    {"value": 2, "label": "high", "probability": 0.5},
                ],
            },
            CHOICE_ANSWER,
        ],
        "model": "stuntd",
        "usage": usage(INPUT_TOKENS),
    }
    assert response.headers["x-stuntd"] == "jev; mode=local; questions=4; live=0; zeroshot=4"
    assert len(response.headers["x-typesafe-request-id"]) == 12
    state, questions = decider.batches[0]
    assert state == TEXT
    assert questions["#0"] == {
        "type": "noul",
        "instructions": "Is the product visibly damaged?",
    }
    assert questions["flag"]["criteria"] == {"true": None, "false": None}


async def test_the_model_name_setting_names_the_model(serving):
    app = serving(FakeDecider(), jev_model_name="stuntd-local")
    response = await post(app, [PREDICATE])
    assert json.loads(response.content)["model"] == "stuntd-local"


async def test_a_live_head_answers_its_question_in_the_typed_values(serving):
    decider = FakeDecider()
    app = serving(decider, model=site_model("flag", labels=("false", "true")), mode=MODE_LIVE)
    response = await post(app, [FLAG, SCORE])
    answers = json.loads(response.content)["answers"]
    assert answers[0] == {
        "type": "choice",
        "name": "flag",
        "choice": True,
        "confidence": 0.9,
        "probabilities": [
            {"value": True, "probability": 0.75},
            {"value": False, "probability": 0.25},
        ],
    }
    assert answers[1]["name"] == "urgency"
    assert list(decider.batches[0][1]) == ["urgency"]
    assert response.headers["x-stuntd"] == "jev; mode=local; questions=2; live=1; zeroshot=1"
    rows = app.state.store.decisions("flag", 10)
    assert [(row.mode, row.answer) for row in rows] == [("live", "true")]


async def test_a_head_trained_on_another_encoder_answers_zero_shot(serving):
    decider = FakeDecider()
    stock = replace(site_model("flag", labels=("false", "true")), encoder="other")
    app = serving(decider, model=stock, mode=MODE_LIVE)
    response = await post(app, [FLAG])
    assert json.loads(response.content)["answers"][0]["confidence"] == 0.7
    assert list(decider.batches[0][1]) == ["flag"]
    assert response.headers["x-stuntd"] == (
        "jev; mode=local; questions=1; live=0; zeroshot=1; reason=encoder-mismatch"
    )
    assert app.state.store.decisions("flag", 10) == []


IMAGE = {
    "role": "user",
    "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}],
}


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "gpt-6-luna", "input": [IMAGE], "questions": [PREDICATE]},
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
        {"model": "gpt-6-luna", "input": "a\ud800b", "questions": [PREDICATE]},
        {
            "model": "gpt-6-luna",
            "input": TEXT,
            "questions": [dict(CHOICE, choices=[{"value": "billing"}])],
        },
    ],
    ids=["image", "colliding-values", "eleven-levels", "lone-surrogate", "one-choice"],
)
async def test_a_request_that_cannot_be_learned_gets_an_openai_400(serving, payload):
    decider = FakeDecider()
    app = serving(decider)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        response = await client.post("/v1/decisions", content=json.dumps(payload).encode())
    assert response.status_code == 400
    error = json.loads(response.content)["error"]
    assert set(error) == {"message", "type", "param", "code"}
    assert (error["type"], error["param"], error["code"]) == ("invalid_request_error", None, None)
    assert response.headers["x-stuntd"] == "jev; mode=local; reason=bad-request"
    assert decider.batches == []


async def test_a_required_key_is_asked_for(serving):
    app = serving(FakeDecider(), jev_require_key=True)
    refused = await post(app, [PREDICATE])
    allowed = await post(app, [PREDICATE], headers={"Authorization": "Bearer k"})
    assert refused.status_code == 401
    assert json.loads(refused.content)["error"]["message"] == "missing API key"
    assert allowed.status_code == 200


async def test_without_a_decider_the_route_says_so(serving):
    response = await post(serving(), [PREDICATE])
    assert response.status_code == 503
    assert json.loads(response.content)["error"]["message"] == "serving needs the train extra"
    assert response.headers["x-stuntd"] == "jev; mode=local; reason=no-runtime"


async def test_a_decider_that_fails_is_reported(serving):
    app = serving(FakeDecider(error=RuntimeError("boom")))
    response = await post(app, [PREDICATE])
    assert response.status_code == 503
    assert json.loads(response.content)["error"]["message"] == "the model could not answer"
    assert response.headers["x-stuntd"] == "jev; mode=local; reason=model-error"
