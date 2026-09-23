import json
import logging
from dataclasses import dataclass

import httpx
import pytest

from stuntd.proxy.app import build_app
from stuntd.serve.modes import MODE_LIVE, MODE_SHADOW, write_mode
from stuntd.settings import Settings, models_path
from stuntd.train.artifacts import SiteModel, save_model
from stuntd.train.metrics import ClassStats

pytestmark = pytest.mark.anyio

STATE = "Hi, I was double charged."

TONE = {
    "type": "choice",
    "instructions": "What is the tone?",
    "criteria": {"calm": None, "angry": None},
}
BILLING = {"type": "noul"}
URGENCY = {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]}

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
ANSWERS = {"tone": CHOICE_ANSWER, "billing": NOUL_ANSWER, "urgency": SCORE_ANSWER}

INPUT_TOKENS = 11


@dataclass(frozen=True)
class FakeVerdict:
    label: int
    confidence: float
    latency_ms: int
    probabilities: tuple[float, ...]


SURE_ANGRY = FakeVerdict(1, 0.9, 7, (0.25, 0.75))


class FakeDecider:
    def __init__(self, answers=ANSWERS, verdict=SURE_ANGRY, error=None):
        self.answers = answers
        self.verdict = verdict
        self.error = error
        self.batches = []

    def decide(self, model, head_path, text):
        return self.verdict

    def answer(self, state, questions):
        self.batches.append(questions)
        if self.error is not None:
            raise self.error
        return {
            "answers": {name: self.answers[name] for name in questions if name in self.answers},
            "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": 0},
        }


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
async def serving(data_dir):
    apps = []

    def make(decider=None, *, model=None, mode=None, upstream="http://upstream", **overrides):
        settings = Settings(upstream=upstream, **overrides)
        if model is not None:
            write_mode(save_model(models_path(settings), model), mode, now=0.0)
        app = build_app(settings, decider=decider)
        apps.append(app)
        return app

    yield make
    for app in apps:
        await app.state.proxy.aclose()


async def post(app, questions, *, state=STATE, headers=None):
    payload = {"state": state, "model": "jev-latest", "questions": questions}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        return await client.post("/v1/systemone", json=payload, headers=headers)


@pytest.mark.parametrize(
    ("headers", "status"),
    [(None, 401), ({"Authorization": "Bearer k"}, 200)],
    ids=["without-key", "with-key"],
)
async def test_a_required_key_is_asked_for(serving, headers, status):
    app = serving(FakeDecider(), jev_require_key=True)
    response = await post(app, {"billing": BILLING}, headers=headers)
    assert response.status_code == status
    if status == 401:
        assert json.loads(response.content) == {"error": {"message": "missing API key"}}
        assert response.headers["x-stuntd"] == "jev; mode=local; reason=no-key"


async def test_a_malformed_request_is_refused(serving):
    app = serving(FakeDecider())
    response = await post(app, {"tone": {"type": "poll"}})
    assert response.status_code == 400
    assert json.loads(response.content) == {
        "error": {"message": 'Question "tone" must have a type of choice, noul, or score'}
    }
    assert response.headers["x-stuntd"] == "jev; mode=local; reason=bad-request"


async def test_without_a_decider_the_route_says_so(serving):
    app = serving()
    response = await post(app, {"billing": BILLING})
    assert response.status_code == 503
    assert json.loads(response.content) == {"error": {"message": "serving needs the train extra"}}
    assert response.headers["x-stuntd"] == "jev; mode=local; reason=no-runtime"


@pytest.mark.parametrize(
    ("name", "question", "answer"),
    [
        ("tone", TONE, CHOICE_ANSWER),
        ("billing", BILLING, NOUL_ANSWER),
        ("urgency", URGENCY, SCORE_ANSWER),
    ],
    ids=["choice", "noul", "score"],
)
async def test_a_question_is_answered_zero_shot(serving, name, question, answer):
    app = serving(FakeDecider())
    response = await post(app, {name: question})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert json.loads(response.content) == {
        "model": "stuntd",
        "answers": {name: answer},
        "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": 0},
    }
    assert response.headers["x-stuntd"] == "jev; mode=local; questions=1; live=0; zeroshot=1"


async def test_a_question_is_answered_without_an_openai_upstream(serving):
    app = serving(FakeDecider(), upstream="")
    response = await post(app, {"tone": TONE})
    assert response.status_code == 200
    assert json.loads(response.content)["answers"] == {"tone": CHOICE_ANSWER}
    assert response.headers["x-stuntd"] == "jev; mode=local; questions=1; live=0; zeroshot=1"


async def test_a_live_site_answers_from_its_head(serving):
    decider = FakeDecider()
    app = serving(decider, model=site_model("tone"), mode=MODE_LIVE)
    response = await post(app, {"tone": TONE, "billing": BILLING})
    body = json.loads(response.content)
    assert body["answers"]["tone"] == {
        "type": "choice",
        "choice": "angry",
        "confidence": 0.9,
        "probabilities": {"calm": 0.25, "angry": 0.75},
    }
    assert body["answers"]["billing"] == NOUL_ANSWER
    assert list(decider.batches[0]) == ["billing"]
    assert response.headers["x-stuntd"] == "jev; mode=local; questions=2; live=1; zeroshot=1"
    rows = app.state.store.decisions("tone", 10)
    assert [(row.mode, row.answer, row.agree) for row in rows] == [("live", "angry", None)]


async def test_a_live_site_answers_a_question_whose_criteria_came_in_another_order(serving):
    decider = FakeDecider(verdict=FakeVerdict(0, 0.9, 7, (0.75, 0.25)))
    app = serving(decider, model=site_model("tone", labels=("angry", "calm")), mode=MODE_LIVE)
    response = await post(app, {"tone": TONE})
    assert json.loads(response.content)["answers"]["tone"] == {
        "type": "choice",
        "choice": "angry",
        "confidence": 0.9,
        "probabilities": {"calm": 0.25, "angry": 0.75},
    }
    assert decider.batches == []


@pytest.mark.parametrize(
    ("name", "question", "labels", "label"),
    [
        ("tone", TONE, ("calm", "angry"), 0),
        ("billing", BILLING, ("false", "true"), 1),
        ("urgency", URGENCY, ("0", "1", "2"), 2),
    ],
    ids=["choice", "noul", "score"],
)
async def test_a_shadow_site_answers_zero_shot_and_compares_nothing(
    serving, name, question, labels, label
):
    decider = FakeDecider(verdict=FakeVerdict(label, 0.9, 7, (0.9,) * len(labels)))
    app = serving(decider, model=site_model(name, labels=labels), mode=MODE_SHADOW)
    response = await post(app, {name: question})
    assert json.loads(response.content)["answers"][name] == ANSWERS[name]
    assert response.headers["x-stuntd"] == "jev; mode=local; questions=1; live=0; zeroshot=1"
    assert app.state.store.decisions(name, 10) == []


async def test_a_head_below_its_threshold_answers_zero_shot(serving):
    decider = FakeDecider(verdict=FakeVerdict(1, 0.4, 7, (0.25, 0.75)))
    app = serving(decider, model=site_model("tone"), mode=MODE_LIVE)
    response = await post(app, {"tone": TONE})
    assert json.loads(response.content)["answers"]["tone"] == CHOICE_ANSWER
    assert list(decider.batches[0]) == ["tone"]
    assert response.headers["x-stuntd"] == "jev; mode=local; questions=1; live=0; zeroshot=1"
    assert app.state.store.decisions("tone", 10) == []


async def test_a_sampled_check_is_still_answered_by_the_head(serving):
    decider = FakeDecider(verdict=FakeVerdict(0, 0.9, 7, (0.75, 0.25)))
    app = serving(decider, model=site_model("tone"), mode=MODE_LIVE, check_share=0.9999)
    response = await post(app, {"tone": TONE})
    assert json.loads(response.content)["answers"]["tone"]["choice"] == "calm"
    assert decider.batches == []
    assert response.headers["x-stuntd"] == "jev; mode=local; questions=1; live=1; zeroshot=0"
    rows = app.state.store.decisions("tone", 10)
    assert [(row.mode, row.answer, row.agree) for row in rows] == [("live", "calm", None)]


async def test_a_decider_that_fails_is_reported(serving, caplog):
    app = serving(FakeDecider(error=RuntimeError("boom")))
    with caplog.at_level(logging.ERROR):
        response = await post(app, {"billing": BILLING})
    assert response.status_code == 503
    assert json.loads(response.content) == {"error": {"message": "the model could not answer"}}
    assert response.headers["x-stuntd"] == "jev; mode=local; reason=model-error"
    assert "zero-shot answers not built" in caplog.text


async def test_a_zero_shot_answer_that_never_came_is_reported(serving, caplog):
    app = serving(FakeDecider(answers={}))
    with caplog.at_level(logging.ERROR):
        response = await post(app, {"billing": BILLING})
    assert response.status_code == 503
    assert json.loads(response.content) == {"error": {"message": "the model could not answer"}}
    assert response.headers["x-stuntd"] == "jev; mode=local; reason=model-error"
    assert "zero-shot answers missing: ['billing']" in caplog.text


async def test_a_live_site_whose_question_changed_its_labels_answers_zero_shot(serving):
    decider = FakeDecider()
    app = serving(decider, model=site_model("tone"), mode=MODE_LIVE)
    changed = dict(TONE, criteria={"calm": None, "cross": None})
    response = await post(app, {"tone": changed})
    assert json.loads(response.content)["answers"]["tone"] == CHOICE_ANSWER
    assert list(decider.batches[0]) == ["tone"]
    assert response.headers["x-stuntd"] == "jev; mode=local; questions=1; live=0; zeroshot=1"
    assert app.state.store.decisions("tone", 10) == []


async def test_the_answer_carries_a_request_id(serving):
    app = serving(FakeDecider())
    first = await post(app, {"billing": BILLING})
    again = await post(app, {"billing": BILLING})
    other = await post(app, {"billing": BILLING}, state="Where is my invoice?")
    request_id = first.headers["x-typesafe-request-id"]
    assert len(request_id) == 12
    assert int(request_id, 16) >= 0
    assert again.headers["x-typesafe-request-id"] == request_id
    assert other.headers["x-typesafe-request-id"] != request_id


async def test_a_namespaced_question_name_lands_in_its_own_folder(serving):
    decider = FakeDecider(answers={"moderation:verdict": CHOICE_ANSWER})
    app = serving(decider, model=site_model("moderation.verdict"), mode=MODE_LIVE)
    response = await post(app, {"moderation:verdict": TONE})
    assert json.loads(response.content)["answers"]["moderation:verdict"]["choice"] == "angry"
    assert decider.batches == []
    rows = app.state.store.decisions("moderation.verdict", 10)
    assert [(row.mode, row.answer) for row in rows] == [("live", "angry")]


async def test_a_question_without_instructions_or_criteria_is_answered(serving):
    decider = FakeDecider()
    app = serving(decider)
    response = await post(app, {"billing": BILLING})
    assert response.status_code == 200
    assert json.loads(response.content)["answers"] == {"billing": NOUL_ANSWER}
    assert decider.batches == [{"billing": {"type": "noul", "instructions": ""}}]


async def test_the_model_list_names_the_configured_model(serving):
    app = serving(FakeDecider(), jev_model_name="stuntd-local")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        response = await client.get("/v1/models")
    body = json.loads(response.content)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert [model["name"] for model in body["models"]] == ["stuntd-local"]
    assert set(body["models"][0]) == {"name", "description", "release_date"}
