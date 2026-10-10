import json
from dataclasses import dataclass

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from stuntd.proxy.app import build_app
from stuntd.settings import Settings

pytestmark = pytest.mark.anyio

SYSTEMONE = "/v1/systemone"
DECISIONS = "/v1/decisions"

COLUMNS = "site, answer, model, prompt_tokens, completion_tokens"


@dataclass(frozen=True)
class Teacher:
    origin: str
    path: str
    request: dict
    response: dict
    captured: list[tuple]


KEV = Teacher(
    origin="http://127.0.0.1:8009",
    path=SYSTEMONE,
    request={
        "state": (
            "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card."
        ),
        "model": "kev-latest",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "returns": "Exchanges, refunds, wrong or damaged items",
                    "shipping": "Delivery status, delays, lost packages",
                    "billing": "Charges, invoices, payment problems",
                },
            },
            "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["Calm", "Frustrated", "Very angry"],
            },
        },
    },
    response={
        "model": "kev-latest",
        "answers": {
            "department": {
                "type": "choice",
                "choice": "returns",
                "confidence": 0.21,
                "probabilities": {"returns": 0.47, "shipping": 0.28, "billing": 0.25},
            },
            "escalate": {"type": "noul", "noul": 0.93},
            "frustration": {
                "type": "score",
                "score": 1.44,
                "confidence": 0.34,
                "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                "probabilities": {"0": 0.00, "1": 0.56, "2": 0.44},
            },
        },
        "usage": {"input_tokens": 101, "output_tokens": 161},
        "latency_ms": 495,
    },
    captured=[
        ("department", "returns", "kev-latest", 101, 161),
        ("escalate", "true", "kev-latest", 101, 161),
        ("frustration", "1", "kev-latest", 101, 161),
    ],
)

MICA = Teacher(
    origin="http://127.0.0.1:8010",
    path=SYSTEMONE,
    request={
        "state": "The user asked to delete the staging database. No approval has been given.",
        "model": "jev-latest",
        "questions": {
            "q": {"type": "noul", "instructions": "Should the agent delete it now?"},
            "risk": {
                "type": "score",
                "instructions": "How risky is the action?",
                "criteria": ["Safe", "Careful", "Dangerous"],
            },
            "action": {
                "type": "choice",
                "instructions": "What should the agent do?",
                "criteria": {"delete": None, "ask": None},
            },
        },
    },
    response={
        "model": "mica-v0.1-4b",
        "answers": {
            "q": {"type": "noul", "noul": 0.04, "answer": False, "confidence": 0.96},
            "risk": {
                "type": "score",
                "score": 2,
                "probabilities": {"0": 0.01, "1": 0.09, "2": 0.9},
                "answer": 2,
                "confidence": 0.9,
            },
            "action": {
                "type": "choice",
                "choice": "ask",
                "probabilities": {"delete": 0.05, "ask": 0.95},
                "answer": "ask",
                "confidence": 0.95,
            },
        },
        "usage": {"input_tokens": 31, "output_tokens": 0},
        "latency_ms": 60,
    },
    captured=[
        ("q", "false", "mica-v0.1-4b", 31, 0),
        ("risk", "2", "mica-v0.1-4b", 31, 0),
        ("action", "ask", "mica-v0.1-4b", 31, 0),
    ],
)

PERPLEXITY = Teacher(
    origin="https://api.perplexity.ai",
    path=DECISIONS,
    request={
        "model": "pplx-decider-v1.1-27b",
        "state": {
            "title": "Battery died after two weeks",
            "review": "The headphones sound great, but the battery stopped charging after two weeks.",
        },
        "questions": {
            "defect": {"type": "noul", "instructions": "Does the review report a product defect?"},
            "sentiment": {
                "type": "choice",
                "instructions": "What is the overall sentiment of the review?",
                "criteria": {
                    "positive": "Mostly satisfied",
                    "mixed": "Praise and complaints in one review",
                    "negative": "Mostly dissatisfied",
                },
            },
            "severity": {
                "type": "score",
                "instructions": "How severe is the reported problem?",
                "criteria": ["Cosmetic", "Inconvenient", "Product unusable"],
            },
        },
    },
    response={
        "model": "pplx-decider-v1.1-27b",
        "answers": {
            "defect": {"type": "noul", "noul": 0.9424522889347015},
            "sentiment": {
                "type": "choice",
                "choice": "mixed",
                "confidence": 0.9255246944002182,
                "probabilities": {
                    "positive": 0.020649883775315993,
                    "mixed": 0.9503497962668123,
                    "negative": 0.02900031995787183,
                },
            },
            "severity": {
                "type": "score",
                "score": 1.7838686319784252,
                "confidence": 0.7838686319784252,
                "legend": {"0": "Cosmetic", "1": "Inconvenient", "2": "Product unusable"},
                "probabilities": {
                    "0": 0.008423954913615923,
                    "1": 0.199283458194343,
                    "2": 0.7922925868920411,
                },
            },
        },
        "usage": {"input_tokens": 367, "output_tokens": 3},
    },
    captured=[
        ("defect", "true", "pplx-decider-v1.1-27b", 367, 3),
        ("sentiment", "mixed", "pplx-decider-v1.1-27b", 367, 3),
        ("severity", "2", "pplx-decider-v1.1-27b", 367, 3),
    ],
)

STROM = Teacher(
    origin="https://api.uprelic.com",
    path=SYSTEMONE,
    request={
        "model": "strom-1.0.7",
        "state": "Our whole team is blocked. The app has been down since this morning.",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "engineering": "Outages and bugs",
                    "billing": "Payments and invoices",
                    "account": "Login and access",
                },
            },
            "urgent": {"type": "noul", "instructions": "This message needs urgent attention."},
        },
    },
    response={
        "model": "strom-1.0.7",
        "answers": {
            "team": {
                "type": "choice",
                "choice": "engineering",
                "confidence": 0.83,
                "probabilities": {"engineering": 0.89, "billing": 0.07, "account": 0.04},
            },
            "urgent": {"type": "noul", "noul": 0.96},
        },
        "usage": {
            "input_tokens": 283,
            "output_tokens": 2,
            "cached_input_tokens": 256,
            "cost": 0.000011886,
            "currency": "usd",
        },
        "metadata": {"server_processing_time_ms": 85},
    },
    captured=[
        ("team", "engineering", "strom-1.0.7", 283, 2),
        ("urgent", "true", "strom-1.0.7", 283, 2),
    ],
)

TEACHERS = [KEV, MICA, PERPLEXITY, STROM]
TEACHER_IDS = ["kev", "mica", "perplexity", "strom"]


class Provider:
    """Fake teacher that keeps what reached it and answers a fixed body."""

    def __init__(self, body, status=200):
        self.body = body
        self.status = status
        self.seen = []

    async def handle(self, request: Request):
        self.seen.append({"path": request.url.path, "body": await request.body()})
        return Response(content=self.body, status_code=self.status, media_type="application/json")


@pytest.fixture
async def teaching(data_dir):
    apps = []

    def make(teacher, provider):
        routes = [Route("/{path:path}", provider.handle, methods=["GET", "POST"])]
        settings = Settings(
            upstream="http://upstream", jev_upstream=teacher.origin, jev_path=teacher.path
        )
        app = build_app(settings, transport=httpx.ASGITransport(app=Starlette(routes=routes)))
        apps.append(app)
        return app

    yield make
    for app in apps:
        await app.state.proxy.aclose()


async def post(app, content):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        return await client.post(
            SYSTEMONE, content=content, headers={"content-type": "application/json"}
        )


def captures(app, columns=COLUMNS):
    return app.state.store._conn.execute(f"select {columns} from captures order by id").fetchall()


@pytest.mark.parametrize("teacher", TEACHERS, ids=TEACHER_IDS)
async def test_a_teacher_answer_crosses_byte_for_byte_and_is_recorded(teaching, teacher):
    body = json.dumps(teacher.request).encode()
    provider = Provider(json.dumps(teacher.response).encode())
    app = teaching(teacher, provider)
    response = await post(app, body)
    assert response.status_code == 200
    assert response.content == provider.body
    assert provider.seen == [{"path": teacher.path, "body": body}]
    assert captures(app) == teacher.captured


async def test_a_perplexity_state_object_is_recorded_as_its_json_text(teaching):
    app = teaching(PERPLEXITY, Provider(json.dumps(PERPLEXITY.response).encode()))
    await post(app, json.dumps(PERPLEXITY.request).encode())
    texts = {text for (text,) in captures(app, "input_text")}
    assert texts == {json.dumps(PERPLEXITY.request["state"], ensure_ascii=False)}


async def test_the_mica_readme_request_without_a_model_is_relayed_and_not_learned(teaching):
    body = json.dumps(
        {"state": MICA.request["state"], "questions": {"q": MICA.request["questions"]["q"]}}
    ).encode()
    provider = Provider(json.dumps(MICA.response).encode())
    app = teaching(MICA, provider)
    response = await post(app, body)
    assert response.content == provider.body
    assert provider.seen == [{"path": SYSTEMONE, "body": body}]
    assert response.headers["x-stuntd"] == "jev; mode=proxy; reason=not-parsed"
    assert captures(app) == []


@pytest.mark.parametrize(
    ("teacher", "status", "error"),
    [
        (KEV, 422, {"detail": "score criteria must contain between 1 and 255 levels"}),
        (MICA, 400, {"error": "input exceeds 8192 tokens"}),
        (
            PERPLEXITY,
            400,
            {"error": {"message": "model is required", "type": "invalid_request", "code": 400}},
        ),
        (STROM, 422, {"detail": [{"loc": ["body", "questions"], "msg": "field required"}]}),
    ],
    ids=TEACHER_IDS,
)
async def test_a_teacher_error_is_relayed_and_nothing_is_recorded(teaching, teacher, status, error):
    provider = Provider(json.dumps(error).encode(), status)
    app = teaching(teacher, provider)
    response = await post(app, json.dumps(teacher.request).encode())
    assert (response.status_code, response.content) == (status, provider.body)
    assert captures(app) == []


async def test_a_choice_answer_without_probabilities_is_still_recorded(teaching):
    answers = {
        "team": {"type": "choice", "choice": "engineering"},
        "urgent": {"type": "noul", "noul": 0.1},
    }
    provider = Provider(json.dumps({**STROM.response, "answers": answers}).encode())
    app = teaching(STROM, provider)
    response = await post(app, json.dumps(STROM.request).encode())
    assert response.content == provider.body
    assert captures(app, "site, answer") == [("team", "engineering"), ("urgent", "false")]


async def test_a_score_answer_without_probabilities_is_relayed_and_not_recorded(teaching):
    request = {**KEV.request, "questions": {"frustration": KEV.request["questions"]["frustration"]}}
    score = {"type": "score", "score": 1, "confidence": 0.5}
    provider = Provider(json.dumps({**KEV.response, "answers": {"frustration": score}}).encode())
    app = teaching(KEV, provider)
    response = await post(app, json.dumps(request).encode())
    assert response.content == provider.body
    assert captures(app) == []


async def test_a_kev_score_with_more_than_ten_levels_is_relayed_and_not_learned(teaching):
    levels = [str(level) for level in range(11)]
    question = {"type": "score", "instructions": "How bad?", "criteria": levels}
    request = {**KEV.request, "questions": {"badness": question}}
    answer = {
        "type": "score",
        "score": 10.0,
        "confidence": 1.0,
        "legend": {str(level): str(level) for level in range(11)},
        "probabilities": {str(level): float(level == 10) for level in range(11)},
    }
    provider = Provider(json.dumps({**KEV.response, "answers": {"badness": answer}}).encode())
    app = teaching(KEV, provider)
    response = await post(app, json.dumps(request).encode())
    assert response.content == provider.body
    assert response.headers["x-stuntd"] == "jev; mode=proxy; reason=not-parsed"
    assert captures(app) == []
