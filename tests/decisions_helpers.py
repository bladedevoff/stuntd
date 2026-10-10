import json
from dataclasses import dataclass

from stuntd.train.artifacts import SiteModel
from stuntd.train.metrics import ClassStats

PREDICATE = {
    "type": "predicate",
    "name": "damaged",
    "instructions": "Is the product visibly damaged?",
}
CHOICE = {
    "type": "choice",
    "name": "route",
    "instructions": "Which team should handle this?",
    "choices": [
        {"value": "billing", "description": "Payment questions"},
        {"value": "shipping"},
        {"value": "other"},
    ],
}
SCORE = {
    "type": "score",
    "name": "urgency",
    "instructions": "How urgent is this?",
    "levels": [
        {"label": "low"},
        {"label": "medium", "description": "Needs a reply today"},
        {"label": "high"},
    ],
}
UNNAMED_FLAG = {
    "type": "choice",
    "instructions": "Is this a repeat customer?",
    "choices": [{"value": True}, {"value": False}],
}

TEXT = "The package arrived crushed and I was charged twice."

PREDICATE_REQUEST = {"model": "gpt-6-luna", "input": TEXT, "questions": [PREDICATE]}
CHOICE_REQUEST = {
    "model": "gpt-6-luna",
    "input": [{"role": "user", "type": "message", "content": TEXT}],
    "questions": [CHOICE],
}
SCORE_REQUEST = {
    "model": "gpt-6-luna",
    "input": [{"role": "user", "content": [{"type": "input_text", "text": TEXT}]}],
    "questions": [SCORE],
}

PREDICATE_ANSWER = {"type": "predicate", "name": "damaged", "probability": 0.92}
CHOICE_ANSWER = {
    "type": "choice",
    "name": "route",
    "choice": "billing",
    "confidence": 0.6,
    "probabilities": [
        {"value": "billing", "probability": 0.6},
        {"value": "shipping", "probability": 0.3},
        {"value": "other", "probability": 0.1},
    ],
}
SCORE_ANSWER = {
    "type": "score",
    "name": "urgency",
    "score": 1.4,
    "confidence": 0.5,
    "probabilities": [
        {"value": 0, "label": "low", "probability": 0.1},
        {"value": 1, "label": "medium", "probability": 0.4},
        {"value": 2, "label": "high", "probability": 0.5},
    ],
}
REFUSAL_ANSWER = {"type": "refusal", "name": "damaged"}

PROVIDER_MODEL = "gpt-6-luna"


def usage(input_tokens, output_tokens=0):
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }


def decision_body(answers, input_tokens=120, output_tokens=0):
    body = {
        "answers": answers,
        "model": PROVIDER_MODEL,
        "usage": usage(input_tokens, output_tokens),
    }
    return json.dumps(body).encode()


@dataclass(frozen=True)
class FakeVerdict:
    label: int
    confidence: float
    latency_ms: int
    probabilities: tuple[float, ...]
    novelty: float | None = None


def site_model(site, kind="choice", field="route", labels=("billing", "other", "shipping")):
    return SiteModel(
        site=site,
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
