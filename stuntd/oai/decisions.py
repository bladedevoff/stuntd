from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from stuntd.jev.schema import JevError, Question, build_question, load_body

__all__ = [
    "DecisionQuestion",
    "DecisionsRequest",
    "answers_body",
    "error_body",
    "parse_request",
    "provider_answers",
]

_TEXT_PART = "input_text"
_ERROR_TYPE = "invalid_request_error"


@dataclass(frozen=True)
class DecisionQuestion:
    """One Decisions question as the Jev question it means, with what its answer must echo."""

    name: str | None
    question: Question
    options: tuple[str | bool, ...]
    """The typed choice values of a choice question, the level labels of a score question."""


@dataclass(frozen=True)
class DecisionsRequest:
    """A parsed POST /v1/decisions body that stuntd can learn from."""

    text: str
    items: tuple[DecisionQuestion, ...]

    @property
    def questions(self) -> tuple[Question, ...]:
        return tuple(item.question for item in self.items)


def parse_request(body: bytes) -> DecisionsRequest:
    """Parses a POST /v1/decisions body into its text and the Jev questions it means; a request
    stuntd cannot learn from, such as one with images, raises JevError."""
    payload = load_body(body)
    if not isinstance(payload, dict):
        raise JevError("request body must be a JSON object")
    if not isinstance(payload.get("model"), str):
        raise JevError("model must be a string")
    text = _text(payload.get("input"))
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise JevError("questions must be a non-empty array")
    items = tuple(_item(index, raw) for index, raw in enumerate(questions))
    keys = [item.question.name for item in items]
    if len(set(keys)) != len(keys):
        raise JevError("question names must be unique")
    return DecisionsRequest(text, items)


def answers_body(
    model: str, request: DecisionsRequest, answers: Sequence[dict[str, Any]], input_tokens: int
) -> bytes:
    """Builds the Decisions success body out of Jev answers, one per question and in order."""
    body = {
        "answers": [
            _answer(item, answer) for item, answer in zip(request.items, answers, strict=True)
        ],
        "model": model,
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens,
        },
    }
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def error_body(message: str) -> bytes:
    """Builds the OpenAI error body."""
    body = {"error": {"message": message, "type": _ERROR_TYPE, "param": None, "code": None}}
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def provider_answers(
    request: DecisionsRequest, answered: dict[str, Any]
) -> list[dict[str, Any] | None] | None:
    """The provider's answers as Jev answers aligned with the questions, or None when the
    response carries no matching list; a refusal or a malformed answer stands as None."""
    answers = answered.get("answers")
    if not isinstance(answers, list) or len(answers) != len(request.items):
        return None
    return [_jev_answer(answer) for answer in answers]


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not value:
        raise JevError("input must be a string or a non-empty array of messages")
    texts: list[str] = []
    for message in value:
        if not isinstance(message, dict) or message.get("role") != "user":
            raise JevError("input messages must be user messages")
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        if not isinstance(content, list):
            raise JevError("message content must be a string or an array of parts")
        for part in content:
            if not isinstance(part, dict) or part.get("type") != _TEXT_PART:
                raise JevError("stuntd learns from text only: input parts must be input_text")
            if not isinstance(part.get("text"), str):
                raise JevError("input_text must carry a text string")
            texts.append(part["text"])
    return "\n".join(texts)


def _item(index: int, raw: object) -> DecisionQuestion:
    if not isinstance(raw, dict):
        raise JevError(f"question {index} must be an object")
    name = raw.get("name")
    if name is not None and not isinstance(name, str):
        raise JevError(f"question {index} must have a name as a string")
    key = name if name is not None else f"#{index}"
    instructions = raw.get("instructions")
    if not isinstance(instructions, str):
        raise JevError(f'Question "{key}" must have instructions as a string')
    question_type = raw.get("type")
    options: tuple[str | bool, ...] = ()
    jev: dict[str, object] = {"instructions": instructions}
    if question_type == "predicate":
        jev["type"] = "noul"
    elif question_type == "choice":
        options, jev["criteria"] = _choices(key, raw.get("choices"))
        jev["type"] = "choice"
    elif question_type == "score":
        options, jev["criteria"] = _levels(key, raw.get("levels"))
        jev["type"] = "score"
    else:
        raise JevError(f'Question "{key}" must have a type of predicate, choice, or score')
    return DecisionQuestion(name, build_question(key, jev), options)


def _choices(key: str, raw: object) -> tuple[tuple[str | bool, ...], dict[str, str | None]]:
    if not isinstance(raw, list):
        raise JevError(f'Question "{key}": choices must be an array')
    values: list[str | bool] = []
    criteria: dict[str, str | None] = {}
    for choice in raw:
        if not isinstance(choice, dict) or not isinstance(choice.get("value"), (str, bool)):
            raise JevError(f'Question "{key}": each choice needs a string or boolean value')
        description = choice.get("description")
        if description is not None and not isinstance(description, str):
            raise JevError(f'Question "{key}": a choice description must be a string')
        label = _label(choice["value"])
        if label in criteria:
            raise JevError(f'Question "{key}": the choice values collide on {label!r}')
        values.append(choice["value"])
        criteria[label] = description
    if len(values) < 2:
        raise JevError(f'Question "{key}": choices needs at least two values')
    return tuple(values), criteria


def _levels(key: str, raw: object) -> tuple[tuple[str | bool, ...], list[str]]:
    if not isinstance(raw, list):
        raise JevError(f'Question "{key}": levels must be an array')
    labels: list[str] = []
    criteria: list[str] = []
    for level in raw:
        if not isinstance(level, dict) or not isinstance(level.get("label"), str):
            raise JevError(f'Question "{key}": each level needs a label string')
        description = level.get("description")
        if description is not None and not isinstance(description, str):
            raise JevError(f'Question "{key}": a level description must be a string')
        labels.append(level["label"])
        criteria.append(f"{level['label']}: {description}" if description else level["label"])
    return tuple(labels), criteria


def _label(value: object) -> str:
    """The label a choice value trains under: a string as itself, a boolean as true or false."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    raise TypeError(f"not a choice value: {value!r}")


def _answer(item: DecisionQuestion, answer: dict[str, Any]) -> dict[str, Any]:
    question = item.question
    if question.type == "noul":
        return {"type": "predicate", "name": item.name, "probability": answer["noul"]}
    probabilities = answer["probabilities"]
    if question.type == "choice":
        typed = dict(zip(question.labels, item.options, strict=True))
        return {
            "type": "choice",
            "name": item.name,
            "choice": typed[answer["choice"]],
            "confidence": answer["confidence"],
            "probabilities": [
                {"value": typed[label], "probability": probabilities[label]}
                for label in question.labels
            ],
        }
    return {
        "type": "score",
        "name": item.name,
        "score": answer["score"],
        "confidence": answer["confidence"],
        "probabilities": [
            {"value": level, "label": label, "probability": probabilities[str(level)]}
            for level, label in enumerate(item.options)
        ],
    }


def _jev_answer(answer: object) -> dict[str, Any] | None:
    if not isinstance(answer, dict):
        return None
    try:
        if answer.get("type") == "predicate":
            return {"type": "noul", "noul": answer["probability"]}
        if answer.get("type") == "choice":
            return {"type": "choice", "choice": _label(answer["choice"])}
        if answer.get("type") == "score":
            probabilities = {str(p["value"]): p["probability"] for p in answer["probabilities"]}
            return {"type": "score", "probabilities": probabilities}
    except (KeyError, TypeError):
        return None
    return None
