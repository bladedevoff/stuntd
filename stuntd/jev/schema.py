from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "JevError",
    "MAX_CHOICE_OPTIONS",
    "MAX_SCORE_LEVELS",
    "MIN_SCORE_LEVELS",
    "Question",
    "SystemOneRequest",
    "kind_for",
    "parse_request",
    "typed_field",
]

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

_SITE_LENGTH = 16
_NOUL_LABELS = ("false", "true")
_KINDS = {"choice": "choice", "noul": "boolean", "score": "number"}
_CANONICAL_KEYS = frozenset({"type", "instructions", "criteria"})
# The site whitelist of train.artifacts plus ':', which a question name may use to namespace its
# site; it becomes '.' in the folder name.
_QUESTION_NAME = re.compile(r"[A-Za-z0-9_.:-]{1,64}")


class JevError(Exception):
    """A malformed Jev request: the message and status code its error body carries."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass(frozen=True)
class Question:
    """One typed question of a System One request and the decision site it lands on."""

    name: str
    type: str
    instructions: str | dict[str, object] | list[object] | None
    criteria: object
    canonical: str
    labels: tuple[str, ...]
    site: str


@dataclass(frozen=True)
class SystemOneRequest:
    """A parsed POST /v1/systemone body."""

    state: object
    model: str
    questions: tuple[Question, ...]


def kind_for(question: Question) -> str:
    """The decision kind the store and the trainer use for a question's type."""
    kind = _KINDS.get(question.type)
    if kind is None:
        raise ValueError(f"unknown question type: {question.type!r}")
    return kind


def typed_field(site: str, canonical: str) -> tuple[str, str, tuple[str, ...]] | None:
    """What a Jev canonical trains as: its site as the field, its kind, and the labels _labels
    derives, here in the canonical's order, which sorts a choice's criteria keys. None for any
    other schema and for criteria that no longer parse, so both schema readers stay total."""
    try:
        parsed = json.loads(canonical)
    except ValueError:
        return None
    if not isinstance(parsed, dict) or set(parsed) != _CANONICAL_KEYS:
        return None
    question_type = parsed["type"]
    if not isinstance(question_type, str) or question_type not in _KINDS:
        return None
    try:
        labels = _labels(site, question_type, parsed["criteria"])
    except JevError:
        return None
    return site, _KINDS[question_type], labels


def _canonical(question_type: str, instructions: object, criteria: object) -> str:
    return json.dumps(
        {"type": question_type, "instructions": instructions, "criteria": criteria},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _site(name: str, canonical: str) -> str:
    folder = name.replace(":", ".")
    # A name of nothing but dots passes the charset and still walks up out of the models folder.
    if _QUESTION_NAME.fullmatch(name) and set(folder) != {"."}:
        return folder
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_SITE_LENGTH]


def _labels(name: str, question_type: str, criteria: object) -> tuple[str, ...]:
    if question_type == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise JevError(f'Question "{name}": choice criteria must be a non-empty object')
        if len(criteria) > MAX_CHOICE_OPTIONS:
            raise JevError(
                f'Question "{name}": choice criteria cannot contain more than '
                f"{MAX_CHOICE_OPTIONS} options"
            )
        return tuple(criteria)
    if question_type == "score":
        levels = len(criteria) if isinstance(criteria, list) else 0
        if not MIN_SCORE_LEVELS <= levels <= MAX_SCORE_LEVELS:
            raise JevError(
                f'Question "{name}": score criteria must contain between '
                f"{MIN_SCORE_LEVELS} and {MAX_SCORE_LEVELS} levels"
            )
        return tuple(str(level) for level in range(levels))
    if criteria is not None and not isinstance(criteria, dict):
        raise JevError(f'Question "{name}": noul criteria must be an object')
    return _NOUL_LABELS


def _question(name: str, raw: object) -> Question:
    if not isinstance(raw, dict):
        raise JevError(f'Question "{name}" must be an object')
    question_type = raw.get("type")
    if not isinstance(question_type, str) or question_type not in _KINDS:
        raise JevError(f'Question "{name}" must have a type of choice, noul, or score')
    # The SDK leaves instructions off the wire when the caller did not set them.
    instructions = raw.get("instructions")
    if instructions is not None and not isinstance(instructions, (str, dict, list)):
        raise JevError(f'Question "{name}" must have instructions as a string, object, or array')
    criteria = raw.get("criteria")
    labels = _labels(name, question_type, criteria)
    canonical = _canonical(question_type, instructions, criteria)
    return Question(
        name=name,
        type=question_type,
        instructions=instructions,
        criteria=criteria,
        canonical=canonical,
        labels=labels,
        site=_site(name, canonical),
    )


def parse_request(body: bytes) -> SystemOneRequest:
    """Parses a POST /v1/systemone body into its state and its typed questions."""
    try:
        # JSON values are Any by nature; every branch below narrows before it is used.
        payload: Any = json.loads(body)
    except ValueError:
        raise JevError("request body must be valid JSON")
    if not isinstance(payload, dict):
        raise JevError("request body must be a JSON object")
    if "state" not in payload:
        raise JevError("request must contain state")
    state = payload["state"]
    if not isinstance(state, (str, dict, list)):
        raise JevError("state must be a string, object, or array")
    model = payload.get("model")
    if not isinstance(model, str):
        raise JevError("model must be a string")
    if "questions" not in payload:
        raise JevError("request must contain questions")
    questions = payload["questions"]
    if not isinstance(questions, dict) or not questions:
        raise JevError("questions must be a non-empty object")
    return SystemOneRequest(
        state=state,
        model=model,
        questions=tuple(_question(name, raw) for name, raw in questions.items()),
    )
