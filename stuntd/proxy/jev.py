from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from starlette.requests import Request
from starlette.responses import Response

from stuntd.jev.answer import error_body, response_body
from stuntd.jev.schema import Question, parse_request
from stuntd.jev.state import serialize_state
from stuntd.proxy.questions import QuestionRoutes, Wire

if TYPE_CHECKING:
    from stuntd.proxy.app import Proxy

__all__ = ["JevRoutes"]

_MODELS = "/v1/models"
_MODEL_DESCRIPTION = "Local Laya heads trained by stuntd on this daemon's own traffic."
_RELEASE_DATE = "2026-09-22"


@dataclass(frozen=True)
class _Asked:
    text: str
    questions: tuple[Question, ...]


def _parse(body: bytes) -> _Asked:
    parsed = parse_request(body)
    return _Asked(serialize_state(parsed.state), parsed.questions)


def _answers_body(
    model: str, asked: _Asked, answers: Sequence[dict[str, Any]], input_tokens: int
) -> bytes:
    named = {
        question.name: answer for question, answer in zip(asked.questions, answers, strict=True)
    }
    return response_body(model, named, input_tokens)


def _provider_answers(
    asked: _Asked, answered: dict[str, Any]
) -> list[dict[str, Any] | None] | None:
    answers = answered.get("answers")
    if not isinstance(answers, dict):
        return None
    return [
        answer if isinstance(answer := answers.get(question.name), dict) else None
        for question in asked.questions
    ]


class JevRoutes:
    """The Jev endpoints of one proxy: System One answers and the model list."""

    def __init__(self, proxy: Proxy) -> None:
        self._proxy = proxy
        wire = Wire(proxy.settings.jev_path, _parse, _answers_body, error_body, _provider_answers)
        self._questions = QuestionRoutes(proxy, wire, proxy.settings.jev_upstream)

    def __repr__(self) -> str:
        return f"JevRoutes(model={self._proxy.settings.jev_model_name!r})"

    async def systemone(self, request: Request) -> Response:
        """Answers one System One request: through the Jev provider when one is configured, from
        the trained heads and the base Laya checkpoint otherwise."""
        return await self._questions.answer(request)

    async def models(self, request: Request) -> Response:
        """Lists the models on offer: the provider's own in proxy mode, otherwise the one model
        this daemon serves, in the shape the SDK's model list expects."""
        if self._proxy.settings.jev_upstream:
            # The relayed answers name the provider's models, so the list must be its own.
            return await self._questions.relay(request, _MODELS)
        body = json.dumps(
            {
                "models": [
                    {
                        "name": self._proxy.settings.jev_model_name,
                        "description": _MODEL_DESCRIPTION,
                        "release_date": _RELEASE_DATE,
                    }
                ]
            },
            separators=(",", ":"),
        ).encode("utf-8")
        return Response(content=body, media_type="application/json")
