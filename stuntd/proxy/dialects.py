from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from stuntd.decisions.schema import (
    DecisionSchema,
    Schema,
    declares_message_schema,
    declares_schema,
    detect_message_schema,
    detect_schema,
)
from stuntd.decisions.site import content_text, input_text, system_prompt
from stuntd.proxy.capture import (
    extract_answer,
    extract_message_answer,
    message_usage_tokens,
    usage_tokens,
)
from stuntd.serve.answer import completion_body, message_body

__all__ = ["ANTHROPIC_MESSAGES", "DIALECTS", "OPENAI_CHAT", "Dialect"]

_CHAT_COMPLETIONS = "/chat/completions"
_MESSAGES = "/v1/messages"


@dataclass(frozen=True)
class Dialect:
    """How one provider API carries a decision: reading its requests and responses, and answering."""

    matches: Callable[[str, str], bool]
    detect: Callable[[dict[str, Any]], Schema | None]
    declares_schema: Callable[[dict[str, Any]], bool]
    input_text: Callable[[dict[str, Any]], str]
    system_prompt: Callable[[dict[str, Any]], str]
    extract_answer: Callable[[DecisionSchema, dict[str, Any]], str | None]
    usage_tokens: Callable[[dict[str, Any]], tuple[int | None, int | None]]
    completion_body: Callable[[Schema, Sequence[str], str, int, str], bytes]
    request_id_prefix: str


OPENAI_CHAT = Dialect(
    matches=lambda method, path: method == "POST" and path.endswith(_CHAT_COMPLETIONS),
    detect=detect_schema,
    declares_schema=declares_schema,
    input_text=lambda payload: input_text(payload.get("messages")),
    system_prompt=lambda payload: system_prompt(payload.get("messages")),
    extract_answer=extract_answer,
    usage_tokens=usage_tokens,
    completion_body=completion_body,
    request_id_prefix="chatcmpl-stuntd-",
)

ANTHROPIC_MESSAGES = Dialect(
    matches=lambda method, path: method == "POST" and path.endswith(_MESSAGES),
    detect=detect_message_schema,
    declares_schema=declares_message_schema,
    input_text=lambda payload: input_text(payload.get("messages")),
    system_prompt=lambda payload: content_text(payload.get("system")),
    extract_answer=extract_message_answer,
    usage_tokens=message_usage_tokens,
    completion_body=message_body,
    request_id_prefix="msg_stuntd_",
)

DIALECTS = (OPENAI_CHAT, ANTHROPIC_MESSAGES)
