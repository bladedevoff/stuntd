from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from stuntd.decisions.schema import DecisionSchema

__all__ = [
    "decoded_json",
    "extract_answer",
    "extract_message_answer",
    "message_usage_tokens",
    "token_count",
    "usage_tokens",
]


def _answered_value(schema: DecisionSchema, response_json: dict[str, Any]) -> object:
    message = response_json["choices"][0]["message"]
    if schema.source == "tool":
        function = message["tool_calls"][0]["function"]
        if function["name"] != schema.tool_name:
            return None
        return json.loads(function["arguments"])[schema.field]
    return json.loads(message["content"])[schema.field]


def _message_value(schema: DecisionSchema, response_json: dict[str, Any]) -> object:
    blocks = response_json["content"]
    if schema.source == "tool":
        block = next(b for b in blocks if b["type"] == "tool_use" and b["name"] == schema.tool_name)
        return block["input"][schema.field]
    block = next(b for b in blocks if b["type"] == "text")
    return json.loads(block["text"])[schema.field]


def _label(value: object) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return None


def token_count(value: object) -> int | None:
    """A token count a provider reported, or None unless it reported an int."""
    # bool is an int subclass, and a token count is never a flag.
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def decoded_json(upstream: httpx.Response, body: bytes) -> dict[str, Any] | None:
    """The provider's buffered response as a JSON object, decompressing it, or None."""
    # A second Response over the buffered bytes runs httpx's own gzip, deflate, brotli and zstd
    # decoders, which reading the stream raw skipped.
    try:
        parsed = httpx.Response(upstream.status_code, headers=upstream.headers, content=body).json()
    except (httpx.DecodingError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_answer(schema: DecisionSchema, response_json: dict[str, Any]) -> str | None:
    """The typed field the model filled in, as a string, or None if the response does not carry it."""
    try:
        value = _answered_value(schema, response_json)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return _label(value)


def usage_tokens(response_json: dict[str, Any]) -> tuple[int | None, int | None]:
    """The prompt and completion counts a provider reported, each None unless it reported an int."""
    usage = response_json.get("usage")
    if not isinstance(usage, dict):
        return None, None
    return token_count(usage.get("prompt_tokens")), token_count(usage.get("completion_tokens"))


def extract_message_answer(schema: DecisionSchema, response_json: dict[str, Any]) -> str | None:
    """The typed field a Messages response filled in, as a string, or None if it does not carry it."""
    try:
        value = _message_value(schema, response_json)
    except (KeyError, StopIteration, TypeError, ValueError):
        return None
    return _label(value)


def message_usage_tokens(response_json: dict[str, Any]) -> tuple[int | None, int | None]:
    """The input and output counts a Messages response reported, each None unless it was an int."""
    usage = response_json.get("usage")
    if not isinstance(usage, dict):
        return None, None
    return token_count(usage.get("input_tokens")), token_count(usage.get("output_tokens"))
