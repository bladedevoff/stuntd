from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

from stuntd.decisions.schema import DecisionSchema, Schema

__all__ = ["completion_body", "label_value", "message_body"]


def label_value(kind: str, label: str) -> str | bool | int | float:
    """Converts a model's text label to the typed value its schema kind expects."""
    if kind == "choice":
        return label
    if kind == "boolean":
        return label == "true"
    if kind == "number":
        try:
            return int(label)
        except ValueError:
            pass
        try:
            value = float(label)
        except ValueError:
            raise ValueError(f"non-numeric label for a number field: {label!r}") from None
        if not math.isfinite(value):
            raise ValueError(f"number label must be finite, got {label!r}")
        return value
    raise ValueError(f"unknown schema kind: {kind!r}")


def _decoded(
    schema: Schema, labels: Sequence[str]
) -> tuple[DecisionSchema, dict[str, str | bool | int | float]]:
    first = schema.fields[0]
    if first.source not in ("response_format", "tool"):
        raise ValueError(f"unknown schema source {first.source!r}")
    values = {
        field.field: label_value(field.kind, label)
        for field, label in zip(schema.fields, labels, strict=True)
    }
    return first, values


def _message(schema: DecisionSchema, content: str, request_id: str) -> dict[str, Any]:
    if schema.source == "tool":
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{request_id}",
                    "type": "function",
                    "function": {"name": schema.tool_name, "arguments": content},
                }
            ],
        }
    return {"role": "assistant", "content": content}


def completion_body(
    schema: Schema,
    labels: Sequence[str],
    model: str,
    created: int,
    request_id: str,
) -> bytes:
    """Builds an OpenAI chat.completion body carrying one decoded label per field of the schema."""
    first, values = _decoded(schema, labels)
    content = json.dumps(values, separators=(",", ":"), ensure_ascii=False)
    finish_reason = "tool_calls" if first.source == "tool" else "stop"
    body = {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": _message(first, content, request_id),
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def message_body(
    schema: Schema,
    labels: Sequence[str],
    model: str,
    created: int,
    request_id: str,
) -> bytes:
    """Builds an Anthropic message body carrying one decoded label per field of the schema."""
    first, values = _decoded(schema, labels)
    if first.source == "tool":
        block = {
            "type": "tool_use",
            "id": f"toolu_{request_id.removeprefix('msg_')}",
            "name": first.tool_name,
            "input": values,
        }
    else:
        block = {
            "type": "text",
            "text": json.dumps(values, separators=(",", ":"), ensure_ascii=False),
        }
    body = {
        "id": request_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [block],
        "stop_reason": "tool_use" if first.source == "tool" else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
