from __future__ import annotations

import json
import math
from typing import Any

from stuntd.decisions.schema import DecisionSchema

__all__ = ["completion_body", "label_value"]


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


def _message(
    schema: DecisionSchema, value: str | bool | int | float, request_id: str
) -> dict[str, Any]:
    content = json.dumps({schema.field: value}, separators=(",", ":"), ensure_ascii=False)
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
    schema: DecisionSchema,
    label: str,
    model: str,
    created: int,
    request_id: str,
) -> bytes:
    """Builds an OpenAI chat.completion body carrying the decoded label for the given schema."""
    if schema.source not in ("response_format", "tool"):
        raise ValueError(f"unknown schema source {schema.source!r}")
    value = label_value(schema.kind, label)
    finish_reason = "tool_calls" if schema.source == "tool" else "stop"
    body = {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": _message(schema, value, request_id),
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
