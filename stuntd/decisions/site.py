from __future__ import annotations

import hashlib
from typing import Any

from stuntd.decisions.schema import DecisionSchema

__all__ = ["input_text", "site_key", "system_prompt"]

_KEY_LENGTH = 16


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        )
    return ""


def _message_dicts(messages: object) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    return [m for m in messages if isinstance(m, dict)]


def system_prompt(messages: object) -> str:
    """Joins the content of every system message."""
    return "\n".join(
        _text(m.get("content")) for m in _message_dicts(messages) if m.get("role") == "system"
    )


def input_text(messages: object) -> str:
    """Joins every other message as `role: content`."""
    return "\n".join(
        f"{m.get('role')}: {_text(m.get('content'))}"
        for m in _message_dicts(messages)
        if m.get("role") != "system"
    )


def site_key(schema: DecisionSchema, messages: object, override: str | None = None) -> str:
    """Names the decision site: the same schema and system prompt always give the same key."""
    if override:
        return override
    material = schema.canonical + "\n" + system_prompt(messages)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:_KEY_LENGTH]
