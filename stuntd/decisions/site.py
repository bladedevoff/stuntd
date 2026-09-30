from __future__ import annotations

import hashlib
import re
from typing import Any

from stuntd.decisions.schema import Schema

__all__ = ["content_text", "field_site", "input_text", "site_key", "system_prompt"]

_KEY_LENGTH = 16
_FIELD_HASH_LENGTH = 8
_SITE_NAME = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def content_text(content: object) -> str:
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
        content_text(m.get("content"))
        for m in _message_dicts(messages)
        if m.get("role") == "system"
    )


def input_text(messages: object) -> str:
    """Joins every other message as `role: content`."""
    return "\n".join(
        f"{m.get('role')}: {content_text(m.get('content'))}"
        for m in _message_dicts(messages)
        if m.get("role") != "system"
    )


def site_key(schema: Schema, system: str, override: str | None = None) -> str:
    """Names the decision site: the same schema and system prompt always give the same key."""
    if override:
        return override
    material = schema.canonical + "\n" + system
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:_KEY_LENGTH]


def field_site(site: str, field: str) -> str:
    """Names the site of one field of a multi-field decision."""
    named = f"{site}.{field}"
    # Windows drops a trailing dot from a folder name, so the head would land under another site.
    if _SITE_NAME.fullmatch(named) and not named.endswith("."):
        return named
    return f"{site}.f{hashlib.sha256(field.encode('utf-8')).hexdigest()[:_FIELD_HASH_LENGTH]}"
