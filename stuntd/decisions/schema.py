from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MAX_FIELDS",
    "DecisionSchema",
    "MultiFieldSchema",
    "Schema",
    "detect_message_schema",
    "detect_schema",
    "single_typed_field",
]

MAX_FIELDS = 8


@dataclass(frozen=True)
class DecisionSchema:
    """The single typed field a request asks the model to fill in."""

    kind: str
    field: str
    options: tuple[str, ...]
    source: str
    tool_name: str | None
    canonical: str

    @property
    def fields(self) -> tuple[DecisionSchema, ...]:
        return (self,)


@dataclass(frozen=True)
class MultiFieldSchema:
    """Several typed fields asked in one request; each is a decision of its own."""

    fields: tuple[DecisionSchema, ...]
    canonical: str


Schema = DecisionSchema | MultiFieldSchema


def _typed_field(field: str, spec: object) -> tuple[str, str, tuple[str, ...]] | None:
    if not isinstance(spec, dict):
        return None
    kind = spec.get("type")
    enum = spec.get("enum")
    if (
        kind == "string"
        and isinstance(enum, list)
        and enum
        and all(isinstance(o, str) for o in enum)
    ):
        return field, "choice", tuple(enum)
    if kind == "boolean":
        return field, "boolean", ()
    if kind in ("integer", "number"):
        return field, "number", ()
    return None


def _typed_fields(
    schema: object,
) -> list[tuple[str, str, tuple[str, ...], object]] | None:
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return None
    props = schema.get("properties")
    if not isinstance(props, dict) or not 0 < len(props) <= MAX_FIELDS:
        return None
    found = []
    for field, spec in props.items():
        typed = _typed_field(field, spec)
        if typed is None:
            return None
        found.append((*typed, spec))
    return found


def single_typed_field(schema: object) -> tuple[str, str, tuple[str, ...]] | None:
    """Returns the one typed field of an object schema as (field, kind, options), or None."""
    found = _typed_fields(schema)
    return found[0][:3] if found is not None and len(found) == 1 else None


def _canonical(schema: object) -> str:
    return json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _schema(schema: object, source: str, tool_name: str | None) -> Schema | None:
    found = _typed_fields(schema)
    if found is None:
        return None
    if len(found) == 1:
        field, kind, options, _ = found[0]
        return DecisionSchema(kind, field, options, source, tool_name, _canonical(schema))
    fields = tuple(
        DecisionSchema(
            kind,
            field,
            options,
            source,
            tool_name,
            _canonical({"type": "object", "properties": {field: spec}}),
        )
        for field, kind, options, spec in found
    )
    return MultiFieldSchema(fields, _canonical(schema))


def detect_schema(request: dict[str, Any]) -> Schema | None:
    """Returns the typed fields a chat request asks for, or None if it is free text."""
    fmt = request.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        json_schema = fmt.get("json_schema")
        schema = json_schema.get("schema") if isinstance(json_schema, dict) else None
        return _schema(schema, "response_format", None)
    tools = request.get("tools")
    if not isinstance(tools, list) or len(tools) != 1:
        return None
    tool = tools[0]
    if not isinstance(tool, dict) or tool.get("type") != "function":
        return None
    fn = tool.get("function")
    if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
        return None
    return _schema(fn.get("parameters"), "tool", fn["name"])


def detect_message_schema(request: dict[str, Any]) -> Schema | None:
    """Returns the typed fields a Messages request asks for, or None if it is free text."""
    output_config = request.get("output_config")
    fmt = output_config.get("format") if isinstance(output_config, dict) else None
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        return _schema(fmt.get("schema"), "response_format", None)
    tools = request.get("tools")
    if not isinstance(tools, list) or len(tools) != 1:
        return None
    tool = tools[0]
    if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
        return None
    return _schema(tool.get("input_schema"), "tool", tool["name"])
