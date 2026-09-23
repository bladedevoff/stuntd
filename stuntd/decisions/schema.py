from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

__all__ = ["DecisionSchema", "detect_schema", "single_typed_field"]


@dataclass(frozen=True)
class DecisionSchema:
    """The single typed field a request asks the model to fill in."""

    kind: str
    field: str
    options: tuple[str, ...]
    source: str
    tool_name: str | None
    canonical: str


def single_typed_field(schema: object) -> tuple[str, str, tuple[str, ...]] | None:
    """Returns the one typed field of an object schema as (field, kind, options), or None."""
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return None
    props = schema.get("properties")
    if not isinstance(props, dict) or len(props) != 1:
        return None
    field, spec = next(iter(props.items()))
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


def _canonical(schema: object) -> str:
    return json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def detect_schema(request: dict[str, Any]) -> DecisionSchema | None:
    """Returns the single typed field a chat request asks for, or None if it is free text."""
    fmt = request.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        json_schema = fmt.get("json_schema")
        schema = json_schema.get("schema") if isinstance(json_schema, dict) else None
        found = single_typed_field(schema)
        if found is None:
            return None
        field, kind, options = found
        return DecisionSchema(kind, field, options, "response_format", None, _canonical(schema))
    tools = request.get("tools")
    if not isinstance(tools, list) or len(tools) != 1:
        return None
    tool = tools[0]
    if not isinstance(tool, dict) or tool.get("type") != "function":
        return None
    fn = tool.get("function")
    if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
        return None
    params = fn.get("parameters")
    found = single_typed_field(params)
    if found is None:
        return None
    field, kind, options = found
    return DecisionSchema(kind, field, options, "tool", fn["name"], _canonical(params))
