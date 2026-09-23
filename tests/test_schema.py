import pytest

from stuntd.decisions.schema import detect_schema


def chat(**extra):
    base = {"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}]}
    base.update(extra)
    return base


def test_enum_in_response_format_is_a_choice():
    req = chat(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "mod",
                "schema": {
                    "type": "object",
                    "properties": {"verdict": {"type": "string", "enum": ["allow", "block"]}},
                    "required": ["verdict"],
                    "additionalProperties": False,
                },
            },
        }
    )
    schema = detect_schema(req)
    assert schema is not None
    assert (schema.kind, schema.field, schema.options, schema.source) == (
        "choice",
        "verdict",
        ("allow", "block"),
        "response_format",
    )


def test_boolean_and_number_fields():
    boolean = chat(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "b",
                "schema": {
                    "type": "object",
                    "properties": {"spam": {"type": "boolean"}},
                },
            },
        }
    )
    number = chat(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "n",
                "schema": {
                    "type": "object",
                    "properties": {"score": {"type": "integer"}},
                },
            },
        }
    )
    assert detect_schema(boolean).kind == "boolean"
    assert detect_schema(number).kind == "number"


def test_tool_with_single_enum_parameter():
    req = chat(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "route",
                    "parameters": {
                        "type": "object",
                        "properties": {"queue": {"type": "string", "enum": ["billing", "tech"]}},
                    },
                },
            }
        ]
    )
    schema = detect_schema(req)
    assert schema.source == "tool" and schema.tool_name == "route" and schema.kind == "choice"


def test_multi_field_and_free_text_are_not_decisions():
    two_fields = chat(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "x",
                "schema": {
                    "type": "object",
                    "properties": {"a": {"type": "boolean"}, "b": {"type": "boolean"}},
                },
            },
        }
    )
    free = chat(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "x",
                "schema": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                },
            },
        }
    )
    assert detect_schema(two_fields) is None
    assert detect_schema(free) is None
    assert detect_schema(chat()) is None
    assert detect_schema(chat(response_format={"type": "json_object"})) is None


def test_two_tools_are_not_a_decision():
    req = chat(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "a",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "boolean"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "b",
                    "parameters": {
                        "type": "object",
                        "properties": {"y": {"type": "boolean"}},
                    },
                },
            },
        ]
    )
    assert detect_schema(req) is None


def test_canonical_ignores_key_order():
    a = chat(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "m",
                "schema": {
                    "type": "object",
                    "properties": {"v": {"enum": ["x", "y"], "type": "string"}},
                },
            },
        }
    )
    b = chat(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "m",
                "schema": {
                    "properties": {"v": {"type": "string", "enum": ["x", "y"]}},
                    "type": "object",
                },
            },
        }
    )
    assert detect_schema(a).canonical == detect_schema(b).canonical


@pytest.mark.parametrize(
    "request_body",
    [
        chat(response_format={"type": "json_schema", "json_schema": "mod"}),
        chat(tools=[{"type": "function", "function": "route"}]),
        chat(tools=[{"type": "function"}]),
        chat(
            tools=[
                {
                    "type": "custom",
                    "function": {
                        "name": "route",
                        "parameters": {
                            "type": "object",
                            "properties": {"queue": {"type": "boolean"}},
                        },
                    },
                }
            ]
        ),
        chat(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "e",
                    "schema": {
                        "type": "object",
                        "properties": {"v": {"type": "string", "enum": []}},
                    },
                },
            }
        ),
        chat(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "e",
                    "schema": {
                        "type": "object",
                        "properties": {"v": {"type": "string", "enum": ["ok", 7]}},
                    },
                },
            }
        ),
        chat(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "p",
                    "schema": {
                        "type": "object",
                        "properties": [{"v": {"type": "boolean"}}],
                    },
                },
            }
        ),
    ],
    ids=[
        "json-schema-not-a-table",
        "tool-function-not-a-table",
        "tool-without-a-function",
        "tool-type-is-not-function",
        "empty-enum",
        "enum-with-a-non-string",
        "properties-not-a-table",
    ],
)
def test_malformed_payloads_are_not_decisions(request_body):
    assert detect_schema(request_body) is None
