import pytest

from stuntd.decisions.schema import (
    MAX_FIELDS,
    DecisionSchema,
    MultiFieldSchema,
    declares_message_schema,
    declares_schema,
    detect_schema,
)


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


def test_free_text_is_not_a_decision():
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


def typed_properties(count):
    return {f"f{i}": {"type": "boolean"} for i in range(count)}


def form(properties, **extra):
    return chat(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "x",
                "schema": {"type": "object", "properties": properties, **extra},
            },
        }
    )


def test_two_typed_fields_are_a_multi_field_decision():
    schema = detect_schema(
        form(
            {
                "category": {"type": "string", "enum": ["billing", "tech"]},
                "urgent": {"type": "boolean"},
                "score": {"type": "integer"},
            }
        )
    )
    assert isinstance(schema, MultiFieldSchema)
    assert [(f.field, f.kind, f.options) for f in schema.fields] == [
        ("category", "choice", ("billing", "tech")),
        ("urgent", "boolean", ()),
        ("score", "number", ()),
    ]
    assert {f.source for f in schema.fields} == {"response_format"}


def test_max_fields_are_a_decision():
    schema = detect_schema(form(typed_properties(MAX_FIELDS)))
    assert isinstance(schema, MultiFieldSchema)
    assert len(schema.fields) == MAX_FIELDS


def test_more_than_max_fields_is_free_text():
    assert detect_schema(form(typed_properties(MAX_FIELDS + 1))) is None


@pytest.mark.parametrize(
    "untyped",
    [
        {"type": "string"},
        {"type": "array", "items": {"type": "string"}},
        {"type": "object", "properties": {"a": {"type": "boolean"}}},
        {"type": "string", "enum": []},
    ],
    ids=["text", "array", "nested", "empty-enum"],
)
def test_a_field_that_is_not_typed_makes_the_request_free_text(untyped):
    assert detect_schema(form({"a": {"type": "boolean"}, "b": untyped})) is None


def test_optional_field_is_still_a_field():
    schema = detect_schema(
        form({"a": {"type": "boolean"}, "b": {"type": "boolean"}}, required=["a"])
    )
    assert isinstance(schema, MultiFieldSchema)
    assert len(schema.fields) == 2


def test_multi_field_tool_carries_the_tool_name_on_every_field():
    schema = detect_schema(
        chat(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "triage",
                        "parameters": {"type": "object", "properties": typed_properties(2)},
                    },
                }
            ]
        )
    )
    assert isinstance(schema, MultiFieldSchema)
    assert {(f.source, f.tool_name) for f in schema.fields} == {("tool", "triage")}


def test_field_canonical_is_a_single_field_schema():
    schema = detect_schema(form({"a": {"type": "boolean"}, "b": {"type": "integer"}}))
    assert [f.canonical for f in schema.fields] == [
        '{"properties":{"a":{"type":"boolean"}},"type":"object"}',
        '{"properties":{"b":{"type":"integer"}},"type":"object"}',
    ]


def test_single_field_stays_a_decision_schema():
    assert isinstance(detect_schema(form(typed_properties(1))), DecisionSchema)


def test_chat_request_with_a_json_schema_declares_a_schema():
    assert declares_schema(form({"a": {"type": "string"}}))


def test_chat_request_with_one_function_tool_declares_a_schema():
    tool = {"type": "function", "function": {"name": "note", "parameters": {"type": "object"}}}
    assert declares_schema(chat(tools=[tool]))


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"response_format": {"type": "json_object"}},
        {"tools": []},
        {"tools": [{"type": "function", "function": {"name": "a"}}] * 2},
    ],
    ids=["plain", "json-object", "no-tools", "two-tools"],
)
def test_chat_request_without_one_schema_declares_none(extra):
    assert not declares_schema(chat(**extra))


def test_message_request_with_an_output_schema_declares_a_schema():
    assert declares_message_schema({"output_config": {"format": {"type": "json_schema"}}})


def test_message_request_with_one_named_tool_declares_a_schema():
    assert declares_message_schema({"tools": [{"name": "note", "input_schema": {}}]})


def test_message_request_without_one_schema_declares_none():
    assert not declares_message_schema({"messages": [], "tools": [{"name": "a"}, {"name": "b"}]})
