from __future__ import annotations

import json

import pytest

from stuntd.decisions.schema import DecisionSchema
from stuntd.serve.answer import completion_body, label_value

CHOICE = DecisionSchema("choice", "verdict", ("allow", "block"), "response_format", None, "{}")
TOOL = DecisionSchema("number", "priority", (), "tool", "set_priority", "{}")


@pytest.mark.parametrize(
    ("kind", "label", "expected"),
    [
        ("choice", "block", "block"),
        ("boolean", "true", True),
        ("boolean", "false", False),
        ("number", "7", 7),
        ("number", "2.5", 2.5),
    ],
    ids=["choice", "true", "false", "int", "float"],
)
def test_label_value_is_typed(kind, label, expected):
    value = label_value(kind, label)
    assert value == expected and type(value) is type(expected)


def test_response_format_completion_shape():
    body = json.loads(
        completion_body(CHOICE, "block", "gpt-x", 1_700_000_000, "chatcmpl-stuntd-abc")
    )
    assert (
        body["object"] == "chat.completion"
        and body["model"] == "gpt-x"
        and body["id"] == "chatcmpl-stuntd-abc"
    )
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop" and json.loads(choice["message"]["content"]) == {
        "verdict": "block"
    }
    assert body["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_tool_completion_shape():
    body = json.loads(completion_body(TOOL, "3", "gpt-x", 1, "chatcmpl-stuntd-abc"))
    choice = body["choices"][0]
    call = choice["message"]["tool_calls"][0]
    assert choice["finish_reason"] == "tool_calls" and choice["message"]["content"] is None
    assert call["function"]["name"] == "set_priority" and json.loads(
        call["function"]["arguments"]
    ) == {"priority": 3}
    assert call["id"] == "call_chatcmpl-stuntd-abc" and call["type"] == "function"


def test_body_is_compact_utf8():
    raw = completion_body(CHOICE, "block", "модель", 1, "x")
    assert b": " not in raw and "модель".encode() in raw


def test_label_value_number_keeps_float_for_dot_zero():
    value = label_value("number", "7.0")
    assert value == 7.0 and type(value) is float


def test_label_value_unknown_kind_raises():
    with pytest.raises(ValueError, match="'mystery'"):
        label_value("mystery", "x")


def test_label_value_non_numeric_number_raises():
    with pytest.raises(ValueError, match="'not-a-number'"):
        label_value("number", "not-a-number")


@pytest.mark.parametrize("label", ["nan", "inf", "-inf"], ids=["nan", "inf", "-inf"])
def test_label_value_non_finite_number_raises(label):
    with pytest.raises(ValueError, match=f"'{label}'"):
        label_value("number", label)


def test_completion_body_unknown_source_raises():
    unknown = DecisionSchema("choice", "verdict", ("allow", "block"), "mystery", None, "{}")
    with pytest.raises(ValueError, match="'mystery'"):
        completion_body(unknown, "block", "gpt-x", 1, "x")
