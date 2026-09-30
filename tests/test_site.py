import re

import pytest

from stuntd.decisions.schema import detect_schema
from stuntd.decisions.site import field_site, input_text, site_key, system_prompt

REQ = {
    "model": "m",
    "messages": [
        {"role": "system", "content": "You moderate comments."},
        {"role": "user", "content": "Buy cheap pills"},
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "mod",
            "schema": {
                "type": "object",
                "properties": {"verdict": {"type": "string", "enum": ["allow", "block"]}},
            },
        },
    },
}


def test_key_is_stable_and_ignores_user_content():
    schema = detect_schema(REQ)
    other = dict(REQ, messages=[REQ["messages"][0], {"role": "user", "content": "Nice post"}])
    assert site_key(schema, system_prompt(REQ["messages"])) == site_key(
        schema, system_prompt(other["messages"])
    )
    assert len(site_key(schema, system_prompt(REQ["messages"]))) == 16


def test_key_changes_with_system_prompt_or_schema():
    schema = detect_schema(REQ)
    changed_prompt = [
        {"role": "system", "content": "You rate jokes."},
        REQ["messages"][1],
    ]
    assert site_key(schema, system_prompt(REQ["messages"])) != site_key(
        schema, system_prompt(changed_prompt)
    )
    wider = {
        "type": "json_schema",
        "json_schema": {
            "name": "mod",
            "schema": {
                "type": "object",
                "properties": {"verdict": {"type": "string", "enum": ["allow", "block", "review"]}},
            },
        },
    }
    changed_schema = detect_schema(dict(REQ, response_format=wider))
    assert site_key(schema, system_prompt(REQ["messages"])) != site_key(
        changed_schema, system_prompt(REQ["messages"])
    )


def test_override_wins():
    assert (
        site_key(detect_schema(REQ), system_prompt(REQ["messages"]), override="moderation")
        == "moderation"
    )


def test_system_prompt_and_input_text_split_roles():
    assert system_prompt(REQ["messages"]) == "You moderate comments."
    assert input_text(REQ["messages"]) == "user: Buy cheap pills"


def test_content_parts_are_flattened():
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        }
    ]
    assert input_text(messages) == "user: a\nb"


def test_malformed_messages_do_not_raise():
    messages = [
        7,
        {"role": "system", "content": None},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": 5},
                "raw",
                {"type": "text", "text": "kept"},
            ],
        },
    ]
    assert system_prompt(messages) == ""
    assert input_text(messages) == "user: kept"
    assert system_prompt({"role": "system", "content": "not a list"}) == ""
    assert input_text({"role": "user", "content": "not a list"}) == ""


def test_field_site_joins_site_and_field():
    assert field_site("abc123", "category") == "abc123.category"


@pytest.mark.parametrize(
    "field",
    ["needs human", "kategorieä", "", "name.", "x" * 64],
    ids=["space", "non-ascii", "empty", "trailing-dot", "too-long"],
)
def test_field_site_hashes_a_field_that_does_not_fit_the_site_name(field):
    site = field_site("abc123", field)
    assert re.fullmatch(r"abc123\.f[0-9a-f]{8}", site)
    assert site == field_site("abc123", field)


def test_field_site_hashes_distinct_fields_apart():
    assert field_site("abc123", "a b") != field_site("abc123", "a c")
