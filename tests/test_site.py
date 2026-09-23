from stuntd.decisions.schema import detect_schema
from stuntd.decisions.site import input_text, site_key, system_prompt

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
    assert site_key(schema, REQ["messages"]) == site_key(schema, other["messages"])
    assert len(site_key(schema, REQ["messages"])) == 16


def test_key_changes_with_system_prompt_or_schema():
    schema = detect_schema(REQ)
    changed_prompt = [
        {"role": "system", "content": "You rate jokes."},
        REQ["messages"][1],
    ]
    assert site_key(schema, REQ["messages"]) != site_key(schema, changed_prompt)
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
    assert site_key(schema, REQ["messages"]) != site_key(changed_schema, REQ["messages"])


def test_override_wins():
    assert site_key(detect_schema(REQ), REQ["messages"], override="moderation") == "moderation"


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
