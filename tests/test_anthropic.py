import json

import httpx
import pytest
from anthropic.types import Message
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from stuntd.decisions.schema import detect_message_schema, detect_schema
from stuntd.decisions.site import field_site, site_key
from stuntd.proxy.app import build_app
from stuntd.proxy.dialects import ANTHROPIC_MESSAGES, DIALECTS, OPENAI_CHAT
from stuntd.settings import Settings

pytestmark = pytest.mark.anyio

VERDICT = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["allow", "block"]}},
}
TRIAGE = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["billing", "tech"]},
        "urgent": {"type": "boolean"},
    },
}
FREE_TEXT_TOOL = {
    "name": "note",
    "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
}

JSON_REQUEST = {
    "model": "claude-x",
    "max_tokens": 64,
    "system": "Moderate.",
    "messages": [{"role": "user", "content": "buy pills"}],
    "output_config": {"format": {"type": "json_schema", "schema": VERDICT}},
}

TOOL_REQUEST = {
    "model": "claude-x",
    "max_tokens": 64,
    "system": [{"type": "text", "text": "Moderate."}],
    "messages": [{"role": "user", "content": [{"type": "text", "text": "buy pills"}]}],
    "tools": [{"name": "mod", "description": "Moderate", "input_schema": VERDICT}],
}

OPENAI_REQUEST = {
    "model": "gpt-x",
    "messages": [
        {"role": "system", "content": "Moderate."},
        {"role": "user", "content": "buy pills"},
    ],
    "response_format": {"type": "json_schema", "json_schema": {"name": "mod", "schema": VERDICT}},
}

TRIAGE_REQUEST = {
    **JSON_REQUEST,
    "output_config": {"format": {"type": "json_schema", "schema": TRIAGE}},
}


def message(content, stop_reason):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-x",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }


def text_response(answer):
    return message([{"type": "text", "text": json.dumps(answer)}], "end_turn")


def tool_response(answer):
    block = {"type": "tool_use", "id": "toolu_1", "name": "mod", "input": answer}
    return message([block], "tool_use")


async def relay(request_body, provider_body, headers=None):
    calls = []

    async def provider(request: Request):
        calls.append((await request.body(), request.headers))
        return Response(json.dumps(provider_body), media_type="application/json")

    upstream = Starlette(routes=[Route("/{path:path}", provider, methods=["POST"])])
    app = build_app(
        Settings(upstream="http://upstream"), transport=httpx.ASGITransport(app=upstream)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        response = await client.post("/v1/messages", json=request_body, headers=headers)
    store = app.state.store
    captured = {info.site: store.examples(info.site) for info in store.sites()}
    await app.state.client.aclose()
    store.close()
    return response, captured, calls


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("POST", "/v1/messages", True),
        ("POST", "/proxy/v1/messages", True),
        ("GET", "/v1/messages", False),
        ("POST", "/v1/messages/count_tokens", False),
        ("POST", "/v1/chat/completions", False),
    ],
    ids=["v1", "prefixed", "get", "count-tokens", "openai"],
)
def test_anthropic_messages_matches_only_post_to_messages(method, path, expected):
    assert ANTHROPIC_MESSAGES.matches(method, path) is expected


def test_dialects_lists_openai_before_anthropic():
    assert DIALECTS == (OPENAI_CHAT, ANTHROPIC_MESSAGES)


@pytest.mark.parametrize(
    ("request_body", "source"),
    [(JSON_REQUEST, "response_format"), (TOOL_REQUEST, "tool")],
    ids=["output-config", "tool"],
)
def test_detect_message_schema_reads_both_forms(request_body, source):
    schema = detect_message_schema(request_body)
    assert (schema.field, schema.kind, schema.options, schema.source) == (
        "verdict",
        "choice",
        ("allow", "block"),
        source,
    )


def test_detect_message_schema_reads_several_fields():
    schema = detect_message_schema(TRIAGE_REQUEST)
    assert [field.field for field in schema.fields] == ["category", "urgent"]


@pytest.mark.parametrize(
    "request_body",
    [
        {"model": "claude-x", "messages": []},
        {**TOOL_REQUEST, "tools": [{"type": "web_search_20250305", "name": "web_search"}]},
        {**TOOL_REQUEST, "tools": [TOOL_REQUEST["tools"][0], TOOL_REQUEST["tools"][0]]},
        {**TOOL_REQUEST, "tools": [FREE_TEXT_TOOL]},
    ],
    ids=["plain", "server-tool", "two-tools", "free-text-tool"],
)
def test_detect_message_schema_refuses_free_text(request_body):
    assert detect_message_schema(request_body) is None


@pytest.mark.parametrize("request_body", [JSON_REQUEST, TOOL_REQUEST], ids=["string", "blocks"])
def test_anthropic_messages_reads_system_prompt_and_input_text(request_body):
    assert ANTHROPIC_MESSAGES.system_prompt(request_body) == "Moderate."
    assert ANTHROPIC_MESSAGES.input_text(request_body) == "user: buy pills"


def test_anthropic_messages_reads_answers_and_usage():
    schema = detect_message_schema(JSON_REQUEST)
    tool_schema = detect_message_schema(TOOL_REQUEST)
    answer = {"verdict": "block"}
    assert ANTHROPIC_MESSAGES.extract_answer(schema, text_response(answer)) == "block"
    assert ANTHROPIC_MESSAGES.extract_answer(tool_schema, tool_response(answer)) == "block"
    assert ANTHROPIC_MESSAGES.extract_answer(schema, tool_response(answer)) is None
    assert ANTHROPIC_MESSAGES.extract_answer(schema, {"error": "x"}) is None
    assert ANTHROPIC_MESSAGES.usage_tokens(text_response(answer)) == (12, 3)
    assert ANTHROPIC_MESSAGES.usage_tokens({}) == (None, None)


def test_anthropic_site_key_matches_the_openai_form_of_the_same_decision():
    anthropic = site_key(
        detect_message_schema(JSON_REQUEST), ANTHROPIC_MESSAGES.system_prompt(JSON_REQUEST)
    )
    openai = site_key(detect_schema(OPENAI_REQUEST), OPENAI_CHAT.system_prompt(OPENAI_REQUEST))
    assert anthropic == openai


def test_anthropic_text_body_validates_as_a_message():
    schema = detect_message_schema(JSON_REQUEST)
    raw = ANTHROPIC_MESSAGES.completion_body(schema, ("block",), "claude-x", 1, "msg_stuntd_abc")
    parsed = Message.model_validate_json(raw)
    assert (parsed.id, parsed.type, parsed.role, parsed.model, parsed.stop_reason) == (
        "msg_stuntd_abc",
        "message",
        "assistant",
        "claude-x",
        "end_turn",
    )
    assert parsed.stop_sequence is None
    assert (parsed.usage.input_tokens, parsed.usage.output_tokens) == (0, 0)
    assert json.loads(parsed.content[0].text) == {"verdict": "block"}


def test_anthropic_tool_body_validates_as_a_message():
    schema = detect_message_schema(TOOL_REQUEST)
    raw = ANTHROPIC_MESSAGES.completion_body(schema, ("block",), "claude-x", 1, "msg_stuntd_abc")
    parsed = Message.model_validate_json(raw)
    assert parsed.stop_reason == "tool_use"
    assert (parsed.content[0].type, parsed.content[0].name) == ("tool_use", "mod")
    assert parsed.content[0].input == {"verdict": "block"}


def test_anthropic_body_carries_every_field_in_schema_order():
    raw = ANTHROPIC_MESSAGES.completion_body(
        detect_message_schema(TRIAGE_REQUEST), ("tech", "true"), "claude-x", 1, "msg_stuntd_abc"
    )
    assert Message.model_validate_json(raw).content[0].text == '{"category":"tech","urgent":true}'


async def test_relay_captures_a_message_and_forwards_anthropic_headers(data_dir):
    headers = {"x-api-key": "sk-test", "anthropic-version": "2023-06-01", "anthropic-beta": "b1"}
    response, captured, calls = await relay(
        JSON_REQUEST, text_response({"verdict": "block"}), headers
    )
    site = site_key(detect_message_schema(JSON_REQUEST), "Moderate.")
    assert response.headers["x-stuntd"] == f"collect; site={site}"
    assert [(e.input_text, e.answer) for e in captured[site]] == [("user: buy pills", "block")]
    assert {name: calls[0][1][name] for name in headers} == headers


async def test_relay_captures_a_tool_use_answer_per_field(data_dir):
    request = {**TOOL_REQUEST, "tools": [{"name": "mod", "input_schema": TRIAGE}]}
    _, captured, _ = await relay(request, tool_response({"category": "tech", "urgent": False}))
    site = site_key(detect_message_schema(request), "Moderate.")
    assert {name: [e.answer for e in examples] for name, examples in captured.items()} == {
        field_site(site, "category"): ["tech"],
        field_site(site, "urgent"): ["false"],
    }


@pytest.mark.parametrize(
    ("request_body", "reason"),
    [
        ({**JSON_REQUEST, "stream": True}, "streaming"),
        (
            {"model": "claude-x", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
            "no-schema",
        ),
        ({**TOOL_REQUEST, "tools": [FREE_TEXT_TOOL]}, "unsupported-schema"),
    ],
    ids=["stream", "free-text", "free-text-tool"],
)
async def test_relay_passes_through_untouched(data_dir, request_body, reason):
    response, captured, calls = await relay(request_body, text_response({"verdict": "block"}))
    assert response.headers["x-stuntd"] == f"passthrough; reason={reason}"
    assert response.json() == text_response({"verdict": "block"})
    assert json.loads(calls[0][0]) == request_body
    assert captured == {}
