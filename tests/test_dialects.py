import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from stuntd.decisions.schema import detect_schema
from stuntd.decisions.site import site_key
from stuntd.proxy.app import build_app
from stuntd.proxy.dialects import DIALECTS, OPENAI_CHAT
from stuntd.settings import Settings

pytestmark = pytest.mark.anyio

REQUEST = {
    "model": "gpt-x",
    "messages": [
        {"role": "system", "content": "Moderate."},
        {"role": "user", "content": "buy pills"},
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

RESPONSE = {
    "choices": [{"message": {"role": "assistant", "content": '{"verdict": "block"}'}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
}

KNOWN_SITE = "2ed31b07fc388d96"


def test_site_key_of_a_known_request_is_unchanged():
    assert site_key(detect_schema(REQUEST), "Moderate.") == KNOWN_SITE


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("POST", "/v1/chat/completions", True),
        ("POST", "/chat/completions", True),
        ("POST", "/openai/deployments/gpt/chat/completions", True),
        ("GET", "/v1/chat/completions", False),
        ("POST", "/v1/completions", False),
        ("POST", "/v1/chat/completions/extra", False),
    ],
    ids=["v1", "bare", "prefixed", "get", "other-path", "trailing-segment"],
)
def test_openai_chat_matches_only_post_to_chat_completions(method, path, expected):
    assert OPENAI_CHAT.matches(method, path) is expected


def test_openai_chat_is_the_dialect_for_chat_completions():
    assert DIALECTS[0] is OPENAI_CHAT


def test_openai_chat_reads_the_request_and_the_response():
    schema = OPENAI_CHAT.detect(REQUEST)
    assert schema == detect_schema(REQUEST)
    assert OPENAI_CHAT.system_prompt(REQUEST) == "Moderate."
    assert OPENAI_CHAT.input_text(REQUEST) == "user: buy pills"
    assert OPENAI_CHAT.extract_answer(schema, RESPONSE) == "block"
    assert OPENAI_CHAT.usage_tokens(RESPONSE) == (12, 3)


def test_openai_chat_builds_a_completion_body():
    body = json.loads(
        OPENAI_CHAT.completion_body(detect_schema(REQUEST), ("block",), "gpt-x", 1, "id")
    )
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == '{"verdict":"block"}'


async def test_relay_captures_a_chat_completion_under_the_known_site_key(data_dir):
    async def provider(request: Request):
        return Response(json.dumps(RESPONSE), media_type="application/json")

    upstream = Starlette(routes=[Route("/{path:path}", provider, methods=["POST"])])
    app = build_app(
        Settings(upstream="http://upstream"), transport=httpx.ASGITransport(app=upstream)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        response = await client.post("/v1/chat/completions", json=REQUEST)
    stats = app.state.store.stats()
    await app.state.client.aclose()
    app.state.store.close()
    assert response.headers["x-stuntd"] == f"collect; site={KNOWN_SITE}"
    assert [entry["site"] for entry in stats] == [KNOWN_SITE]
