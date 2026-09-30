import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from stuntd.decisions.schema import detect_schema
from stuntd.decisions.site import field_site, site_key
from stuntd.proxy.app import build_app
from stuntd.settings import Settings

pytestmark = pytest.mark.anyio

PROPERTIES = {
    "category": {"type": "string", "enum": ["billing", "tech"]},
    "urgent": {"type": "boolean"},
    "score": {"type": "integer"},
}

REQUEST = {
    "model": "gpt-x",
    "messages": [
        {"role": "system", "content": "Triage."},
        {"role": "user", "content": "card declined"},
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "triage",
            "schema": {"type": "object", "properties": PROPERTIES, "required": ["category"]},
        },
    },
}

TOOL_REQUEST = {
    "model": "gpt-x",
    "messages": REQUEST["messages"],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "triage",
                "parameters": {"type": "object", "properties": PROPERTIES},
            },
        }
    ],
}


def completion(answer):
    return {
        "choices": [{"message": {"role": "assistant", "content": json.dumps(answer)}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }


def tool_completion(answer):
    call = {"function": {"name": "triage", "arguments": json.dumps(answer)}}
    return {"choices": [{"message": {"role": "assistant", "tool_calls": [call]}}]}


async def relay(request_body, provider_body, headers=None):
    calls = []

    async def provider(request: Request):
        calls.append(await request.body())
        return Response(json.dumps(provider_body), media_type="application/json")

    upstream = Starlette(routes=[Route("/{path:path}", provider, methods=["POST"])])
    app = build_app(
        Settings(upstream="http://upstream"), transport=httpx.ASGITransport(app=upstream)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        response = await client.post("/v1/chat/completions", json=request_body, headers=headers)
    store = app.state.store
    captured = {info.site: store.examples(info.site) for info in store.sites()}
    await app.state.client.aclose()
    store.close()
    return response, captured, calls


async def test_multi_field_request_is_forwarded_and_answered_by_the_provider(data_dir):
    answer = {"category": "billing", "urgent": True, "score": 3}
    response, _, calls = await relay(REQUEST, completion(answer))
    assert response.json()["choices"][0]["message"]["content"] == json.dumps(answer)
    assert json.loads(calls[0]) == REQUEST


async def test_multi_field_request_captures_one_example_per_field(data_dir):
    answer = {"category": "billing", "urgent": True, "score": 3}
    response, captured, _ = await relay(REQUEST, completion(answer))
    parent = site_key(detect_schema(REQUEST), "Triage.")
    assert response.headers["x-stuntd"] == f"collect; site={parent}"
    assert {
        site: [(e.input_text, e.answer) for e in examples] for site, examples in captured.items()
    } == {
        f"{parent}.category": [("user: card declined", "billing")],
        f"{parent}.urgent": [("user: card declined", "true")],
        f"{parent}.score": [("user: card declined", "3")],
    }


async def test_multi_field_tool_call_captures_every_field(data_dir):
    answer = {"category": "tech", "urgent": False, "score": 1}
    _, captured, _ = await relay(TOOL_REQUEST, tool_completion(answer))
    assert sorted(captured) == sorted(
        field_site(site_key(detect_schema(TOOL_REQUEST), "Triage."), field) for field in PROPERTIES
    )


async def test_multi_field_omitted_optional_field_records_nothing(data_dir):
    response, captured, _ = await relay(REQUEST, completion({"category": "billing"}))
    parent = site_key(detect_schema(REQUEST), "Triage.")
    assert response.status_code == 200
    assert list(captured) == [f"{parent}.category"]


async def test_multi_field_answer_without_any_field_records_nothing(data_dir):
    response, captured, _ = await relay(REQUEST, completion({"other": 1}))
    assert response.status_code == 200
    assert captured == {}


async def test_multi_field_site_header_names_every_field_site(data_dir):
    response, captured, _ = await relay(
        REQUEST,
        completion({"category": "billing", "urgent": True, "score": 3}),
        headers={"X-Stuntd-Site": "triage"},
    )
    assert response.headers["x-stuntd"] == "collect; site=triage"
    assert sorted(captured) == ["triage.category", "triage.score", "triage.urgent"]


async def test_multi_field_stream_passes_through(data_dir):
    response, captured, _ = await relay(
        dict(REQUEST, stream=True), completion({"category": "tech"})
    )
    assert response.headers["x-stuntd"] == "passthrough; reason=streaming"
    assert captured == {}
