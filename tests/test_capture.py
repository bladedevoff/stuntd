import gzip
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
from stuntd.proxy.capture import extract_answer, usage_tokens
from stuntd.settings import Settings

pytestmark = pytest.mark.anyio

MOD_REQUEST = {
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

UPSTREAM_BODY = json.dumps(
    {
        "id": "x",
        "model": "gpt-x",
        "choices": [{"message": {"role": "assistant", "content": '{"verdict": "block"}'}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }
).encode()


async def upstream(request: Request):
    return Response(content=UPSTREAM_BODY, media_type="application/json", headers={"x-up": "1"})


fake = Starlette(routes=[Route("/{path:path}", upstream, methods=["POST"])])


def app_for(handler):
    provider = Starlette(routes=[Route("/{path:path}", handler, methods=["POST"])])
    return build_app(
        Settings(upstream="http://upstream"), transport=httpx.ASGITransport(app=provider)
    )


async def decision_through(handler, headers=None):
    app = app_for(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as c:
        response = await c.post("/v1/chat/completions", json=MOD_REQUEST, headers=headers)
    stats = app.state.store.stats()
    await app.state.client.aclose()
    app.state.store.close()
    return response, stats


@pytest.fixture
async def proxy(data_dir):
    app = build_app(Settings(upstream="http://upstream"), transport=httpx.ASGITransport(app=fake))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as c:
        yield c, app
    await app.state.client.aclose()
    app.state.store.close()


def test_extract_answer_from_response_format_and_tool():
    schema = detect_schema(MOD_REQUEST)
    assert extract_answer(schema, json.loads(UPSTREAM_BODY)) == "block"
    tool_req = {
        "messages": [],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "route",
                    "parameters": {"type": "object", "properties": {"spam": {"type": "boolean"}}},
                },
            }
        ],
    }
    tool_schema = detect_schema(tool_req)
    resp = {
        "choices": [
            {
                "message": {
                    "tool_calls": [{"function": {"name": "route", "arguments": '{"spam": true}'}}]
                }
            }
        ]
    }
    assert extract_answer(tool_schema, resp) == "true"
    wrong_tool = {
        "choices": [
            {"message": {"tool_calls": [{"function": {"name": "other", "arguments": "{}"}}]}}
        ]
    }
    assert extract_answer(tool_schema, wrong_tool) is None


async def test_typed_request_is_recorded_and_response_untouched(proxy):
    client, app = proxy
    r = await client.post(
        "/v1/chat/completions", json=MOD_REQUEST, headers={"X-Stuntd-Site": "moderation"}
    )
    assert r.status_code == 200
    assert r.content == UPSTREAM_BODY
    assert r.headers["x-up"] == "1"
    assert r.headers["x-stuntd"] == "collect; site=moderation"
    stats = app.state.store.stats()
    assert stats == [
        {
            "site": "moderation",
            "kind": "choice",
            "count": 1,
            "first_at": stats[0]["first_at"],
            "last_at": stats[0]["last_at"],
        }
    ]
    row = app.state.store._conn.execute(
        "select input_text, answer, model, prompt_tokens from captures"
    ).fetchone()
    assert row == ("user: buy pills", "block", "gpt-x", 12)


async def test_streaming_typed_request_is_not_recorded(proxy):
    client, app = proxy
    r = await client.post("/v1/chat/completions", json=dict(MOD_REQUEST, stream=True))
    assert r.headers["x-stuntd"] == "passthrough; reason=streaming"
    assert app.state.store.stats() == []


async def test_untyped_chat_request_is_not_recorded(proxy):
    client, app = proxy
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.headers["x-stuntd"] == "passthrough; reason=no-schema"
    assert app.state.store.stats() == []


async def test_rejected_site_header_falls_back_to_the_hashed_key(proxy):
    client, app = proxy
    r = await client.post(
        "/v1/chat/completions", json=MOD_REQUEST, headers={"X-Stuntd-Site": "../evil"}
    )
    hashed = site_key(detect_schema(MOD_REQUEST), MOD_REQUEST["messages"])
    assert r.headers["x-stuntd"] == f"collect; site={hashed}"
    assert [entry["site"] for entry in app.state.store.stats()] == [hashed]


async def test_dots_only_site_header_falls_back_to_the_hashed_key(proxy):
    client, app = proxy
    r = await client.post("/v1/chat/completions", json=MOD_REQUEST, headers={"X-Stuntd-Site": ".."})
    hashed = site_key(detect_schema(MOD_REQUEST), MOD_REQUEST["messages"])
    assert r.headers["x-stuntd"] == f"collect; site={hashed}"
    assert [entry["site"] for entry in app.state.store.stats()] == [hashed]


async def test_site_header_is_never_forwarded_upstream(data_dir):
    seen: list[httpx.Headers] = []

    class Recording(httpx.AsyncBaseTransport):
        def __init__(self, inner):
            self.inner = inner

        async def handle_async_request(self, request):
            seen.append(request.headers)
            return await self.inner.handle_async_request(request)

    app = build_app(
        Settings(upstream="http://upstream"),
        transport=Recording(httpx.ASGITransport(app=fake)),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as c:
        await c.post(
            "/v1/chat/completions", json=MOD_REQUEST, headers={"X-Stuntd-Site": "moderation"}
        )
        await c.post("/v1/embeddings", json={"input": "x"}, headers={"X-Stuntd-Site": "../evil"})
    await app.state.client.aclose()
    app.state.store.close()
    assert [("x-stuntd-site" in headers) for headers in seen] == [False, False]


async def test_compressed_response_is_relayed_undecoded_and_still_recorded(data_dir):
    async def compressed(request: Request):
        return Response(
            content=gzip.compress(UPSTREAM_BODY),
            headers={"content-encoding": "gzip", "content-type": "application/json"},
        )

    app = app_for(compressed)
    async with (
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c,
        c.stream(
            "POST",
            "/v1/chat/completions",
            json=MOD_REQUEST,
            headers={"X-Stuntd-Site": "moderation"},
        ) as r,
    ):
        raw = b"".join([chunk async for chunk in r.aiter_raw()])
    stats = app.state.store.stats()
    await app.state.client.aclose()
    app.state.store.close()
    assert r.headers["content-encoding"] == "gzip"
    assert int(r.headers["content-length"]) == len(raw)
    assert gzip.decompress(raw) == UPSTREAM_BODY
    assert [(entry["site"], entry["count"]) for entry in stats] == [("moderation", 1)]


async def test_store_failure_leaves_the_response_intact(proxy, caplog):
    client, app = proxy
    app.state.store.close()
    r = await client.post(
        "/v1/chat/completions", json=MOD_REQUEST, headers={"X-Stuntd-Site": "moderation"}
    )
    assert r.status_code == 200
    assert r.content == UPSTREAM_BODY
    assert r.headers["x-stuntd"] == "collect; site=moderation"
    assert "capture not recorded" in caplog.text


async def test_upstream_down_on_the_decision_path_gives_502(data_dir):
    class Down(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.ConnectError("refused")

    app = build_app(Settings(upstream="http://upstream"), transport=Down())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as c:
        r = await c.post(
            "/v1/chat/completions", json=MOD_REQUEST, headers={"X-Stuntd-Site": "moderation"}
        )
    await app.state.client.aclose()
    app.state.store.close()
    assert r.status_code == 502
    assert r.content == b'{"error":"upstream unreachable"}'
    assert r.headers["x-stuntd"] == "collect; site=moderation; reason=upstream-error"


async def test_configured_patterns_survive_disabled_builtin_redaction(data_dir):
    app = build_app(
        Settings(upstream="http://upstream", redact=False, redaction_patterns=[r"ACC-\d{6}"]),
        transport=httpx.ASGITransport(app=fake),
    )
    payload = dict(
        MOD_REQUEST,
        messages=[
            MOD_REQUEST["messages"][0],
            {"role": "user", "content": "bob@example.com ACC-123456"},
        ],
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as c:
        await c.post("/v1/chat/completions", json=payload)
    row = app.state.store._conn.execute("select input_text from captures").fetchone()
    await app.state.client.aclose()
    app.state.store.close()
    assert row == ("user: bob@example.com [redacted]",)


async def test_body_failure_on_the_decision_path_gives_502(data_dir):
    class Stalling(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"id": "x"'
            raise httpx.ReadTimeout("body stalled")

    class HalfAnswer(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(
                200, headers={"content-type": "application/json"}, stream=Stalling()
            )

    app = build_app(Settings(upstream="http://upstream"), transport=HalfAnswer())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as c:
        r = await c.post(
            "/v1/chat/completions", json=MOD_REQUEST, headers={"X-Stuntd-Site": "moderation"}
        )
    stats = app.state.store.stats()
    await app.state.client.aclose()
    app.state.store.close()
    assert r.status_code == 502
    assert r.content == b'{"error":"upstream unreachable"}'
    assert r.headers["x-stuntd"] == "collect; site=moderation; reason=upstream-error"
    assert stats == []


async def test_unparsable_decision_response_is_relayed_and_not_recorded(data_dir):
    async def not_json(request: Request):
        return Response(content=b"not json at all", media_type="text/plain")

    r, stats = await decision_through(not_json)
    assert r.status_code == 200
    assert r.content == b"not json at all"
    assert r.headers["x-stuntd"].startswith("collect; site=")
    assert stats == []


async def test_failed_decision_response_is_relayed_and_not_recorded(data_dir):
    async def rate_limited(request: Request):
        return Response(
            content=b'{"error": "rate limit"}', status_code=429, media_type="application/json"
        )

    r, stats = await decision_through(rate_limited)
    assert r.status_code == 429
    assert r.content == b'{"error": "rate limit"}'
    assert r.headers["x-stuntd"].startswith("collect; site=")
    assert stats == []


async def test_upstream_stuntd_header_is_replaced_on_the_decision_path(data_dir):
    async def opinionated(request: Request):
        return Response(
            content=UPSTREAM_BODY,
            media_type="application/json",
            headers={"x-stuntd": "collect; site=upstream"},
        )

    r, stats = await decision_through(opinionated, headers={"X-Stuntd-Site": "moderation"})
    assert r.headers.get_list("x-stuntd") == ["collect; site=moderation"]
    assert int(r.headers["content-length"]) == len(UPSTREAM_BODY)
    assert r.content == UPSTREAM_BODY
    assert [entry["site"] for entry in stats] == ["moderation"]


@pytest.mark.parametrize(
    ("response_json", "expected"),
    [
        ({}, (None, None)),
        ({"usage": {"prompt_tokens": 12, "completion_tokens": 3}}, (12, 3)),
        ({"usage": {"prompt_tokens": True, "completion_tokens": 3}}, (None, 3)),
        ({"usage": {"prompt_tokens": "12", "completion_tokens": None}}, (None, None)),
    ],
    ids=["no-usage", "both-counts", "bool-is-not-a-count", "non-int-is-dropped"],
)
def test_usage_tokens_keeps_only_integer_counts(response_json, expected):
    assert usage_tokens(response_json) == expected
