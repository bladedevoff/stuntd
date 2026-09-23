import gzip
import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from stuntd.proxy.app import build_app
from stuntd.settings import Settings

pytestmark = pytest.mark.anyio

SEEN: list[dict] = []


async def echo(request: Request):
    body = await request.body()
    SEEN.append(
        {
            "path": request.url.path,
            "raw_path": request.scope["raw_path"],
            "query": request.url.query,
            "headers": dict(request.headers),
            "body": body,
        }
    )
    if request.url.path.endswith("/stream"):

        async def gen():
            for i in range(3):
                yield f"data: {json.dumps({'i': i})}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"x-up": "1"})
    if request.url.path.endswith("/raw"):
        return Response(content=body, media_type="application/json", headers={"x-up": "1"})
    if request.url.path.endswith("/stuntd"):
        return Response(
            content=b"ok",
            media_type="text/plain",
            headers={"x-stuntd": "collect; site=upstream", "x-up": "1"},
        )
    if request.url.path.endswith("/cookies"):
        response = Response(content=b"ok", media_type="text/plain", headers={"set-cookie": "a=1"})
        response.raw_headers.append((b"set-cookie", b"b=2"))
        return response
    if request.url.path.endswith("/gz"):
        return Response(
            content=gzip.compress(b'{"ok":true}'),
            media_type="application/json",
            headers={"content-encoding": "gzip"},
        )
    if request.url.path.endswith("/unicode"):
        response = Response(content=b"ok", media_type="text/plain")
        response.raw_headers.append((b"x-uni", "café 漢".encode()))
        return response
    return JSONResponse({"ok": True})


fake_upstream = Starlette(
    routes=[Route("/{path:path}", echo, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])]
)


class Recording(httpx.AsyncBaseTransport):
    """Keeps the last upstream response so a test can assert it was closed."""

    def __init__(self, inner: httpx.AsyncBaseTransport):
        self.inner = inner
        self.last: httpx.Response | None = None

    async def handle_async_request(self, request):
        self.last = await self.inner.handle_async_request(request)
        return self.last


@pytest.fixture
async def client(data_dir):
    SEEN.clear()
    app = build_app(
        Settings(upstream="http://upstream"),
        transport=httpx.ASGITransport(app=fake_upstream),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as c:
        yield c
    await app.state.client.aclose()
    app.state.store.close()


async def test_body_and_field_order_are_untouched(client):
    body = b'{"model":"gpt-x","zeta":1,"alpha":{"b":2,"a":1}}'
    r = await client.post(
        "/v1/raw?x=1",
        content=body,
        headers={"content-type": "application/json", "authorization": "Bearer k"},
    )
    assert r.status_code == 200
    assert r.content == body
    assert SEEN[0]["body"] == body
    assert SEEN[0]["query"] == "x=1"
    assert SEEN[0]["headers"]["authorization"] == "Bearer k"
    assert "host" not in SEEN[0]["headers"] or SEEN[0]["headers"]["host"] != "proxy"
    assert r.headers["x-up"] == "1"
    assert r.headers["x-stuntd"].startswith("passthrough")


async def test_streaming_chunks_arrive_in_order(client):
    async with client.stream("POST", "/v1/stream", content=b"{}") as r:
        chunks = [c async for c in r.aiter_raw()]
    joined = b"".join(chunks)
    assert joined == b'data: {"i": 0}\n\ndata: {"i": 1}\n\ndata: {"i": 2}\n\ndata: [DONE]\n\n'
    assert r.headers["content-type"].startswith("text/event-stream")


async def test_duplicate_response_headers_are_all_forwarded(client):
    r = await client.get("/v1/cookies")
    assert r.headers.get_list("set-cookie") == ["a=1", "b=2"]


async def test_no_headers_are_added_to_the_upstream_request(client):
    for name in ("accept", "accept-encoding", "user-agent"):
        client.headers.pop(name, None)
    await client.post("/v1/raw", content=b"{}", headers={"content-type": "application/json"})
    sent = set(SEEN[0]["headers"])
    assert "accept-encoding" not in sent
    assert "user-agent" not in sent
    assert "accept" not in sent


async def test_percent_encoded_path_is_forwarded_verbatim(client):
    r = await client.post("/v1/models/a%2Fb", content=b"{}")
    assert r.status_code == 200
    assert SEEN[0]["raw_path"] == b"/v1/models/a%2Fb"


async def test_options_is_forwarded_with_stuntd_header(client):
    r = await client.request("OPTIONS", "/v1/anything")
    assert r.status_code == 200
    assert r.headers["x-stuntd"] == "passthrough; reason=no-schema"


async def test_gzip_body_is_forwarded_undecoded(client):
    async with client.stream("GET", "/v1/gz") as r:
        raw = b"".join([chunk async for chunk in r.aiter_raw()])
    assert raw.startswith(b"\x1f\x8b")
    assert gzip.decompress(raw) == b'{"ok":true}'
    assert r.headers["content-encoding"] == "gzip"


async def test_content_length_is_dropped_and_reason_is_exact(client):
    r = await client.post(
        "/v1/raw", content=b'{"a":1}', headers={"content-type": "application/json"}
    )
    assert "content-length" not in r.headers
    assert r.headers["x-stuntd"] == "passthrough; reason=no-schema"
    assert r.content == b'{"a":1}'


async def test_upstream_header_outside_latin1_is_forwarded_untouched(data_dir):
    recording = Recording(httpx.ASGITransport(app=fake_upstream))
    app = build_app(Settings(upstream="http://upstream"), transport=recording)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as c:
        r = await c.get("/v1/unicode")
    await app.state.client.aclose()
    app.state.store.close()
    assert r.status_code == 200
    assert dict(r.headers.raw)[b"x-uni"] == "café 漢".encode()
    assert recording.last.is_closed


async def test_upstream_down_gives_502(data_dir):
    class Down(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.ConnectError("refused")

    app = build_app(Settings(upstream="http://upstream"), transport=Down())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as c:
        r = await c.post("/v1/chat/completions", content=b"{}")
    await app.state.client.aclose()
    app.state.store.close()
    assert r.status_code == 502
    assert r.content == b'{"error":"upstream unreachable"}'
    assert r.headers["x-stuntd"] == "passthrough; reason=upstream-error"


async def test_upstream_cookie_is_not_replayed_to_the_next_caller(data_dir):
    SEEN.clear()
    app = build_app(
        Settings(upstream="http://upstream"), transport=httpx.ASGITransport(app=fake_upstream)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as first:
        r = await first.get("/v1/cookies")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as second:
        await second.post("/v1/raw", content=b"{}", headers={"content-type": "application/json"})
    await app.state.client.aclose()
    app.state.store.close()
    assert r.headers.get_list("set-cookie") == ["a=1", "b=2"]
    assert "cookie" not in SEEN[1]["headers"]


async def test_upstream_stuntd_header_is_replaced(client):
    r = await client.get("/v1/stuntd")
    assert r.headers.get_list("x-stuntd") == ["passthrough; reason=no-schema"]
    assert "content-length" not in r.headers
    assert r.headers["x-up"] == "1"


async def test_websocket_handshake_is_closed(data_dir):
    app = build_app(
        Settings(upstream="http://upstream"),
        transport=httpx.ASGITransport(app=fake_upstream),
    )
    sent = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "scheme": "ws",
        "path": "/v1/realtime",
        "raw_path": b"/v1/realtime",
        "root_path": "",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "server": ("proxy", 80),
    }
    await app(scope, receive, send)
    await app.state.client.aclose()
    app.state.store.close()
    assert sent == [{"type": "websocket.close", "code": 1000, "reason": ""}]


class Refusing(httpx.AsyncBaseTransport):
    """Fails the test if the proxy sends anything to a provider."""

    async def handle_async_request(self, request):
        raise AssertionError(f"{request.url} reached a provider")


DECISION = {
    "model": "gpt-x",
    "messages": [{"role": "user", "content": "buy pills"}],
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


@pytest.mark.parametrize(
    "payload",
    [DECISION, {**DECISION, "stream": True}, {"model": "gpt-x", "messages": []}],
    ids=["decision", "streamed-decision", "no-schema"],
)
async def test_without_an_upstream_the_openai_path_answers_502(data_dir, payload):
    app = build_app(Settings(upstream=""), transport=Refusing())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as c:
        r = await c.post("/v1/chat/completions", json=payload)
    await app.state.client.aclose()
    store = app.state.store
    assert r.status_code == 502
    assert json.loads(r.content) == {
        "error": {"message": "stuntd: no upstream configured", "type": "stuntd"}
    }
    assert r.headers["x-stuntd"] == "passthrough; reason=no-upstream"
    assert store.sites() == [] and store.decision_counts() == {}
    store.close()


async def test_without_an_upstream_the_request_body_is_never_read(data_dir):
    async def unreadable():
        raise AssertionError("the request body was read")
        yield b"{}"

    app = build_app(Settings(upstream=""), transport=Refusing())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as c:
        r = await c.post("/v1/chat/completions", content=unreadable())
    await app.state.client.aclose()
    app.state.store.close()
    assert r.status_code == 502
    assert r.headers["x-stuntd"] == "passthrough; reason=no-upstream"
