from __future__ import annotations

import contextlib
import json
import logging
import re
import sqlite3
import time
from collections.abc import AsyncIterator
from http.cookiejar import CookieJar, DefaultCookiePolicy
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTasks
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route, request_response
from starlette.types import Receive, Scope, Send
from starlette.websockets import WebSocketClose

from stuntd.decisions.schema import DecisionSchema, Schema
from stuntd.decisions.site import field_site, site_key
from stuntd.proxy.capture import decoded_json
from stuntd.proxy.dialects import DIALECTS, Dialect
from stuntd.proxy.headers import (
    DROPPED_WHEN_BUFFERED,
    DROPPED_WHEN_STREAMED,
    forwardable,
    relayed_headers,
    stuntd_header,
)
from stuntd.proxy.jev import JevRoutes
from stuntd.serve.modes import MODE_CHECK, MODE_COLLECT, MODE_LIVE, MODE_SHADOW, SiteState
from stuntd.serve.retrain import Retrainer
from stuntd.serve.runtime import DeciderLike, Outcome, Runtime, input_hash
from stuntd.settings import Settings, database_path
from stuntd.store.db import Capture, Store
from stuntd.store.redact import Redactor

__all__ = ["Proxy", "build_app"]

_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
# The caller alone decides about compression, and the provider must see its agent string, not ours.
_STRIPPED_CLIENT_HEADERS = ("accept", "accept-encoding", "user-agent")
_SITE_HEADER = "x-stuntd-site"
# The caller names the site, so the value is accepted only as a bare identifier. fullmatch, not $,
# which would also accept a trailing newline.
_SITE_OVERRIDE = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_MODEL_ERROR = "model-error"
_NOT_LIVE = "not-live"
_LEARNING_OFF = "learning-off"
_NO_UPSTREAM = "no-upstream"
_REQUEST_ID_CHARS = 12

_log = logging.getLogger(__name__)


class Proxy:
    """Owns the upstream client and the capture store for the lifetime of the app."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
        decider: DeciderLike | None = None,
        config: Path | None = None,
    ) -> None:
        self.settings = settings
        self.decider = decider
        self.store: Store | None = None
        self.runtime: Runtime | None = None
        self.retrainer: Retrainer | None = None
        if settings.learn:
            # enabled governs only the built-in rules; patterns from the settings file always run.
            redactor = Redactor(settings.redaction_patterns, builtin=settings.redact)
            self.store = Store(
                database_path(settings),
                redactor,
                max_rows=settings.max_rows,
                max_age_days=settings.max_age_days,
            )
            self.runtime = Runtime(settings, self.store, decider)
            if settings.auto_retrain:
                self.retrainer = Retrainer(settings, self.store, self.runtime, config)
        self.client = httpx.AsyncClient(
            base_url=settings.upstream,
            transport=transport,
            timeout=_TIMEOUT,
            # A Set-Cookie sent to one caller must never be replayed into another caller's
            # request, so the jar accepts no domain and therefore stores and sends nothing.
            # httpx keeps a bare jar as given; an httpx.Cookies wrapper would be copied into a
            # fresh jar and lose the policy.
            cookies=CookieJar(policy=DefaultCookiePolicy(allowed_domains=[])),
        )
        for name in _STRIPPED_CLIENT_HEADERS:
            self.client.headers.pop(name, None)

    def __repr__(self) -> str:
        # httpx.URL hides the password of an upstream that carries credentials.
        return f"Proxy(upstream={self.client.base_url!r}, store={self.store!r})"

    async def relay(self, request: Request) -> Response:
        if not self.settings.upstream:
            # Nothing here has anywhere to go, so the caller's body is never read and the store
            # never sees this request.
            return _no_upstream()
        body = await request.body()
        # scope["path"] is percent-decoded, which would turn an escaped %2F into a separator.
        raw_path = request.scope.get("raw_path")
        path = raw_path.decode("latin-1") if raw_path else request.url.path
        target = path + (f"?{request.url.query}" if request.url.query else "")
        upstream_request = self.client.build_request(
            request.method,
            target,
            headers=[
                (name, value)
                for name, value in forwardable(request.headers.items())
                if name.lower() != _SITE_HEADER
            ],
            content=body,
        )
        # Both are None exactly when learning is off, and then nothing here is recognised.
        store, runtime = self.store, self.runtime
        if store is None or runtime is None:
            return await self._stream(upstream_request, _LEARNING_OFF)
        decision = _decision_request(request, body)
        if isinstance(decision, str):
            return await self._stream(upstream_request, decision)
        dialect, payload, schema = decision
        return await self._serve(
            store, runtime, upstream_request, dialect, payload, schema, _site_override(request)
        )

    async def buffered(self, upstream_request: httpx.Request) -> tuple[httpx.Response, bytes, int]:
        """Sends one request and reads the whole answer: the response, its raw bytes, the latency."""
        started = time.monotonic()
        upstream = await self.client.send(upstream_request, stream=True)
        try:
            # The raw bytes are what the caller gets, so the relayed content-length stays true
            # even when the provider compressed the body.
            body = b"".join([chunk async for chunk in upstream.aiter_raw()])
        finally:
            await upstream.aclose()
        return upstream, body, int((time.monotonic() - started) * 1000)

    async def aclose(self) -> None:
        await self.client.aclose()
        if self.store is not None:
            self.store.close()

    async def _stream(self, upstream_request: httpx.Request, reason: str) -> Response:
        try:
            upstream = await self.client.send(upstream_request, stream=True)
        except httpx.HTTPError:
            return _upstream_error(None)
        # Until the response owns the stream, any failure would strand an open connection.
        try:
            return _passthrough(upstream, reason)
        except BaseException:
            await upstream.aclose()
            raise

    async def _serve(
        self,
        store: Store,
        runtime: Runtime,
        upstream_request: httpx.Request,
        dialect: Dialect,
        payload: dict[str, Any],
        schema: Schema,
        override: str | None,
    ) -> Response:
        site = site_key(schema, dialect.system_prompt(payload), override)
        text = dialect.input_text(payload)
        if isinstance(schema, DecisionSchema):
            states = [runtime.state(site)]
        else:
            states = [runtime.state(field_site(site, field.field)) for field in schema.fields]
        if all(state.mode == MODE_LIVE for state in states):
            return await self._serve_live(
                store, runtime, upstream_request, dialect, payload, schema, site, states, text
            )
        mode = MODE_SHADOW if any(state.mode == MODE_SHADOW for state in states) else MODE_COLLECT
        if isinstance(schema, DecisionSchema):
            reason = runtime.state_reason(site)
        elif any(state.model is not None for state in states):
            index = next(i for i, state in enumerate(states) if state.mode != MODE_LIVE)
            reason = _field_reason(_NOT_LIVE, schema, site, index)
        else:
            reason = None
        return await self._collect(
            store,
            runtime,
            upstream_request,
            dialect,
            payload,
            schema,
            site,
            states,
            text,
            stuntd_header(mode, site=site, reason=reason),
        )

    async def _serve_live(
        self,
        store: Store,
        runtime: Runtime,
        upstream_request: httpx.Request,
        dialect: Dialect,
        payload: dict[str, Any],
        schema: Schema,
        site: str,
        states: list[SiteState],
        text: str,
    ) -> Response:
        outcomes = await runtime.live_fields(states, text, site)
        answers = [outcome.answer for outcome in outcomes if outcome.answer is not None]
        if len(answers) < len(states) or outcomes[-1].mode != MODE_LIVE:
            reason = _field_reason(outcomes[-1].reason, schema, site, len(outcomes) - 1)
            return await self._collect(
                store,
                runtime,
                upstream_request,
                dialect,
                payload,
                schema,
                site,
                states,
                text,
                stuntd_header(MODE_COLLECT, site=site, reason=reason),
                dict(zip((state.site for state in states), outcomes, strict=False)),
            )
        request_id = dialect.request_id_prefix + input_hash(site, text)[:_REQUEST_ID_CHARS]
        try:
            body = dialect.completion_body(
                schema, answers, str(payload.get("model", "")), int(time.time()), request_id
            )
        except ValueError:
            # A caller can pin a site whose labels do not fit the schema this request asks for,
            # and only the provider can answer that one.
            _log.exception("live answer not built")
            return await self._collect(
                store,
                runtime,
                upstream_request,
                dialect,
                payload,
                schema,
                site,
                states,
                text,
                stuntd_header(MODE_COLLECT, site=site, reason=_MODEL_ERROR),
            )
        # Only the decision is recorded: no provider answered, so there is no capture to train on.
        for state, outcome in zip(states, outcomes, strict=True):
            runtime.record_live(state, outcome)
        response = Response(content=body, status_code=200, media_type="application/json")
        response.headers["X-Stuntd"] = stuntd_header(
            MODE_LIVE,
            site=site,
            confidence=min(o.confidence for o in outcomes if o.confidence is not None),
        )
        return response

    async def _collect(
        self,
        store: Store,
        runtime: Runtime,
        upstream_request: httpx.Request,
        dialect: Dialect,
        payload: dict[str, Any],
        schema: Schema,
        site: str,
        states: list[SiteState],
        text: str,
        header: str,
        outcomes: dict[str, Outcome] | None = None,
    ) -> Response:
        try:
            upstream, body, latency_ms = await self.buffered(upstream_request)
        except httpx.HTTPError:
            return _upstream_error(site)
        response = Response(content=body, status_code=upstream.status_code)
        response.raw_headers = relayed_headers(upstream.headers.raw, header, DROPPED_WHEN_BUFFERED)
        answered = _answered(upstream, body, dialect, schema)
        if answered is None:
            return response
        parsed, answers = answered
        try:
            for field, state, answer in zip(schema.fields, states, answers, strict=True):
                if answer is not None:
                    _record(
                        store, dialect, payload, field, state.site, text, answer, parsed, latency_ms
                    )
                    if self.retrainer is not None:
                        self.retrainer.record(state.site)
        except sqlite3.Error:
            # The provider has already produced and billed this completion, so a database that
            # cannot take the capture must still not cost the caller the answer.
            _log.exception("capture not recorded")
        background = BackgroundTasks()
        for state, answer in zip(states, answers, strict=True):
            if answer is None:
                continue
            outcome = outcomes.get(state.site) if outcomes is not None else None
            if state.mode == MODE_SHADOW:
                # The shadow pass waits for the response to leave, so the model never adds its own
                # latency to the caller's.
                background.add_task(runtime.shadow, state, text, answer)
            elif outcome is not None and outcome.reason == MODE_CHECK:
                runtime.compare(state, outcome, answer)
        response.background = background
        return response


def _record(
    store: Store,
    dialect: Dialect,
    payload: dict[str, Any],
    schema: DecisionSchema,
    site: str,
    text: str,
    answer: str,
    response_json: dict[str, Any],
    latency_ms: int,
) -> None:
    prompt_tokens, completion_tokens = dialect.usage_tokens(response_json)
    store.record(
        Capture(
            site,
            schema.canonical,
            schema.kind,
            text,
            answer,
            str(payload.get("model", "")),
            latency_ms,
            prompt_tokens,
            completion_tokens,
        )
    )


def _decision_request(
    request: Request, body: bytes
) -> tuple[Dialect, dict[str, Any], Schema] | str:
    dialect = next((d for d in DIALECTS if d.matches(request.method, request.url.path)), None)
    if dialect is None:
        return "no-schema"
    try:
        payload = json.loads(body)
    except ValueError:
        return "no-schema"
    if not isinstance(payload, dict):
        return "no-schema"
    schema = dialect.detect(payload)
    if schema is None:
        return "no-schema"
    if payload.get("stream") is True:
        return "streaming"
    return dialect, payload, schema


def _site_override(request: Request) -> str | None:
    value = request.headers.get(_SITE_HEADER)
    if value is None or _SITE_OVERRIDE.fullmatch(value) is None:
        return None
    # A name of nothing but dots passes the charset and still walks up a directory, so training,
    # which turns the name into a folder, never sees one.
    if set(value) == {"."}:
        return None
    return value


def _answered(
    upstream: httpx.Response, body: bytes, dialect: Dialect, schema: Schema
) -> tuple[dict[str, Any], list[str | None]] | None:
    """The provider's parsed response and its answer per field, or None when it answered none."""
    if upstream.status_code != 200:
        return None
    parsed = decoded_json(upstream, body)
    if parsed is None:
        return None
    answers = [dialect.extract_answer(field, parsed) for field in schema.fields]
    if all(answer is None for answer in answers):
        return None
    return parsed, answers


def _field_reason(reason: str | None, schema: Schema, site: str, index: int) -> str | None:
    if reason is None or isinstance(schema, DecisionSchema):
        return reason
    # A field name is the caller's, so the header carries the token its site name is made of.
    field = field_site(site, schema.fields[index].field).removeprefix(f"{site}.")
    return f"{reason}:{field}"


def _upstream_error(site: str | None) -> Response:
    # A request that had already been read as a decision says which site went unanswered, the
    # way every other answer on that path does.
    mode = "passthrough" if site is None else MODE_COLLECT
    return JSONResponse(
        {"error": "upstream unreachable"},
        status_code=502,
        headers={"X-Stuntd": stuntd_header(mode, site=site, reason="upstream-error")},
    )


def _no_upstream() -> Response:
    # The shape an OpenAI SDK reads an error out of, so the caller sees the message rather than
    # a bare 502.
    return JSONResponse(
        {"error": {"message": "stuntd: no upstream configured", "type": "stuntd"}},
        status_code=502,
        headers={"X-Stuntd": stuntd_header("passthrough", reason=_NO_UPSTREAM)},
    )


def _passthrough(upstream: httpx.Response, reason: str) -> Response:
    pairs = relayed_headers(
        upstream.headers.raw, stuntd_header("passthrough", reason=reason), DROPPED_WHEN_STREAMED
    )

    async def stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    response = StreamingResponse(stream(), status_code=upstream.status_code)
    # Starlette would fold repeated headers and re-encode values; both break a byte-exact relay.
    response.raw_headers = pairs
    return response


def build_app(
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
    decider: DeciderLike | None = None,
    config: Path | None = None,
) -> Starlette:
    """Wires a Starlette app around one Proxy: a single mount, the lifespan and app.state."""
    proxy = Proxy(settings, transport, decider, config)
    asgi_app = request_response(proxy.relay)

    async def endpoint(scope: Scope, receive: Receive, send: Send) -> None:
        # Only http and websocket scopes reach a mounted app, and nothing here speaks websocket.
        if scope["type"] != "http":
            await WebSocketClose()(scope, receive, send)
            return
        await asgi_app(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await proxy.aclose()

    jev = JevRoutes(proxy)
    # A Route with a function endpoint answers 405 to anything it was not given; which methods
    # exist is the provider's business. The Jev routes come first, or the mount below would relay
    # them to the OpenAI provider.
    app = Starlette(
        routes=[
            Route("/v1/systemone", jev.systemone, methods=["POST"]),
            Route("/v1/models", jev.models, methods=["GET"]),
            Mount("/", app=endpoint),
        ],
        lifespan=lifespan,
    )
    app.state.proxy = proxy
    app.state.client = proxy.client
    app.state.store = proxy.store
    app.state.runtime = proxy.runtime
    app.state.settings = settings
    return app
