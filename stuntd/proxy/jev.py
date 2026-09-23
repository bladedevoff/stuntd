from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import httpx
from anyio import to_thread
from starlette.background import BackgroundTask, BackgroundTasks
from starlette.requests import Request
from starlette.responses import Response

from stuntd.jev.answer import choice_answer, error_body, noul_answer, response_body, score_answer
from stuntd.jev.schema import JevError, Question, SystemOneRequest, kind_for, parse_request
from stuntd.jev.state import serialize_state
from stuntd.proxy.capture import decoded_json, token_count
from stuntd.proxy.headers import DROPPED_WHEN_BUFFERED, forwardable, jev_header, relayed_headers
from stuntd.serve.modes import MODE_CHECK, MODE_LIVE, MODE_SHADOW, SiteState
from stuntd.serve.runtime import DeciderLike, Outcome, Runtime
from stuntd.store.db import Capture

if TYPE_CHECKING:
    from stuntd.proxy.app import Proxy

__all__ = ["JevRoutes"]

# The two modes this route serves in: local, with no Jev provider behind it, and proxy, where
# the paid provider answers and its answers become training data.
_MODE_LOCAL = "local"
_MODE_PROXY = "proxy"
_SYSTEMONE = "/v1/systemone"
_MODELS = "/v1/models"
# This daemon's own protocol headers are never repeated to the provider.
_STUNTD_PREFIX = "x-stuntd"
_UPSTREAM_ERROR = "upstream-error"
_NOT_PARSED = "not-parsed"
_REQUEST_ID = "x-typesafe-request-id"
_REQUEST_ID_CHARS = 12
_NO_RUNTIME = "no-runtime"
_NO_KEY = "no-key"
_BAD_REQUEST = "bad-request"
_MODEL_ERROR = "model-error"
_NOUL_TRUE = 0.5
_MODEL_DESCRIPTION = "Local Laya heads trained by stuntd on this daemon's own traffic."
_RELEASE_DATE = "2026-09-22"

_log = logging.getLogger(__name__)

# One question of a proxied request: what was asked, the site it lands on, and the outcome that
# site's own head came to, if it has one.
_Site = tuple[Question, SiteState, Outcome | None]
_FromHead = tuple[Question, SiteState, Outcome, dict[str, Any]]


class JevRoutes:
    """The Jev endpoints of one proxy: System One answers and the model list."""

    def __init__(self, proxy: Proxy) -> None:
        self._proxy = proxy

    def __repr__(self) -> str:
        return f"JevRoutes(model={self._proxy.settings.jev_model_name!r})"

    async def systemone(self, request: Request) -> Response:
        """Answers one System One request: through the Jev provider when one is configured, from
        the trained heads and the base Laya checkpoint otherwise."""
        if self._proxy.settings.jev_upstream:
            # The provider checks the caller's key itself, so jev_require_key has nothing to add.
            return await self._proxied(request)
        if self._proxy.settings.jev_require_key and not _has_key(request):
            return _error("missing API key", 401, _NO_KEY)
        try:
            parsed = parse_request(await request.body())
        except JevError as exc:
            return _error(exc.message, exc.status, _BAD_REQUEST)
        decider = self._proxy.decider
        if decider is None:
            return _error("serving needs the train extra", 503, _NO_RUNTIME)
        return await self._answers(parsed, decider)

    async def models(self, request: Request) -> Response:
        """Lists the models on offer: the provider's own in proxy mode, otherwise the one model
        this daemon serves, in the shape the SDK's model list expects."""
        if self._proxy.settings.jev_upstream:
            # The relayed answers name the provider's models, so the list must be its own.
            upstream_request = self._upstream_request(request, _MODELS, b"")
            return await self._relayed(upstream_request, jev_header(_MODE_PROXY), (), "")
        body = json.dumps(
            {
                "models": [
                    {
                        "name": self._proxy.settings.jev_model_name,
                        "description": _MODEL_DESCRIPTION,
                        "release_date": _RELEASE_DATE,
                    }
                ]
            },
            separators=(",", ":"),
        ).encode("utf-8")
        return Response(content=body, media_type="application/json")

    async def _answers(self, parsed: SystemOneRequest, decider: DeciderLike) -> Response:
        runtime = self._proxy.runtime
        text = serialize_state(parsed.state)
        answers: dict[str, dict[str, Any]] = {}
        pending: list[Question] = []
        # The zero-shot answers below come from the checkpoint a head was distilled from, so they
        # are no teacher: nothing is compared or recorded here, and no request is sampled for a
        # comparison that has nobody to make it.
        for question in parsed.questions:
            if runtime is not None:
                state = runtime.state(question.site)
                outcome = await self._outcome(runtime, question, state, text, check=False)
                head = None if outcome is None else _from_head(question, state, outcome)
                if outcome is not None and head is not None:
                    answers[question.name] = head
                    runtime.record_live(state, outcome)
                    continue
            pending.append(question)
        live = len(answers)
        try:
            zero_shot, input_tokens = await _zero_shot(decider, parsed.state, pending)
        except RuntimeError:
            _log.exception("zero-shot answers not built")
            return _error("the model could not answer", 503, _MODEL_ERROR)
        missing = [question.name for question in pending if question.name not in zero_shot]
        if missing:
            _log.error("zero-shot answers missing: %s", missing)
            return _error("the model could not answer", 503, _MODEL_ERROR)
        for question in pending:
            answers[question.name] = zero_shot[question.name]
        body = response_body(
            self._proxy.settings.jev_model_name,
            {question.name: answers[question.name] for question in parsed.questions},
            input_tokens,
        )
        response = Response(content=body, media_type="application/json")
        response.headers["X-Stuntd"] = jev_header(
            _MODE_LOCAL,
            questions=len(parsed.questions),
            live=live,
            zeroshot=len(pending),
            learn_off=runtime is None,
        )
        response.headers[_REQUEST_ID] = _request_id(text, parsed.questions)
        return response

    async def _outcome(
        self,
        runtime: Runtime,
        question: Question,
        state: SiteState,
        text: str,
        *,
        check: bool = True,
    ) -> Outcome | None:
        model = state.model
        # A site is named after the question, so a caller who changed its criteria would otherwise
        # get the earlier head's labels back under the new names. The order may differ: training
        # reads the labels off the canonical, which sorts the criteria keys.
        if (
            state.mode != MODE_LIVE
            or model is None
            or sorted(model.labels) != sorted(question.labels)
        ):
            return None
        return await runtime.live(state, text, check)

    async def _proxied(self, request: Request) -> Response:
        body = await request.body()
        upstream_request = self._upstream_request(request, _SYSTEMONE, body)
        try:
            parsed = parse_request(body)
        except JevError as exc:
            # The provider owns the request format, so a body this daemon cannot read still goes
            # to it; the line is what tells a caller why stuntd never learns from their requests.
            _log.warning("the request was not parsed: %s", exc.message)
            header = jev_header(_MODE_PROXY, reason=_NOT_PARSED)
            return await self._relayed(upstream_request, header, (), "")
        runtime = self._proxy.runtime
        text = serialize_state(parsed.state)
        sites: list[_Site] = []
        # Learning off: no site is resolved and no head runs, so the request is relayed whole.
        if runtime is not None:
            for question in parsed.questions:
                state = runtime.state(question.site)
                sites.append((question, state, await self._outcome(runtime, question, state, text)))
            from_heads = _all_from_heads(sites)
            if from_heads is not None:
                return self._local_answer(runtime, from_heads, text)
        # The counts classify the request before the relay, not what was written afterwards.
        header = jev_header(
            _MODE_PROXY,
            questions=len(parsed.questions),
            live=0,
            shadow=sum(1 for _, state, _ in sites if state.mode == MODE_SHADOW),
            check=sum(
                1 for _, _, outcome in sites if outcome is not None and outcome.reason == MODE_CHECK
            ),
            learn_off=runtime is None,
        )
        return await self._relayed(upstream_request, header, sites, text, runtime)

    def _local_answer(
        self, runtime: Runtime, from_heads: Sequence[_FromHead], text: str
    ) -> Response:
        answers: dict[str, dict[str, Any]] = {}
        for question, state, outcome, answer in from_heads:
            answers[question.name] = answer
            runtime.record_live(state, outcome)
        # The provider was never called, so this request spent no tokens of its own.
        body = response_body(self._proxy.settings.jev_model_name, answers, 0)
        response = Response(content=body, media_type="application/json")
        response.headers["X-Stuntd"] = jev_header(
            _MODE_PROXY, questions=len(from_heads), live=len(from_heads), shadow=0, check=0
        )
        response.headers[_REQUEST_ID] = _request_id(
            text, [question for question, _, _, _ in from_heads]
        )
        return response

    async def _relayed(
        self,
        upstream_request: httpx.Request,
        header: str,
        sites: Sequence[_Site],
        text: str,
        runtime: Runtime | None = None,
    ) -> Response:
        try:
            upstream, body, latency_ms = await self._proxy.buffered(upstream_request)
        except httpx.HTTPError:
            return _error("upstream unreachable", 502, _UPSTREAM_ERROR, mode=_MODE_PROXY)
        response = Response(content=body, status_code=upstream.status_code)
        response.raw_headers = relayed_headers(upstream.headers.raw, header, DROPPED_WHEN_BUFFERED)
        if upstream.status_code != 200 or not sites or runtime is None:
            return response
        answered = decoded_json(upstream, body)
        if answered is None:
            _log.error("the provider answered with no JSON object")
            return response
        tasks = self._learn(runtime, sites, text, answered, latency_ms)
        if tasks:
            response.background = BackgroundTasks(tasks)
        return response

    def _learn(
        self,
        runtime: Runtime,
        sites: Sequence[_Site],
        text: str,
        answered: dict[str, Any],
        latency_ms: int,
    ) -> list[BackgroundTask]:
        answers = answered.get("answers")
        if not isinstance(answers, dict):
            _log.error("the provider answered without answers")
            return []
        model = str(answered.get("model", ""))
        input_tokens, output_tokens = _usage(answered)
        captures: list[Capture] = []
        tasks: list[BackgroundTask] = []
        for question, state, outcome in sites:
            label = _provider_label(question, answers.get(question.name))
            if label is None:
                _log.warning("no usable provider answer for question %r", question.name)
                continue
            if self._proxy.settings.learn:
                captures.append(
                    Capture(
                        question.site,
                        question.canonical,
                        kind_for(question),
                        text,
                        label,
                        model,
                        latency_ms,
                        input_tokens,
                        output_tokens,
                    )
                )
            if state.mode == MODE_SHADOW:
                # The shadow pass runs after the answer has left, so the head never adds its own
                # latency to the caller's.
                tasks.append(BackgroundTask(runtime.shadow, state, text, label))
            elif outcome is not None and outcome.reason == MODE_CHECK:
                runtime.compare(state, outcome, label)
        self._record(captures)
        return tasks

    def _record(self, captures: Sequence[Capture]) -> None:
        store = self._proxy.store
        if store is None:
            # Learning off, so settings.learn left the list empty and there is nothing to write.
            return
        try:
            for capture in captures:
                store.record(capture)
        except sqlite3.Error:
            # The provider has already produced and billed these answers, so a database that
            # cannot take them must still not cost the caller the answer.
            _log.exception("captures not recorded")

    def _upstream_request(self, request: Request, path: str, body: bytes) -> httpx.Request:
        headers = [
            (name, value)
            for name, value in forwardable(request.headers.items())
            if not name.lower().startswith(_STUNTD_PREFIX)
        ]
        return self._proxy.client.build_request(
            request.method,
            self._proxy.settings.jev_upstream.rstrip("/") + path,
            headers=headers,
            content=body,
        )


async def _zero_shot(
    decider: DeciderLike, state: object, pending: Sequence[Question]
) -> tuple[dict[str, Any], int]:
    if not pending:
        return {}, 0
    questions = {question.name: _laya_question(question) for question in pending}
    # laya answers in its own untyped shape, which the decider passes through as object.
    reply: Any = await to_thread.run_sync(decider.answer, state, questions)
    return reply["answers"], int(reply["usage"]["input_tokens"])


def _laya_question(question: Question) -> dict[str, object]:
    # laya reads instructions without a default, and a noul question may carry no criteria at all.
    instructions = question.instructions if question.instructions is not None else ""
    laya: dict[str, object] = {"type": question.type, "instructions": instructions}
    if question.criteria is not None:
        laya["criteria"] = question.criteria
    return laya


def _from_head(question: Question, state: SiteState, outcome: Outcome) -> dict[str, Any] | None:
    """The answer a site's own head gives, or None while the provider's mode still decides."""
    probabilities, confidence = outcome.probabilities, outcome.confidence
    model = state.model
    if outcome.mode != MODE_LIVE or probabilities is None or confidence is None or model is None:
        return None
    if question.type == "choice":
        return choice_answer(
            question.labels,
            _in_question_order(model.labels, question.labels, probabilities),
            confidence,
        )
    if question.type == "noul":
        return noul_answer(probabilities)
    # parse_request admits a score question only with a list of levels.
    return score_answer(cast("list[object]", question.criteria), probabilities, confidence)


def _in_question_order(
    labels: Sequence[str], question_labels: Sequence[str], probabilities: Sequence[float]
) -> tuple[float, ...]:
    """A head's probabilities under the question's own label order; the two agree as sets because
    a head is trained off the canonical, which sorts a choice's criteria keys."""
    place = {label: position for position, label in enumerate(labels)}
    return tuple(probabilities[place[label]] for label in question_labels)


def _all_from_heads(sites: Sequence[_Site]) -> list[_FromHead] | None:
    """Every question answered by its own head, or None as soon as one of them needs the provider."""
    from_heads: list[_FromHead] = []
    for question, state, outcome in sites:
        if outcome is None:
            return None
        answer = _from_head(question, state, outcome)
        if answer is None:
            return None
        from_heads.append((question, state, outcome, answer))
    return from_heads


def _provider_label(question: Question, answer: object) -> str | None:
    """The label one provider answer stands for, or None when it is missing, malformed, or names
    a choice the question never offered."""
    if not isinstance(answer, dict):
        return None
    try:
        label = _label(answer)
    except (KeyError, TypeError, ValueError):
        return None
    return label if label in question.labels else None


def _usage(answered: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = answered.get("usage")
    if not isinstance(usage, dict):
        return None, None
    return token_count(usage.get("input_tokens")), token_count(usage.get("output_tokens"))


def _label(answer: dict[str, Any]) -> str:
    if answer["type"] == "choice":
        return str(answer["choice"])
    if answer["type"] == "noul":
        return "true" if float(answer["noul"]) >= _NOUL_TRUE else "false"
    probabilities: dict[str, float] = answer["probabilities"]
    return max(probabilities, key=probabilities.__getitem__)


def _has_key(request: Request) -> bool:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return scheme.lower() == "bearer" and token.strip() != ""


def _request_id(text: str, questions: Sequence[Question]) -> str:
    joined = "\n".join([text, *(question.canonical for question in questions)])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:_REQUEST_ID_CHARS]


def _error(message: str, status: int, reason: str, *, mode: str = _MODE_LOCAL) -> Response:
    return Response(
        content=error_body(message),
        status_code=status,
        media_type="application/json",
        headers={"X-Stuntd": jev_header(mode, reason=reason)},
    )
