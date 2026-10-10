from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, cast

import httpx
from anyio import to_thread
from starlette.background import BackgroundTask, BackgroundTasks
from starlette.requests import Request
from starlette.responses import Response

from stuntd.jev.answer import answer_label, choice_answer, noul_answer, score_answer
from stuntd.jev.schema import JevError, Question, kind_for, laya_question
from stuntd.proxy.capture import decoded_json, token_count
from stuntd.proxy.headers import DROPPED_WHEN_BUFFERED, forwardable, jev_header, relayed_headers
from stuntd.serve.modes import MODE_CHECK, MODE_LIVE, MODE_SHADOW, SiteState
from stuntd.serve.runtime import ENCODER_MISMATCH, Outcome, Runtime, ZeroShot
from stuntd.store.db import Capture

if TYPE_CHECKING:
    from stuntd.proxy.app import Proxy

__all__ = ["Asked", "QuestionRoutes", "Wire"]

# The two modes these routes serve in: local, with no provider behind them, and proxy, where the
# paid provider answers and its answers become training data.
_MODE_LOCAL = "local"
_MODE_PROXY = "proxy"
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

_log = logging.getLogger(__name__)


class Asked(Protocol):
    """A request that parsed: the text every question reads and the questions, in order."""

    @property
    def text(self) -> str: ...

    @property
    def questions(self) -> Sequence[Question]: ...


_A = TypeVar("_A", bound=Asked)

# One question of a proxied request: what was asked, the site it lands on, and the outcome that
# site's own head came to, if it has one.
_Site = tuple[Question, SiteState, Outcome | None]
_FromHead = tuple[Question, SiteState, Outcome, dict[str, Any]]


@dataclass(frozen=True)
class Wire(Generic[_A]):
    """What differs between two question endpoints: the path, the body formats, and how the
    provider's answers read back as Jev answers."""

    path: str
    parse: Callable[[bytes], _A]
    answers_body: Callable[[str, _A, Sequence[dict[str, Any]], int], bytes]
    error_body: Callable[[str], bytes]
    provider_answers: Callable[[_A, dict[str, Any]], Sequence[dict[str, Any] | None] | None]
    """The provider's answers aligned with the questions, None where one is unusable."""


class QuestionRoutes(Generic[_A]):
    """The head, zero-shot, relay and learn flow of one question endpoint."""

    def __init__(self, proxy: Proxy, wire: Wire[_A], upstream: str) -> None:
        self._proxy = proxy
        self._wire = wire
        self._upstream = upstream

    def __repr__(self) -> str:
        return f"QuestionRoutes(path={self._wire.path!r}, upstream={self._upstream!r})"

    async def answer(self, request: Request) -> Response:
        """Answers one request: through the provider when one is configured, from the trained
        heads and the base Laya checkpoint otherwise."""
        if self._upstream:
            # The provider checks the caller's key itself, so jev_require_key has nothing to add.
            return await self._proxied(request)
        if self._proxy.settings.jev_require_key and not _has_key(request):
            return self._error("missing API key", 401, _NO_KEY)
        try:
            asked = self._wire.parse(await request.body())
        except JevError as exc:
            return self._error(exc.message, exc.status, _BAD_REQUEST)
        decider = self._proxy.zero_shot
        if decider is None:
            return self._error("serving needs the train extra", 503, _NO_RUNTIME)
        return await self._answers(asked, decider)

    async def relay(self, request: Request, path: str) -> Response:
        """Sends a request to the provider as it came, with nothing learned from it."""
        upstream_request = self._upstream_request(request, path, b"")
        return await self._relayed(upstream_request, jev_header(_MODE_PROXY))

    async def _answers(self, asked: _A, decider: ZeroShot) -> Response:
        runtime = self._proxy.runtime
        answers: dict[str, dict[str, Any]] = {}
        pending: list[Question] = []
        # The zero-shot answers below come from the checkpoint a head was distilled from, so they
        # are no teacher: nothing is compared or recorded here, and no request is sampled for a
        # comparison that has nobody to make it.
        for question in asked.questions:
            if runtime is not None:
                state = runtime.state(question.site)
                outcome = await self._outcome(
                    runtime,
                    question,
                    state,
                    asked.text,
                    check=False,
                    gated=self._proxy.settings.local_fallback != "head",
                )
                head = None if outcome is None else _from_head(question, state, outcome)
                if outcome is not None and head is not None:
                    answers[question.name] = head
                    runtime.record_live(state, outcome)
                    continue
            pending.append(question)
        live = len(answers)
        try:
            zero_shot, input_tokens = await _zero_shot(decider, asked.text, pending)
        except RuntimeError:
            _log.exception("zero-shot answers not built")
            return self._error("the model could not answer", 503, _MODEL_ERROR)
        missing = [question.name for question in pending if question.name not in zero_shot]
        if missing:
            _log.error("zero-shot answers missing: %s", missing)
            return self._error("the model could not answer", 503, _MODEL_ERROR)
        for question in pending:
            answers[question.name] = zero_shot[question.name]
        body = self._wire.answers_body(
            self._proxy.settings.jev_model_name,
            asked,
            [answers[question.name] for question in asked.questions],
            input_tokens,
        )
        response = Response(content=body, media_type="application/json")
        response.headers["X-Stuntd"] = jev_header(
            _MODE_LOCAL,
            questions=len(asked.questions),
            live=live,
            zeroshot=len(pending),
            learn_off=runtime is None,
            reason=_mismatch_reason(runtime, asked.questions),
        )
        response.headers[_REQUEST_ID] = _request_id(asked.text, asked.questions)
        return response

    async def _outcome(
        self,
        runtime: Runtime,
        question: Question,
        state: SiteState,
        text: str,
        *,
        check: bool = True,
        gated: bool = True,
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
        return await runtime.live(state, text, check, gated)

    async def _proxied(self, request: Request) -> Response:
        body = await request.body()
        upstream_request = self._upstream_request(request, self._wire.path, body)
        try:
            asked = self._wire.parse(body)
        except JevError as exc:
            # The provider owns the request format, so a body this daemon cannot read still goes
            # to it; the line is what tells a caller why stuntd never learns from their requests.
            _log.warning("the request was not parsed: %s", exc.message)
            header = jev_header(_MODE_PROXY, reason=_NOT_PARSED)
            return await self._relayed(upstream_request, header)
        runtime = self._proxy.runtime
        sites: list[_Site] = []
        # Learning off: no site is resolved and no head runs, so the request is relayed whole.
        if runtime is not None:
            for question in asked.questions:
                state = runtime.state(question.site)
                sites.append(
                    (question, state, await self._outcome(runtime, question, state, asked.text))
                )
            from_heads = _all_from_heads(sites)
            if from_heads is not None:
                return self._local_answer(runtime, asked, from_heads)
        # The counts classify the request before the relay, not what was written afterwards.
        header = jev_header(
            _MODE_PROXY,
            questions=len(asked.questions),
            live=0,
            shadow=sum(1 for _, state, _ in sites if state.mode == MODE_SHADOW),
            check=sum(
                1 for _, _, outcome in sites if outcome is not None and outcome.reason == MODE_CHECK
            ),
            learn_off=runtime is None,
            reason=_mismatch_reason(runtime, asked.questions),
        )
        return await self._relayed(upstream_request, header, sites, asked, runtime)

    def _local_answer(
        self, runtime: Runtime, asked: _A, from_heads: Sequence[_FromHead]
    ) -> Response:
        for _, state, outcome, _ in from_heads:
            runtime.record_live(state, outcome)
        # The provider was never called, so this request spent no tokens of its own.
        body = self._wire.answers_body(
            self._proxy.settings.jev_model_name,
            asked,
            [answer for _, _, _, answer in from_heads],
            0,
        )
        response = Response(content=body, media_type="application/json")
        response.headers["X-Stuntd"] = jev_header(
            _MODE_PROXY, questions=len(from_heads), live=len(from_heads), shadow=0, check=0
        )
        response.headers[_REQUEST_ID] = _request_id(asked.text, asked.questions)
        return response

    async def _relayed(
        self,
        upstream_request: httpx.Request,
        header: str,
        sites: Sequence[_Site] = (),
        asked: _A | None = None,
        runtime: Runtime | None = None,
    ) -> Response:
        try:
            upstream, body, latency_ms = await self._proxy.buffered(upstream_request)
        except httpx.HTTPError:
            return self._error("upstream unreachable", 502, _UPSTREAM_ERROR, mode=_MODE_PROXY)
        response = Response(content=body, status_code=upstream.status_code)
        response.raw_headers = relayed_headers(upstream.headers.raw, header, DROPPED_WHEN_BUFFERED)
        if upstream.status_code != 200 or not sites or asked is None or runtime is None:
            return response
        answered = decoded_json(upstream, body)
        if answered is None:
            _log.error("the provider answered with no JSON object")
            return response
        tasks = self._learn(runtime, sites, asked, answered, latency_ms)
        if tasks:
            response.background = BackgroundTasks(tasks)
        return response

    def _learn(
        self,
        runtime: Runtime,
        sites: Sequence[_Site],
        asked: _A,
        answered: dict[str, Any],
        latency_ms: int,
    ) -> list[BackgroundTask]:
        answers = self._wire.provider_answers(asked, answered)
        if answers is None:
            _log.error("the provider answered without answers")
            return []
        model = str(answered.get("model", ""))
        input_tokens, output_tokens = _usage(answered)
        captures: list[Capture] = []
        tasks: list[BackgroundTask] = []
        for (question, state, outcome), answer in zip(sites, answers, strict=True):
            label = _provider_label(question, answer)
            if label is None:
                _log.warning("no usable provider answer for question %r", question.name)
                continue
            if self._proxy.settings.learn:
                captures.append(
                    Capture(
                        question.site,
                        question.canonical,
                        kind_for(question),
                        asked.text,
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
                tasks.append(BackgroundTask(runtime.shadow, state, asked.text, label))
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
                if self._proxy.retrainer is not None:
                    self._proxy.retrainer.record(capture.site, capture.input_text)
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
            self._upstream.rstrip("/") + path,
            headers=headers,
            content=body,
        )

    def _error(
        self, message: str, status: int, reason: str, *, mode: str = _MODE_LOCAL
    ) -> Response:
        return Response(
            content=self._wire.error_body(message),
            status_code=status,
            media_type="application/json",
            headers={"X-Stuntd": jev_header(mode, reason=reason)},
        )


async def _zero_shot(
    decider: ZeroShot, text: str, pending: Sequence[Question]
) -> tuple[dict[str, Any], int]:
    if not pending:
        return {}, 0
    questions = {question.name: laya_question(question.canonical) for question in pending}
    # laya answers in its own untyped shape, which the decider passes through as object.
    reply: Any = await to_thread.run_sync(decider.answer, text, questions)
    return reply["answers"], int(reply["usage"]["input_tokens"])


def _mismatch_reason(runtime: Runtime | None, questions: Sequence[Question]) -> str | None:
    """ENCODER_MISMATCH when a question's site has a head trained on another encoder, else None."""
    if runtime is not None and any(
        runtime.state_reason(question.site) == ENCODER_MISMATCH for question in questions
    ):
        return ENCODER_MISMATCH
    return None


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
    # build_question admits a score question only with a list of levels.
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


def _provider_label(question: Question, answer: dict[str, Any] | None) -> str | None:
    """The label one provider answer stands for, or None when it is missing, malformed, or names
    a choice the question never offered."""
    if answer is None:
        return None
    try:
        label = answer_label(answer)
    except (KeyError, TypeError, ValueError):
        return None
    return label if label in question.labels else None


def _usage(answered: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = answered.get("usage")
    if not isinstance(usage, dict):
        return None, None
    return token_count(usage.get("input_tokens")), token_count(usage.get("output_tokens"))


def _has_key(request: Request) -> bool:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return scheme.lower() == "bearer" and token.strip() != ""


def _request_id(text: str, questions: Sequence[Question]) -> str:
    joined = "\n".join([text, *(question.canonical for question in questions)])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:_REQUEST_ID_CHARS]
