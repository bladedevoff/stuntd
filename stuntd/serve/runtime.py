from __future__ import annotations

import hashlib
import logging
import sqlite3
import stat
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from anyio import to_thread

from stuntd.serve.modes import (
    MODE_CHECK,
    MODE_COLLECT,
    MODE_FILE,
    MODE_LIVE,
    MODE_SHADOW,
    SiteState,
    site_state,
    write_mode,
)
from stuntd.serve.monitor import (
    Window,
    should_check,
    should_demote,
    should_promote,
    window_agreement,
    window_start,
)
from stuntd.settings import Settings, models_path
from stuntd.store.db import Decision, Store
from stuntd.train.artifacts import HEAD_FILE, META_FILE, SiteModel, site_dir

if TYPE_CHECKING:
    from stuntd.serve.decider import Verdict

__all__ = ["DeciderLike", "Outcome", "Runtime", "input_hash"]

_log = logging.getLogger(__name__)

_LOW_CONFIDENCE = "low-confidence"
_NO_RUNTIME = "no-runtime"
_MODEL_ERROR = "model-error"
_STATE_ERROR = "state-error"
_RETRAINED = "retrained"

_DecisionMode = Literal["shadow", "check", "live"]

_Stamp = tuple[int, int]

_Stamps = tuple[_Stamp, _Stamp | None]


class DeciderLike(Protocol):
    """What serving needs of a decider: an answer from a site's head, a batch of zero-shot answers,
    and a pass to warm a head or the base checkpoint."""

    def decide(self, model: SiteModel, head_path: Path, text: str) -> Verdict: ...

    def answer(
        self, state: object, questions: dict[str, dict[str, object]]
    ) -> dict[str, object]: ...

    def warm(self, model: SiteModel, head_path: Path) -> None: ...

    def warm_base(self) -> None: ...


@dataclass(frozen=True)
class Outcome:
    """What one live request came to: the mode it is served in, why, and what the model said."""

    mode: str
    reason: str | None
    answer: str | None
    confidence: float | None
    latency_ms: int | None
    probabilities: tuple[float, ...] | None = None


class Runtime:
    """Everything the proxy needs to let a trained site answer: modes, cache, the decider and
    bookkeeping."""

    def __init__(
        self,
        settings: Settings,
        store: Store,
        decider: DeciderLike | None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._settings = settings
        self._store = store
        self._decider = decider
        self._now = now
        self._models = models_path(settings)
        self._states: dict[str, tuple[_Stamps, SiteState]] = {}
        self._unreadable: set[str] = set()
        self._cache: OrderedDict[str, Verdict] = OrderedDict()

    def __repr__(self) -> str:
        return (
            f"Runtime(models={str(self._models)!r}, decider={self._decider!r},"
            f" cached={len(self._cache)})"
        )

    def state(self, site: str) -> SiteState:
        """One site's mode and model, re-read only once the files behind them change."""
        folder = site_dir(self._models, site)
        stamps = _stamps(folder)
        if stamps is None:
            return SiteState(site, MODE_COLLECT, None, None)
        cached = self._states.get(site)
        if cached is not None and cached[0] == stamps:
            return cached[1]
        # Training and the mode switches rewrite these two files, so their timestamps are what
        # tells a state still worth reusing from one the daemon must read again.
        try:
            state = site_state(self._models, site)
        except (ValueError, OSError):
            # A half-written file would otherwise turn every request for this site into a bare
            # 500, and the provider can still answer them all.
            if site not in self._unreadable:
                self._unreadable.add(site)
                _log.exception("%s state unreadable", site)
            return SiteState(site, MODE_COLLECT, None, None)
        self._unreadable.discard(site)
        self._states[site] = (stamps, state)
        return state

    def state_reason(self, site: str) -> str | None:
        """Why a site collects although it has a model, for the header to carry, or None."""
        return _STATE_ERROR if site in self._unreadable else None

    async def shadow(self, state: SiteState, text: str, big_answer: str) -> None:
        """Runs the model beside the provider and records whether the two agreed."""
        if self._decider is None or state.model is None:
            return
        try:
            verdict = await self._decide(self._decider, state.model, text)
            answer = state.model.labels[verdict.label]
            self._record(
                state,
                "shadow",
                answer,
                verdict.confidence,
                answer == big_answer,
                verdict.latency_ms,
            )
            self.promote_if_allowed(state)
        except (RuntimeError, sqlite3.Error):
            # The caller already has the provider's answer, so a shadow failure has nothing left
            # to fail: raising here would only break a connection that is already served.
            _log.exception("shadow decision failed")

    async def live(self, state: SiteState, text: str, check: bool = True) -> Outcome:
        """Answers one request from the site's own model, or says why the provider must; with
        check off no request is sampled for comparison, because nothing can answer it instead."""
        model = state.model
        if model is None or model.threshold is None:
            return Outcome(MODE_COLLECT, _LOW_CONFIDENCE, None, None, None)
        if self._decider is None:
            return Outcome(MODE_COLLECT, _NO_RUNTIME, None, None, None)
        request_hash = input_hash(state.site, text)
        key = f"{state.site}:{model.trained_at}:{request_hash}"
        cached = self._cached(key)
        if cached is not None:
            # No pass ran, so the cached verdict's own latency would report time this request
            # never spent.
            return Outcome(
                MODE_LIVE,
                None,
                model.labels[cached.label],
                cached.confidence,
                0,
                cached.probabilities,
            )
        try:
            verdict = await self._decide(self._decider, model, text)
        except RuntimeError:
            _log.exception("live decision failed")
            return Outcome(MODE_COLLECT, _MODEL_ERROR, None, None, None)
        if self._retrained(state.site):
            # Training publishes a new head under the same folder, so an answer decided across
            # that moment came from weights the site no longer serves with.
            return Outcome(MODE_COLLECT, _RETRAINED, None, None, None)
        answer = model.labels[verdict.label]
        if verdict.confidence < model.threshold:
            return Outcome(
                MODE_COLLECT,
                _LOW_CONFIDENCE,
                answer,
                verdict.confidence,
                verdict.latency_ms,
                verdict.probabilities,
            )
        if check and should_check(request_hash, self._settings.check_share):
            return Outcome(
                MODE_COLLECT,
                MODE_CHECK,
                answer,
                verdict.confidence,
                verdict.latency_ms,
                verdict.probabilities,
            )
        self._remember(key, verdict)
        return Outcome(
            MODE_LIVE, None, answer, verdict.confidence, verdict.latency_ms, verdict.probabilities
        )

    def compare(self, state: SiteState, outcome: Outcome, big_answer: str) -> None:
        """Records how a sampled check turned out and demotes the site when it keeps losing."""
        if (
            outcome.reason != MODE_CHECK
            or outcome.answer is None
            or outcome.confidence is None
            or outcome.latency_ms is None
        ):
            return
        try:
            self._record(
                state,
                "check",
                outcome.answer,
                outcome.confidence,
                outcome.answer == big_answer,
                outcome.latency_ms,
            )
        except sqlite3.Error:
            # The caller already has the provider's answer, so a store that cannot take the
            # comparison must not cost them that answer.
            _log.exception("check decision not recorded")
            return
        self.demote_if_needed(state)

    def record_live(self, state: SiteState, outcome: Outcome) -> None:
        """Records one answer the site gave on its own, with nothing to compare it against."""
        if (
            outcome.mode != MODE_LIVE
            or outcome.answer is None
            or outcome.confidence is None
            or outcome.latency_ms is None
        ):
            return
        try:
            self._record(
                state, "live", outcome.answer, outcome.confidence, None, outcome.latency_ms
            )
        except sqlite3.Error:
            # The site has already answered the caller, so its bookkeeping has nothing left to
            # fail: raising here would only take back an answer that was served.
            _log.exception("live decision not recorded")

    def demote_if_needed(self, state: SiteState) -> bool:
        """Puts a live site back into shadow once its recent agreement falls below the target."""
        try:
            window = self._window(state)
            if window is None or not should_demote(
                window, self._settings.min_window, self._settings.target_agreement
            ):
                return False
            self._switch(state.site, MODE_SHADOW)
        except (OSError, sqlite3.Error):
            # A caller holding the provider's answer must not lose it because the decisions or
            # the mode file could not be read or rewritten.
            _log.exception("demotion check failed")
            return False
        _log.warning(
            "%s demoted to shadow: agreement %s over %d decisions",
            state.site,
            window.agreement,
            window.compared,
        )
        return True

    def promote_if_allowed(self, state: SiteState) -> bool:
        """Sends a shadow site live once it has held the target agreement long enough."""
        if not self._settings.auto_promote or state.model is None:
            return False
        try:
            window = self._window(state)
            if window is None or not should_promote(
                state.model,
                window,
                self._settings.min_window,
                self._settings.target_agreement,
                self._settings.auto_promote_after_hours,
                self._now(),
            ):
                return False
            self._switch(state.site, MODE_LIVE)
        except (OSError, sqlite3.Error):
            _log.exception("promotion check failed")
            return False
        _log.info(
            "%s promoted to live: agreement %s over %d decisions",
            state.site,
            window.agreement,
            window.compared,
        )
        return True

    async def _decide(self, decider: DeciderLike, model: SiteModel, text: str) -> Verdict:
        head_path = site_dir(self._models, model.site) / HEAD_FILE
        # Captures are stored redacted, so the head learned redacted text; the cache and the
        # check sample still key on the caller's own text.
        return await to_thread.run_sync(decider.decide, model, head_path, self._store.redact(text))

    def _retrained(self, site: str) -> bool:
        cached = self._states.get(site)
        return cached is not None and _stamps(site_dir(self._models, site)) != cached[0]

    def _window(self, state: SiteState) -> Window | None:
        if state.model is None or state.model.threshold is None:
            return None
        rows = self._store.comparisons(state.site, self._settings.window, window_start(state))
        return window_agreement(rows, state.model.threshold)

    def _switch(self, site: str, mode: str) -> None:
        write_mode(site_dir(self._models, site), mode, self._now())
        self._states.pop(site, None)

    def _record(
        self,
        state: SiteState,
        mode: _DecisionMode,
        answer: str,
        confidence: float,
        agree: bool | None,
        latency_ms: int,
    ) -> None:
        self._store.record_decision(
            Decision(state.site, mode, answer, confidence, agree, latency_ms, self._now())
        )

    def _cached(self, key: str) -> Verdict | None:
        verdict = self._cache.get(key)
        if verdict is None:
            return None
        self._cache.move_to_end(key)
        return verdict

    def _remember(self, key: str, verdict: Verdict) -> None:
        self._cache[key] = verdict
        if len(self._cache) > self._settings.cache_size:
            self._cache.popitem(last=False)


def input_hash(site: str, text: str) -> str:
    """Names one request of one site: the cache key, the check sample and the answer's id."""
    return hashlib.sha256(f"{site}\n{text}".encode()).hexdigest()


def _stamp(path: Path) -> _Stamp | None:
    try:
        info = path.stat()
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    # Size as well as the timestamp, because a clock too coarse to separate two writes of the
    # same file would otherwise hide the second one.
    return info.st_mtime_ns, info.st_size


def _stamps(folder: Path) -> _Stamps | None:
    meta = _stamp(folder / META_FILE)
    if meta is None:
        return None
    return meta, _stamp(folder / MODE_FILE)
