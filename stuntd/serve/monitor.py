from __future__ import annotations

import string
from collections.abc import Sequence
from dataclasses import dataclass

from stuntd.serve.modes import MODE_CHECK, MODE_SHADOW, SiteState
from stuntd.store.db import Decision
from stuntd.train.artifacts import SiteModel

__all__ = [
    "Window",
    "should_check",
    "should_demote",
    "should_promote",
    "window_agreement",
    "window_start",
]

_COMPARABLE_MODES = (MODE_SHADOW, MODE_CHECK)
_SECONDS_PER_HOUR = 3600
_HASH_PREFIX_CHARS = 8
_HASH_SPACE: int = 16**_HASH_PREFIX_CHARS


@dataclass(frozen=True)
class Window:
    """How many confident comparisons a site made recently, and how often they agreed."""

    compared: int
    agreement: float | None


def window_start(state: SiteState) -> float:
    """Comparisons made before a site's last mode switch or retraining judged a different head
    or a different mode, so its window starts at the later of the two."""
    trained_at = 0.0 if state.model is None else state.model.trained_at
    if state.changed_at is None:
        return trained_at
    return max(state.changed_at, trained_at)


def window_agreement(decisions: Sequence[Decision], threshold: float) -> Window:
    """Agreement among shadow and check decisions confident enough to count."""
    compared = [
        decision
        for decision in decisions
        if decision.mode in _COMPARABLE_MODES
        and decision.agree is not None
        and decision.confidence >= threshold
    ]
    if not compared:
        return Window(0, None)
    agreed = sum(1 for decision in compared if decision.agree)
    return Window(len(compared), agreed / len(compared))


def should_demote(window: Window, min_window: int, target: float) -> bool:
    """Whether a live site has fallen below the agreement it was promoted for."""
    return (
        window.compared >= min_window and window.agreement is not None and window.agreement < target
    )


def should_promote(
    model: SiteModel, window: Window, min_window: int, target: float, after_hours: int, now: float
) -> bool:
    """Whether a shadow site is trained, aging and recently accurate enough to go live."""
    if model.threshold is None or model.covered_agreement is None:
        return False
    if model.covered_agreement < target:
        return False
    if window.compared < min_window or window.agreement is None or window.agreement < target:
        return False
    return now - model.trained_at >= after_hours * _SECONDS_PER_HOUR


def should_check(input_hash: str, share: float) -> bool:
    """Whether a live request should still be sent to the provider to check the model."""
    prefix = input_hash[:_HASH_PREFIX_CHARS]
    if len(prefix) < _HASH_PREFIX_CHARS or not all(char in string.hexdigits for char in prefix):
        raise ValueError(f"not an 8-char hex prefix: {input_hash!r}")
    # A hash of the input, not a random draw, so the same request always gets the same
    # decision no matter how many times it is replayed.
    return int(prefix, 16) / _HASH_SPACE < share
