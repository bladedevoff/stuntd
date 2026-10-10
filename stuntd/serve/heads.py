from __future__ import annotations

from dataclasses import dataclass

__all__ = ["HEAD_CACHE_SIZE", "Verdict"]

HEAD_CACHE_SIZE = 4
"""Heads kept in memory at once, dropping the one left unused longest."""


@dataclass(frozen=True)
class Verdict:
    """One answer from a site's head: the label it chose, every label's probability, how sure it
    is, how long it took, and how far the request is from the rows the head trained on, when the
    head kept them."""

    label: int
    confidence: float
    latency_ms: int
    probabilities: tuple[float, ...]
    novelty: float | None = None
