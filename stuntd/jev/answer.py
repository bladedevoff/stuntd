from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

__all__ = ["choice_answer", "error_body", "noul_answer", "response_body", "score_answer"]


def choice_answer(
    labels: Sequence[str], probabilities: Sequence[float], confidence: float
) -> dict[str, Any]:
    """Builds a Jev choice answer from per-label probabilities."""
    best_index = max(range(len(labels)), key=lambda i: probabilities[i])
    rounded = [round(float(p), 4) for p in probabilities]
    return {
        "type": "choice",
        "choice": labels[best_index],
        "confidence": round(float(confidence), 4),
        "probabilities": dict(zip(labels, rounded, strict=True)),
    }


def noul_answer(probabilities: Sequence[float]) -> dict[str, Any]:
    """Builds a Jev noul answer from (p_false, p_true)."""
    return {"type": "noul", "noul": round(float(probabilities[1]), 4)}


def score_answer(
    levels: Sequence[Any], probabilities: Sequence[float], confidence: float
) -> dict[str, Any]:
    """Builds a Jev score answer as the probability-weighted expectation over levels."""
    rounded = [round(float(p), 4) for p in probabilities]
    score = round(sum(i * p for i, p in enumerate(probabilities)), 4)
    keys = [str(i) for i in range(len(levels))]
    return {
        "type": "score",
        "score": score,
        "confidence": round(float(confidence), 4),
        "legend": dict(zip(keys, levels, strict=True)),
        "probabilities": dict(zip(keys, rounded, strict=True)),
    }


def response_body(model: str, answers: dict[str, dict[str, Any]], input_tokens: int) -> bytes:
    """Builds the Jev /v1/systemone success response body."""
    body = {
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def error_body(message: str) -> bytes:
    """Builds the Jev error response body."""
    body = {"error": {"message": message}}
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
