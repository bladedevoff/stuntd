from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "ERROR_TEXT_CHARS",
    "ClassStats",
    "ConfidentError",
    "Operating",
    "Prediction",
    "confidence",
    "confident_errors",
    "ece",
    "fit_temperature",
    "operating_curve",
    "operating_point",
    "per_class",
    "predict",
    "softmax",
]

ERROR_TEXT_CHARS = 80
"""How much of a confident mistake's input is kept for the report to show."""

_CONFIDENCE_DIGITS = 9
_MIN_SCALE = 0.05
_MAX_SCALE = 20.0
_SEARCH_STEPS = 60


@dataclass(frozen=True)
class Prediction:
    """One holdout row: the answer it carried, the answer chosen for it, and how sure that was."""

    label: int
    predicted: int
    confidence: float


@dataclass(frozen=True)
class Operating:
    """One threshold and what it buys: how much is answered, and how much of that is right."""

    threshold: float
    coverage: float
    agreement: float


@dataclass(frozen=True)
class ClassStats:
    """How one label fared, with no agreement to report when nothing carried it."""

    support: int
    agreement: float | None


@dataclass(frozen=True)
class ConfidentError:
    """A mistake worth reading: the input, the answer wanted, and the sure answer given."""

    text: str
    expected: int
    predicted: int
    confidence: float


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Turns logits into probabilities, flattened or sharpened by the temperature."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature!r}")
    scaled = [logit / temperature for logit in logits]
    highest = max(scaled)
    weights = [math.exp(value - highest) for value in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


def confidence(probs: Sequence[float]) -> float:
    """How far a distribution sits from uniform, normalised by its length: 0.0 flat, 1.0 sure."""
    if len(probs) < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
    # Rounded, because a distribution the float arithmetic cannot tell from one-hot must not
    # land a hair under the threshold fitted on a holdout that was just as separable.
    return round(min(1.0, max(0.0, 1.0 - entropy / math.log(len(probs)))), _CONFIDENCE_DIGITS)


def _nll(logits: Sequence[Sequence[float]], labels: Sequence[int], scale: float) -> float:
    total = 0.0
    for row, label in zip(logits, labels, strict=True):
        scaled = [logit * scale for logit in row]
        highest = max(scaled)
        shifted = sum(math.exp(value - highest) for value in scaled)
        total += highest + math.log(shifted) - scaled[label]
    return total / len(logits)


def fit_temperature(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Finds the temperature whose probabilities fit the true labels best."""
    # The loss is convex in the inverse temperature, so ternary search runs over the scale.
    low, high = _MIN_SCALE, _MAX_SCALE
    for _ in range(_SEARCH_STEPS):
        third = (high - low) / 3.0
        left, right = low + third, high - third
        if _nll(logits, labels, left) <= _nll(logits, labels, right):
            high = right
        else:
            low = left
    return 1.0 / ((low + high) / 2.0)


def predict(
    logits: Sequence[Sequence[float]], labels: Sequence[int], temperature: float
) -> list[Prediction]:
    """Scores every row at the given temperature, keeping the true label beside the choice."""
    predictions = []
    for row, label in zip(logits, labels, strict=True):
        probs = softmax(row, temperature)
        chosen = max(range(len(probs)), key=lambda index: probs[index])
        predictions.append(Prediction(label, chosen, confidence(probs)))
    return predictions


def ece(predictions: Sequence[Prediction], bins: int = 15) -> float:
    """Expected calibration error: the gap between stated confidence and how often it was right."""
    if not predictions:
        return 0.0
    counts = [0] * bins
    stated = [0.0] * bins
    correct = [0] * bins
    for prediction in predictions:
        if prediction.confidence <= 0.0:
            continue
        place = min(bins - 1, math.ceil(prediction.confidence * bins) - 1)
        counts[place] += 1
        stated[place] += prediction.confidence
        correct[place] += prediction.label == prediction.predicted
    return sum(
        count / len(predictions) * abs(stated[place] / count - correct[place] / count)
        for place, count in enumerate(counts)
        if count
    )


def operating_curve(predictions: Sequence[Prediction]) -> list[Operating]:
    """Coverage and agreement at every threshold the confidences make distinct, widest last."""
    ordered = sorted(predictions, key=lambda prediction: prediction.confidence, reverse=True)
    points: list[Operating] = []
    correct = 0
    for answered, prediction in enumerate(ordered, start=1):
        correct += prediction.label == prediction.predicted
        if answered < len(ordered) and ordered[answered].confidence == prediction.confidence:
            continue
        points.append(Operating(prediction.confidence, answered / len(ordered), correct / answered))
    return points


def operating_point(predictions: Sequence[Prediction], target: float) -> Operating | None:
    """The widest point of the curve that still reaches the target agreement, if one does."""
    reaching = [point for point in operating_curve(predictions) if point.agreement >= target]
    if not reaching:
        return None
    return max(reaching, key=lambda point: point.coverage)


def per_class(predictions: Sequence[Prediction], labels: Sequence[str]) -> dict[str, ClassStats]:
    """Counts support and agreement for every label, against the answers the rows carried."""
    stats: dict[str, ClassStats] = {}
    for index, name in enumerate(labels):
        carried = [prediction for prediction in predictions if prediction.label == index]
        if not carried:
            stats[name] = ClassStats(0, None)
            continue
        correct = sum(prediction.label == prediction.predicted for prediction in carried)
        stats[name] = ClassStats(len(carried), correct / len(carried))
    return stats


def confident_errors(
    predictions: Sequence[Prediction], texts: Sequence[str], limit: int = 5
) -> list[ConfidentError]:
    """The mistakes the model was surest about, the surest first."""
    # The captured text is a whole conversation, so it is flattened and cut to one short line
    # rather than stored whole in every model's metadata.
    mistakes = [
        ConfidentError(
            " ".join(text.split())[:ERROR_TEXT_CHARS],
            prediction.label,
            prediction.predicted,
            prediction.confidence,
        )
        for prediction, text in zip(predictions, texts, strict=True)
        if prediction.label != prediction.predicted
    ]
    mistakes.sort(key=lambda mistake: mistake.confidence, reverse=True)
    return mistakes[:limit]
