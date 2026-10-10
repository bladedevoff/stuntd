from __future__ import annotations

import json
import time
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import asdict

from stuntd.settings import LAYA_ENCODER
from stuntd.train.artifacts import SiteModel
from stuntd.train.metrics import ConfidentError, Interval, Operating

__all__ = ["TIME_FORMAT", "render", "render_json", "thin_curve"]

TIME_FORMAT = "%Y-%m-%d %H:%M"
"""How stuntd prints a recorded moment, in local time."""

_MIN_HOLDOUT = 200
_MIN_COVERED = 100
_CLASS_WIDTH = 18
_SUPPORT_WIDTH = 8
_CURVE_HEADER = "threshold  coverage  agreement"
_CURVE_STEPS = 20


def _class_row(name: object, support: object, agreement: object) -> str:
    # The space past each pad keeps the columns apart when a value fills its width.
    return f"{name:<{_CLASS_WIDTH - 1}} {support:<{_SUPPORT_WIDTH - 1}} {agreement}"


def _interval(interval: Interval | None, rows: int | None) -> str:
    if interval is None or rows is None:
        return ""
    return f" (95% interval {interval.low:.3f}-{interval.high:.3f}, {rows} rows)"


def _headline(model: SiteModel) -> list[str]:
    trained = time.strftime(TIME_FORMAT, time.localtime(model.trained_at))
    base = (
        f"base {model.base_model}" if model.encoder == LAYA_ENCODER else f"encoder {model.encoder}"
    )
    lines = [
        f"site {model.site}  {model.kind} {model.field}  trained {trained}  {base}",
        f"examples: {model.n_train} train, {model.n_holdout} holdout",
        f"holdout agreement {model.agreement:.3f}"
        f"{_interval(model.agreement_interval, model.n_holdout)}  ece {model.ece:.3f}"
        f"  temperature {model.temperature:.2f}",
    ]
    if model.novelty_cutoff is not None and model.familiar_share is not None:
        lines.append(
            f"novelty cut-off {model.novelty_cutoff:.3f},"
            f" lets {model.familiar_share:.0%} of the holdout through"
        )
    if model.threshold is None:
        lines.append(f"threshold: none reaches {model.target_agreement:.2f}")
        return lines
    lines.append(
        f"at threshold {model.threshold:.2f}: coverage {model.coverage:.2f},"
        f" agreement {model.covered_agreement:.3f}"
        f"{_interval(model.covered_interval, model.n_covered)}"
    )
    return lines


def _warnings(model: SiteModel) -> list[str]:
    warnings = []
    if model.n_holdout < _MIN_HOLDOUT:
        warnings.append(
            f"warning: the holdout has {model.n_holdout} rows, fewer than {_MIN_HOLDOUT}, "
            "so the figures above are rough"
        )
    if model.n_covered is not None and model.n_covered < _MIN_COVERED:
        warnings.append(
            f"warning: the threshold rests on {model.n_covered} rows, fewer than {_MIN_COVERED}, "
            "so its agreement may not hold"
        )
    return warnings


def _table(model: SiteModel) -> list[str]:
    rows = [_class_row("class", "support", "agreement")]
    for name, stats in model.per_class.items():
        agreement = "-" if stats.agreement is None else f"{stats.agreement:.3f}"
        rows.append(_class_row(name, stats.support, agreement))
    return rows


def _error_row(model: SiteModel, error: ConfidentError) -> str:
    expected = model.labels[error.expected]
    predicted = model.labels[error.predicted]
    probability = "-" if error.probability is None else f"{error.probability:.2f}"
    held_back = model.threshold is None or error.confidence < model.threshold
    return (
        f"  {error.confidence:.2f}  {probability}  expected {expected}, got {predicted}: "
        f"{error.text}{' (under the threshold)' if held_back else ''}"
    )


def _curve_row(point: Operating) -> str:
    return f"{point.threshold:>9.2f}  {point.coverage:>8.2f}  {point.agreement:>9.3f}"


def thin_curve(curve: Sequence[Operating], threshold: float | None) -> list[Operating]:
    """The curve at every 5% of coverage plus the operating point, in the curve's own order."""
    coverages = [point.coverage for point in curve]
    # The first point at or past each step is the strictest threshold that answers that much.
    kept = {bisect_left(coverages, step / _CURVE_STEPS) for step in range(1, _CURVE_STEPS + 1)}
    return [
        point for index, point in enumerate(curve) if index in kept or point.threshold == threshold
    ]


def render(model: SiteModel, curve: bool = False) -> str:
    """One trained site as the block of text the report command prints."""
    lines = [*_headline(model), *_warnings(model), "", *_table(model)]
    if model.confident_errors:
        lines.append("")
        lines.append("surest mistakes (confidence, probability of the answer given):")
        lines.extend(_error_row(model, error) for error in model.confident_errors)
    if curve:
        lines.append("")
        lines.append(_CURVE_HEADER)
        lines.extend(_curve_row(point) for point in thin_curve(model.curve, model.threshold))
    return "\n".join(lines)


def render_json(models: Sequence[SiteModel]) -> str:
    """Every given model as one JSON list, with the fields they were saved with."""
    return json.dumps([asdict(model) for model in models], indent=2)
