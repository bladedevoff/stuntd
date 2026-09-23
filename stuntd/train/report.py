from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict

from stuntd.train.artifacts import SiteModel
from stuntd.train.metrics import ConfidentError, Operating

__all__ = ["TIME_FORMAT", "render", "render_json"]

TIME_FORMAT = "%Y-%m-%d %H:%M"
"""How stuntd prints a recorded moment, in local time."""

_CLASS_WIDTH = 18
_SUPPORT_WIDTH = 8
_CURVE_HEADER = "threshold  coverage  agreement"


def _class_row(name: object, support: object, agreement: object) -> str:
    # The space past each pad keeps the columns apart when a value fills its width.
    return f"{name:<{_CLASS_WIDTH - 1}} {support:<{_SUPPORT_WIDTH - 1}} {agreement}"


def _headline(model: SiteModel) -> list[str]:
    trained = time.strftime(TIME_FORMAT, time.localtime(model.trained_at))
    lines = [
        f"site {model.site}  {model.kind} {model.field}"
        f"  trained {trained}  base {model.base_model}",
        f"examples: {model.n_train} train, {model.n_holdout} holdout",
        f"holdout agreement {model.agreement:.3f}  ece {model.ece:.3f}"
        f"  temperature {model.temperature:.2f}",
    ]
    if model.threshold is None:
        lines.append(f"threshold: none reaches {model.target_agreement:.2f}")
        return lines
    lines.append(
        f"at threshold {model.threshold:.2f}: coverage {model.coverage:.2f},"
        f" agreement {model.covered_agreement:.3f}"
    )
    return lines


def _table(model: SiteModel) -> list[str]:
    rows = [_class_row("class", "support", "agreement")]
    for name, stats in model.per_class.items():
        agreement = "-" if stats.agreement is None else f"{stats.agreement:.3f}"
        rows.append(_class_row(name, stats.support, agreement))
    return rows


def _error_row(model: SiteModel, error: ConfidentError) -> str:
    expected = model.labels[error.expected]
    predicted = model.labels[error.predicted]
    return f"  {error.confidence:.2f}  expected {expected}, got {predicted}: {error.text}"


def _curve_row(point: Operating) -> str:
    return f"{point.threshold:>9.2f}  {point.coverage:>8.2f}  {point.agreement:>9.3f}"


def render(model: SiteModel, curve: bool = False) -> str:
    """One trained site as the block of text the report command prints."""
    lines = [*_headline(model), "", *_table(model)]
    if model.confident_errors:
        lines.append("")
        lines.append("confident errors:")
        lines.extend(_error_row(model, error) for error in model.confident_errors)
    if curve:
        lines.append("")
        lines.append(_CURVE_HEADER)
        lines.extend(_curve_row(point) for point in model.curve)
    return "\n".join(lines)


def render_json(models: Sequence[SiteModel]) -> str:
    """Every given model as one JSON list, with the fields they were saved with."""
    return json.dumps([asdict(model) for model in models], indent=2)
