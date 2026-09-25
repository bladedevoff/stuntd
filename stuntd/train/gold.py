from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from stuntd.train.artifacts import SiteModel

__all__ = ["GoldRow", "GoldScore", "render_gold", "render_gold_json", "score_gold"]


@dataclass(frozen=True)
class GoldRow:
    """One verified row: the gold answer, the head's answer and confidence, the teacher's if stored."""

    gold: str
    head: str
    confidence: float
    teacher: str | None


@dataclass(frozen=True)
class GoldScore:
    """How the head, the teacher and what stuntd would serve did against verified answers."""

    read: int
    used: int
    skipped: int
    head_accuracy: float | None
    unseen_rows: int
    unseen_head_accuracy: float | None
    served_accuracy: float | None
    local_share: float | None
    unserved: int
    teacher_rows: int
    teacher_accuracy: float | None
    both_right: int
    both_wrong: int
    head_only: int
    teacher_only: int


def _rate(hits: int, total: int) -> float | None:
    return hits / total if total else None


def score_gold(rows: Sequence[GoldRow], threshold: float | None, skipped: int) -> GoldScore:
    """Scores the rows a site's head answered against their gold answers."""
    local = served = served_right = 0
    for row in rows:
        # A site without a threshold never lets its head answer, so the teacher serves every row.
        if threshold is not None and row.confidence >= threshold:
            local += 1
            answer: str | None = row.head
        else:
            answer = row.teacher
        if answer is not None:
            served += 1
            served_right += answer == row.gold
    # A row the store holds may be one the head trained on; the rest are out of sample.
    unseen = [row for row in rows if row.teacher is None]
    pairs = [
        (row.head == row.gold, row.teacher == row.gold) for row in rows if row.teacher is not None
    ]
    return GoldScore(
        read=len(rows) + skipped,
        used=len(rows),
        skipped=skipped,
        head_accuracy=_rate(sum(row.head == row.gold for row in rows), len(rows)),
        unseen_rows=len(unseen),
        unseen_head_accuracy=_rate(sum(row.head == row.gold for row in unseen), len(unseen)),
        served_accuracy=_rate(served_right, served),
        local_share=_rate(local, len(rows)),
        unserved=len(rows) - served,
        teacher_rows=len(pairs),
        teacher_accuracy=_rate(sum(teacher for _, teacher in pairs), len(pairs)),
        both_right=pairs.count((True, True)),
        both_wrong=pairs.count((False, False)),
        head_only=pairs.count((True, False)),
        teacher_only=pairs.count((False, True)),
    )


def _shown(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def render_gold(model: SiteModel, score: GoldScore) -> str:
    """One site's gold score as the block of text the report command prints."""
    if model.threshold is None:
        at, unserved = "no threshold", f"{score.unserved} with no teacher answer"
    else:
        at = f"threshold {model.threshold:.2f}"
        unserved = f"{score.unserved} below it with no teacher answer"
    local = "-" if score.local_share is None else f"{score.local_share:.0%}"
    return "\n".join(
        [
            f"site {model.site}  gold rows {score.read}: {score.used} used, {score.skipped} skipped",
            f"head accuracy {_shown(score.head_accuracy)}",
            f"head accuracy {_shown(score.unseen_head_accuracy)} on {score.unseen_rows} rows"
            " the store has not seen",
            f"served accuracy {_shown(score.served_accuracy)} at {at}: {local} answered locally,"
            f" {unserved}",
            f"teacher accuracy {_shown(score.teacher_accuracy)} on {score.teacher_rows} rows",
            f"head and teacher: both right {score.both_right}, both wrong {score.both_wrong},"
            f" head only {score.head_only}, teacher only {score.teacher_only}",
        ]
    )


def render_gold_json(model: SiteModel, score: GoldScore) -> str:
    """One site's gold score as a JSON object, with the threshold it was served at."""
    return json.dumps({"site": model.site, "threshold": model.threshold, **asdict(score)}, indent=2)
