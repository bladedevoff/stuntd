from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

from stuntd.decisions.schema import single_typed_field
from stuntd.jev.schema import typed_field
from stuntd.store.db import Example

__all__ = ["MAX_NUMBER_LABELS", "Item", "NotTrainable", "SiteDataset", "build_dataset"]

MAX_NUMBER_LABELS = 10
"""Distinct numeric answers a site may have and still be learned as a choice between them."""

_BOOLEAN_LABELS = ("false", "true")


class NotTrainable(Exception):
    """A site that cannot be trained from the examples at hand; the message says why."""


@dataclass(frozen=True)
class Item:
    """One example ready for training, its answer already turned into a label index."""

    text: str
    label: int
    created_at: float


@dataclass(frozen=True)
class SiteDataset:
    """Everything the trainer needs for one site: its labels and its two time-ordered parts."""

    site: str
    kind: str
    field: str
    labels: tuple[str, ...]
    train: tuple[Item, ...]
    holdout: tuple[Item, ...]
    deduplicated: int
    dropped: int


def _finite_number(answer: str) -> float | None:
    try:
        value = float(answer)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _number_labels(examples: Sequence[Example]) -> tuple[tuple[str, ...], dict[str, int]]:
    values: dict[str, float] = {}
    for example in examples:
        value = _finite_number(example.answer)
        if value is not None:
            values.setdefault(example.answer, value)
    spellings: dict[float, str] = {}
    for answer, value in values.items():
        spellings.setdefault(value, answer)
    if len(spellings) > MAX_NUMBER_LABELS:
        raise NotTrainable(
            f"{len(spellings)} distinct numbers,"
            f" at most {MAX_NUMBER_LABELS} can be learned as choices"
        )
    # Sorted numerically so 2 comes before 10, labelled with the first spelling of each value.
    ordered = sorted(spellings)
    place = {value: position for position, value in enumerate(ordered)}
    labels = tuple(spellings[value] for value in ordered)
    return labels, {answer: place[value] for answer, value in values.items()}


def _labels(
    kind: str, options: tuple[str, ...], examples: Sequence[Example]
) -> tuple[tuple[str, ...], dict[str, int]]:
    if not options:
        if kind == "number":
            return _number_labels(examples)
        options = _BOOLEAN_LABELS if kind == "boolean" else options
    # A Jev site names its labels, the levels of a score site included, so a level no example
    # carries keeps its place instead of being read off the answers.
    return options, {label: position for position, label in enumerate(options)}


def _deduplicate(examples: Sequence[Example]) -> list[Example]:
    latest: dict[str, Example] = {}
    for example in examples:
        kept = latest.get(example.input_text)
        if kept is None or example.created_at >= kept.created_at:
            latest[example.input_text] = example
    return list(latest.values())


def _split(items: Sequence[Item], holdout: float) -> tuple[tuple[Item, ...], tuple[Item, ...]]:
    ordered = sorted(items, key=lambda item: item.created_at)
    size = max(1, math.ceil(len(ordered) * holdout))
    return tuple(ordered[:-size]), tuple(ordered[-size:])


def build_dataset(
    site: str,
    kind: str,
    schema_canonical: str,
    examples: Sequence[Example],
    min_examples: int,
    holdout: float,
) -> SiteDataset:
    """Turns the captures of one site into a deduplicated, labelled, time-split training set."""
    try:
        schema = json.loads(schema_canonical)
    except json.JSONDecodeError as exc:
        raise NotTrainable("schema is not a single typed field") from exc
    found = single_typed_field(schema) or typed_field(site, schema_canonical)
    if found is None:
        raise NotTrainable("schema is not a single typed field")
    field, _, options = found
    kept = _deduplicate(examples)
    labels, index = _labels(kind, options, kept)
    items = [
        Item(example.input_text, index[example.answer], example.created_at)
        for example in kept
        if example.answer in index
    ]
    if len(items) < min_examples:
        raise NotTrainable(f"{len(items)} examples after deduplication, {min_examples} needed")
    train, held_out = _split(items, holdout)
    if len({item.label for item in train}) < 2:
        raise NotTrainable("training examples all carry the same answer")
    if len(held_out) == 1:
        raise NotTrainable("held-out slice has a single example; raise holdout or add examples")
    # A holdout of one answer makes every agreement number meaningless: a model that always
    # says that answer scores a perfect one.
    if len({item.label for item in held_out}) < 2:
        raise NotTrainable("held-out examples all carry the same answer")
    return SiteDataset(
        site=site,
        kind=kind,
        field=field,
        labels=labels,
        train=train,
        holdout=held_out,
        deduplicated=len(examples) - len(kept),
        dropped=len(kept) - len(items),
    )
