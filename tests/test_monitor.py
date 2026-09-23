from __future__ import annotations

from dataclasses import replace

import pytest

from stuntd.serve.monitor import (
    Window,
    should_check,
    should_demote,
    should_promote,
    window_agreement,
)
from stuntd.store.db import Decision
from stuntd.train.artifacts import SiteModel


def d(mode="shadow", confidence=0.9, agree=True):
    return Decision("s", mode, "true", confidence, agree, 10, 0.0)


def test_window_counts_only_confident_comparisons():
    rows = [
        d(),
        d(agree=False),
        d(confidence=0.3, agree=False),
        d(mode="live", agree=None),
        d(mode="live", agree=False),
        d(mode="check"),
    ]
    window = window_agreement(rows, threshold=0.6)
    assert window == Window(compared=3, agreement=pytest.approx(2 / 3))


def test_empty_window_has_no_agreement():
    assert window_agreement([], 0.5) == Window(0, None)


def test_window_agreement_includes_confidence_at_threshold():
    window = window_agreement([d(confidence=0.6)], threshold=0.6)
    assert window == Window(1, 1.0)


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        (Window(19, 0.5), False),
        (Window(20, 0.5), True),
        (Window(20, 0.99), False),
        (Window(0, None), False),
    ],
    ids=["too-few", "low", "at-target", "empty"],
)
def test_should_demote(window, expected):
    assert should_demote(window, min_window=20, target=0.99) is expected


def model(covered=0.995, trained_at=0.0):
    return SiteModel(
        site="s",
        kind="boolean",
        field="f",
        labels=["false", "true"],
        base_model="b",
        temperature=1.0,
        threshold=0.6,
        target_agreement=0.99,
        trained_at=trained_at,
        n_train=1,
        n_holdout=1,
        agreement=1.0,
        coverage=1.0,
        covered_agreement=covered,
        ece=0.0,
        per_class={},
        confident_errors=[],
        curve=[],
    )


@pytest.mark.parametrize(
    ("covered", "window", "now", "expected"),
    [
        (0.995, Window(20, 0.995), 90_000.0, True),
        (0.98, Window(20, 0.995), 90_000.0, False),
        (0.995, Window(19, 1.0), 90_000.0, False),
        (0.995, Window(20, 0.98), 90_000.0, False),
        (0.995, Window(20, 0.995), 3_600.0, False),
        (None, Window(20, 0.995), 90_000.0, False),
    ],
    ids=["all-good", "holdout-low", "window-thin", "window-low", "too-soon", "no-threshold"],
)
def test_should_promote(covered, window, now, expected):
    assert (
        should_promote(model(covered), window, min_window=20, target=0.99, after_hours=24, now=now)
        is expected
    )


def test_should_promote_at_exact_target_boundaries():
    m = model(covered=0.99, trained_at=1000.0)
    window = Window(20, 0.99)
    assert (
        should_promote(m, window, min_window=20, target=0.99, after_hours=24, now=87_400.0) is True
    )


def test_should_promote_ages_from_trained_at_not_towards_it():
    m = model(covered=0.995, trained_at=-100_000.0)
    window = Window(20, 0.995)
    assert (
        should_promote(m, window, min_window=20, target=0.99, after_hours=24, now=90_000.0) is True
    )


def test_should_check_is_deterministic_and_proportional():
    hashes = [f"{i:08x}" + "0" * 56 for i in range(0, 16**8, 16**8 // 1000)]
    picked = [h for h in hashes if should_check(h, 0.02)]
    assert 15 <= len(picked) <= 25
    assert should_check(hashes[0], 0.02) is should_check(hashes[0], 0.02)
    assert not any(should_check(h, 0.0) for h in hashes)


def test_should_check_reads_the_prefix_as_base_16():
    assert should_check("30303030" + "0" * 56, 0.2) is True


def test_should_promote_with_no_threshold_is_false():
    m = replace(model(), threshold=None)
    assert (
        should_promote(
            m, Window(20, 0.995), min_window=20, target=0.99, after_hours=24, now=90_000.0
        )
        is False
    )


@pytest.mark.parametrize(
    "bad_hash",
    ["1234", "ghijklmn", "-1234567", "1_234567"],
    ids=["short", "non-hex", "signed", "underscored"],
)
def test_should_check_rejects_bad_hash(bad_hash):
    with pytest.raises(ValueError, match="not an 8-char hex prefix"):
        should_check(bad_hash, 0.5)
