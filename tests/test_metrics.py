import math
import random

import pytest

from stuntd.train.metrics import (
    ERROR_TEXT_CHARS,
    Prediction,
    confidence,
    confident_errors,
    ece,
    fit_temperature,
    operating_curve,
    operating_point,
    per_class,
    predict,
    quantile,
    softmax,
    wilson_interval,
)


def brute_force_temperature(logits, labels):
    def loss(scale):
        return sum(
            -math.log(softmax(row, 1.0 / scale)[label])
            for row, label in zip(logits, labels, strict=True)
        ) / len(logits)

    grid = [0.05 + (20.0 - 0.05) * step / 1999 for step in range(2000)]
    return 1.0 / min(grid, key=loss)


def noisy_holdout(flipped):
    rng = random.Random(7)
    logits = [[rng.gauss(0.0, 2.0) for _ in range(3)] for _ in range(40)]
    labels = [max(range(3), key=lambda index: row[index]) for row in logits]
    for row in flipped(logits):
        labels[row] = min(range(3), key=lambda index: logits[row][index])
    return logits, labels


def every_tenth(logits):
    return range(0, len(logits), 10)


def surest_row(logits):
    def margin(index):
        ordered = sorted(logits[index])
        return ordered[-1] - ordered[-2]

    return [max(range(len(logits)), key=margin)]


CURVE = [
    Prediction(0, 0, 0.95, 0.95),
    Prediction(0, 0, 0.9, 0.9),
    Prediction(1, 0, 0.8, 0.8),
    Prediction(1, 1, 0.7, 0.7),
    Prediction(0, 0, 0.6, 0.6),
]


def test_softmax_with_temperature():
    assert softmax([0.0, 0.0]) == pytest.approx([0.5, 0.5])
    assert softmax([2.0, 0.0], temperature=2.0) == pytest.approx([0.7311, 0.2689], abs=1e-4)


@pytest.mark.parametrize(
    ("probs", "expected"),
    [([0.5, 0.5], 0.0), ([1.0, 0.0], 1.0), ([0.25] * 4, 0.0), ([1.0], 1.0)],
    ids=["even", "certain", "even-four", "single"],
)
def test_confidence_is_normalised_entropy(probs, expected):
    assert confidence(probs) == pytest.approx(expected, abs=1e-9)


def test_fit_temperature_matches_the_closed_form():
    logits = [[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]
    labels = [0, 1, 0, 0]
    assert fit_temperature(logits, labels) == pytest.approx(2 / math.log(3), abs=0.01)


@pytest.mark.parametrize(
    "flipped", [every_tenth, surest_row], ids=["scattered-noise", "one-sure-mistake"]
)
def test_fit_temperature_matches_a_brute_force_search(flipped):
    logits, labels = noisy_holdout(flipped)
    expected = brute_force_temperature(logits, labels)
    assert fit_temperature(logits, labels) == pytest.approx(expected, abs=1e-2)


@pytest.mark.parametrize("temperature", [0.0, -1.0], ids=["zero", "negative"])
def test_softmax_rejects_a_non_positive_temperature(temperature):
    with pytest.raises(ValueError, match="temperature must be positive"):
        softmax([1.0, 0.0], temperature)


def test_confidence_never_dips_below_zero():
    assert confidence([0.2] * 5) == 0.0


def test_predict_applies_the_temperature():
    preds = predict([[2.0, 0.0]], [1], temperature=2.0)
    expected = softmax([1.0, 0.0])
    assert preds == [
        Prediction(1, 0, pytest.approx(confidence(expected)), pytest.approx(expected[0]))
    ]


def test_ece_weights_bins_by_size():
    preds = [
        Prediction(0, 0, 0.9, 0.9),
        Prediction(0, 0, 0.9, 0.9),
        Prediction(0, 0, 0.55, 0.55),
        Prediction(0, 1, 0.55, 0.55),
    ]
    assert ece(preds) == pytest.approx(0.075)
    assert ece([]) == 0.0


def test_ece_skips_a_zero_confidence():
    assert ece([Prediction(0, 0, 0.0, 0.0)]) == 0.0


def test_operating_curve_has_one_point_per_confidence():
    points = [(p.threshold, p.coverage, p.agreement) for p in operating_curve(CURVE)]
    assert points == pytest.approx(
        [(0.95, 0.2, 1.0), (0.9, 0.4, 1.0), (0.8, 0.6, 2 / 3), (0.7, 0.8, 0.75), (0.6, 1.0, 0.8)]
    )


def test_tied_confidences_stay_together():
    points = operating_curve([Prediction(0, 0, 0.9, 0.9), Prediction(0, 1, 0.9, 0.9)])
    assert len(points) == 1 and points[0].coverage == 1.0 and points[0].agreement == 0.5


@pytest.mark.parametrize(
    ("target", "expected"),
    [(0.99, (0.9, 0.4, 1.0)), (0.75, (0.6, 1.0, 0.8)), (1.01, None)],
    ids=["strict", "loose", "unreachable"],
)
def test_operating_point_takes_the_widest_prefix(target, expected):
    point = operating_point(CURVE, target)
    got = None if point is None else (point.threshold, point.coverage, point.agreement)
    assert got == (expected if expected is None else pytest.approx(expected))


def test_per_class_counts_support_and_agreement():
    preds = [Prediction(0, 0, 0.9, 0.9), Prediction(0, 1, 0.8, 0.8), Prediction(1, 1, 0.7, 0.7)]
    stats = per_class(preds, ["allow", "block", "review"])
    assert stats["allow"].support == 2 and stats["allow"].agreement == 0.5
    assert stats["block"].support == 1 and stats["block"].agreement == 1.0
    assert stats["review"].support == 0 and stats["review"].agreement is None


def test_confident_errors_are_the_surest_mistakes():
    texts = ["a", "b", "c", "d", "e"]
    errors = confident_errors(CURVE, texts, limit=5)
    assert [(e.text, e.expected, e.predicted) for e in errors] == [("c", 1, 0)]
    assert confident_errors(CURVE, texts, limit=0) == []


def test_confident_error_text_is_cut_and_flattened():
    long_text = "user:\n" + "spam " * 98 + "spam"
    assert len(long_text) == 500
    errors = confident_errors(CURVE, [long_text] * 5)
    assert len(errors[0].text) <= ERROR_TEXT_CHARS
    assert errors[0].text.startswith("user: spam spam")


def test_predict_keeps_the_probability_of_the_chosen_answer():
    preds = predict([[2.0, 0.0], [0.0, 1.0]], [0, 0], temperature=1.0)
    assert [pred.probability for pred in preds] == [
        pytest.approx(softmax([2.0, 0.0])[0]),
        pytest.approx(softmax([0.0, 1.0])[1]),
    ]


def test_confident_errors_carry_the_probability_of_the_wrong_answer():
    errors = confident_errors([Prediction(0, 1, 0.8, 0.9), Prediction(0, 1, 0.4, 0.6)], ["a", "b"])
    assert [error.probability for error in errors] == [0.9, 0.6]


@pytest.mark.parametrize(
    ("correct", "total", "low", "high"),
    [
        (10, 10, 0.7225, 1.0),
        (5, 10, 0.2366, 0.7634),
        (0, 10, 0.0, 0.2775),
        (95, 100, 0.8882, 0.9784),
    ],
    ids=["all-right", "half", "none-right", "large-sample"],
)
def test_wilson_interval_matches_hand_computed_values(correct, total, low, high):
    interval = wilson_interval(correct, total)
    assert interval.low == pytest.approx(low, abs=1e-4)
    assert interval.high == pytest.approx(high, abs=1e-4)


def test_wilson_interval_narrows_with_more_rows():
    small, large = wilson_interval(9, 10), wilson_interval(900, 1000)
    assert large.high - large.low < small.high - small.low


@pytest.mark.parametrize(
    ("values", "level", "expected"),
    [
        ([4.0, 0.0, 2.0, 1.0, 3.0], 0.95, 3.8),
        ([4.0, 0.0, 2.0, 1.0, 3.0], 0.5, 2.0),
        ([4.0, 0.0, 2.0, 1.0, 3.0], 1.0, 4.0),
        ([0.7], 0.95, 0.7),
    ],
    ids=["interpolated", "median", "maximum", "single value"],
)
def test_quantile_interpolates_between_ranks(values, level, expected):
    assert quantile(values, level) == pytest.approx(expected)
