from __future__ import annotations

import math

import pytest
from hypothesis import assume, event, given, settings, strategies as st

from stuntd.store.db import Example
from stuntd.train.dataset import MAX_NUMBER_LABELS, NotTrainable, build_dataset
from stuntd.train.metrics import (
    ClassStats,
    ConfidentError,
    Prediction,
    confidence,
    confident_errors,
    ece,
    fit_temperature,
    operating_curve,
    operating_point,
    per_class,
    predict,
    softmax,
)

BOOL = '{"properties":{"refund":{"type":"boolean"}},"type":"object"}'

logits_strategy = st.lists(st.floats(-20, 20), min_size=2, max_size=6)
temperature_strategy = st.floats(0.05, 20)
prediction_strategy = st.builds(
    Prediction,
    label=st.integers(0, 3),
    predicted=st.integers(0, 3),
    confidence=st.floats(0.0, 1.0),
)


def first_max(values):
    return max(range(len(values)), key=lambda index: values[index])


@given(logits=logits_strategy, temperature=temperature_strategy)
@settings(max_examples=200)
def test_softmax_is_a_distribution(logits, temperature):
    probs = softmax(logits, temperature)
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 <= p <= 1.0 for p in probs)


@given(logits=logits_strategy, temperature=temperature_strategy)
@settings(max_examples=200)
def test_softmax_preserves_the_argmax(logits, temperature):
    ordered = sorted(logits)
    assume(ordered[-1] - ordered[-2] >= 1e-9)
    probs = softmax(logits, temperature)
    assert first_max(probs) == first_max(logits)


@given(size=st.integers(2, 8))
@settings(max_examples=50)
def test_confidence_is_zero_for_a_uniform_distribution(size):
    assert confidence([1.0 / size] * size) == pytest.approx(0.0, abs=1e-9)


@given(size=st.integers(2, 8), index=st.integers(0, 7))
@settings(max_examples=50)
def test_confidence_is_one_for_a_one_hot_distribution(size, index):
    place = index % size
    probs = [1.0 if i == place else 0.0 for i in range(size)]
    assert confidence(probs) == 1.0


@given(logits=logits_strategy, temperature=temperature_strategy)
@settings(max_examples=200)
def test_confidence_stays_within_bounds(logits, temperature):
    probs = softmax(logits, temperature)
    assert 0.0 <= confidence(probs) <= 1.0


def nll(logits, labels, scale):
    total = 0.0
    for row, label in zip(logits, labels, strict=True):
        scaled = [logit * scale for logit in row]
        highest = max(scaled)
        shifted = sum(math.exp(value - highest) for value in scaled)
        total += highest + math.log(shifted) - scaled[label]
    return total / len(logits)


@st.composite
def logits_and_labels(draw):
    classes = draw(st.integers(2, 4))
    rows = draw(st.integers(5, 15))
    logits = draw(
        st.lists(
            st.lists(st.floats(-20, 20), min_size=classes, max_size=classes),
            min_size=rows,
            max_size=rows,
        )
    )
    labels = draw(st.lists(st.integers(0, classes - 1), min_size=rows, max_size=rows))
    return logits, labels


@given(data=logits_and_labels())
@settings(max_examples=200)
def test_fit_temperature_does_not_lose_to_a_grid_search(data):
    logits, labels = data
    fitted = fit_temperature(logits, labels)
    assert 0.05 <= fitted <= 20.0
    fitted_nll = nll(logits, labels, 1.0 / fitted)
    grid = [0.05 + (20.0 - 0.05) * step / 39 for step in range(40)]
    grid_nll = min(nll(logits, labels, scale) for scale in grid)
    assert fitted_nll <= grid_nll + 1e-6


def test_fit_temperature_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="argument 2 is shorter"):
        fit_temperature([[1.0, 0.0], [1.0, 0.0]], [0])


@given(
    predictions=st.lists(prediction_strategy, min_size=1, max_size=20),
    target=st.floats(-10.0, 0.0),
)
@settings(max_examples=200)
def test_operating_point_covers_everything_below_zero(predictions, target):
    point = operating_point(predictions, target)
    assert point is not None
    assert point.coverage == 1.0


@given(predictions=st.lists(prediction_strategy, min_size=1, max_size=20), data=st.data())
@settings(max_examples=200)
def test_operating_point_coverage_does_not_grow_with_the_target(predictions, data):
    achievable = sorted({point.agreement for point in operating_curve(predictions)})
    first = data.draw(st.sampled_from(achievable))
    second = data.draw(st.sampled_from(achievable))
    low, high = min(first, second), max(first, second)
    lower = operating_point(predictions, low)
    higher = operating_point(predictions, high)
    if higher is not None:
        event("higher-defined")
        assert lower is not None
        assert lower.coverage >= higher.coverage
    else:
        event("higher-undefined")


@given(predictions=st.lists(prediction_strategy, min_size=0, max_size=20))
@settings(max_examples=200)
def test_ece_stays_within_bounds(predictions):
    assert 0.0 <= ece(predictions) <= 1.0


@given(
    core_pairs=st.integers(4, 12),
    holdout=st.floats(0.2, 0.4),
    min_examples=st.integers(2, 4),
    maybe_count=st.integers(0, 5),
    duplicate_targets=st.lists(st.integers(0, 23), max_size=5),
)
@settings(max_examples=200)
def test_build_dataset_conserves_examples_and_holds_out_the_latest(
    core_pairs, holdout, min_examples, maybe_count, duplicate_targets
):
    core_size = 2 * core_pairs
    core = [Example(f"core-{i}", "true" if i % 2 else "false", float(i)) for i in range(core_size)]
    maybe_rows = [Example(f"maybe-{j}", "maybe", -1000.0 + j) for j in range(maybe_count)]
    duplicate_rows = [
        Example(f"core-{target % core_size}", "false", float(target % core_size) - 1.0)
        for target in duplicate_targets
    ]
    examples = core + maybe_rows + duplicate_rows

    try:
        data = build_dataset("s", "boolean", BOOL, examples, min_examples, holdout)
    except NotTrainable as exc:
        event("not-trainable")
        message = str(exc)
        assert (
            message == "training examples all carry the same answer"
            or message == "held-out examples all carry the same answer"
            or message.endswith(f"examples after deduplication, {min_examples} needed")
        )
        return
    event("trainable")

    assert len(data.train) + len(data.holdout) + data.dropped + data.deduplicated == len(examples)
    assert max(item.created_at for item in data.train) < min(
        item.created_at for item in data.holdout
    )


def test_fit_temperature_converges_to_the_max_scale_when_logits_carry_no_signal():
    logits = [[0.0, 0.0, 0.0]] * 6
    labels = [0, 1, 2, 0, 1, 2]
    assert fit_temperature(logits, labels) == pytest.approx(20.0, abs=1e-4)


def reference_ece(predictions, bins):
    if not predictions:
        return 0.0
    buckets = [[] for _ in range(bins)]
    for prediction in predictions:
        if prediction.confidence <= 0.0:
            continue
        place = min(bins - 1, max(0, math.ceil(prediction.confidence * bins) - 1))
        buckets[place].append(prediction)
    total = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        stated = sum(p.confidence for p in bucket) / len(bucket)
        correct = sum(p.label == p.predicted for p in bucket) / len(bucket)
        total += len(bucket) / len(predictions) * abs(stated - correct)
    return total


@given(predictions=st.lists(prediction_strategy, min_size=0, max_size=20), bins=st.integers(1, 20))
@settings(max_examples=200)
def test_ece_matches_an_independent_reference(predictions, bins):
    assert ece(predictions, bins) == pytest.approx(reference_ece(predictions, bins), abs=1e-9)


def reference_per_class(predictions, labels):
    stats = {}
    for index, name in enumerate(labels):
        carried = [p for p in predictions if p.label == index]
        if not carried:
            stats[name] = ClassStats(0, None)
            continue
        correct = sum(p.label == p.predicted for p in carried)
        stats[name] = ClassStats(len(carried), correct / len(carried))
    return stats


@given(
    predictions=st.lists(prediction_strategy, min_size=0, max_size=20),
    label_count=st.integers(2, 4),
)
@settings(max_examples=200)
def test_per_class_matches_an_independent_reference(predictions, label_count):
    labels = [f"label-{i}" for i in range(label_count)]
    assert per_class(predictions, labels) == reference_per_class(predictions, labels)


def test_confident_errors_defaults_to_five():
    predictions = [Prediction(0, 1, 0.5 + i / 100) for i in range(7)]
    texts = [str(i) for i in range(7)]
    assert len(confident_errors(predictions, texts)) == 5


def test_confident_errors_orders_several_mistakes_by_confidence():
    predictions = [Prediction(0, 1, 0.4), Prediction(0, 1, 0.9), Prediction(0, 1, 0.6)]
    texts = ["low", "high", "mid"]
    errors = confident_errors(predictions, texts, limit=3)
    assert [e.text for e in errors] == ["high", "mid", "low"]
    assert errors[0] == ConfidentError("high", 0, 1, 0.9)


def test_confident_errors_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="argument 2 is longer"):
        confident_errors([Prediction(0, 1, 0.5)], ["a", "b"])


def test_number_labels_at_the_maximum_are_still_trainable():
    number_schema = '{"properties":{"n":{"type":"integer"}},"type":"object"}'
    pairs = [(f"t{i}", str(i)) for i in range(MAX_NUMBER_LABELS)] * 2
    examples = [Example(text, answer, float(i)) for i, (text, answer) in enumerate(pairs)]
    data = build_dataset("s", "number", number_schema, examples, 2, 0.25)
    assert len(data.labels) == MAX_NUMBER_LABELS


def test_duplicates_keep_the_latest_answer_regardless_of_input_order():
    examples = [
        Example("same", "true", 10.0),
        Example("a", "false", 2.0),
        Example("b", "true", 3.0),
        Example("same", "false", 1.0),
        Example("c", "false", 4.0),
    ]
    data = build_dataset("s", "boolean", BOOL, examples, 2, 0.5)
    kept = {item.text: item.label for item in data.train + data.holdout}
    assert kept["same"] == 1


def test_single_item_holdout_is_not_trainable():
    examples = [Example(f"t{i}", "true" if i % 2 else "false", float(i)) for i in range(20)]
    with pytest.raises(NotTrainable, match="^held-out slice has a single example"):
        build_dataset("s", "boolean", BOOL, examples, 2, 0.03)


def test_build_dataset_accepts_exactly_min_examples():
    examples = [
        Example("a", "true", 0.0),
        Example("b", "false", 1.0),
        Example("c", "true", 2.0),
        Example("d", "false", 3.0),
    ]
    data = build_dataset("s", "boolean", BOOL, examples, 4, 0.5)
    assert len(data.train) + len(data.holdout) == 4


def test_build_dataset_reports_invalid_json():
    with pytest.raises(NotTrainable, match="^schema is not a single typed field$"):
        build_dataset("s", "boolean", "not json", [Example("a", "true", 0.0)] * 4, 2, 0.25)


def test_build_dataset_reports_a_schema_without_a_typed_field():
    with pytest.raises(NotTrainable, match="^schema is not a single typed field$"):
        build_dataset("s", "choice", '{"type":"object"}', [Example("a", "x", 0.0)] * 4, 2, 0.25)


def test_build_dataset_reports_a_single_class_train_split():
    pairs = [("a", "true"), ("b", "true"), ("c", "true"), ("d", "false")]
    examples = [Example(text, answer, float(i)) for i, (text, answer) in enumerate(pairs)]
    with pytest.raises(NotTrainable, match="^training examples all carry the same answer$"):
        build_dataset("s", "boolean", BOOL, examples, 3, 0.25)


def test_build_dataset_reports_a_single_class_holdout_split():
    pairs = [
        ("a", "true"),
        ("b", "false"),
        ("c", "true"),
        ("d", "false"),
        ("e", "true"),
        ("f", "true"),
    ]
    examples = [Example(text, answer, float(i)) for i, (text, answer) in enumerate(pairs)]
    with pytest.raises(NotTrainable, match="^held-out examples all carry the same answer$"):
        build_dataset("s", "boolean", BOOL, examples, 3, 0.3)


def test_deduplicate_breaks_a_tie_in_favour_of_the_later_entry():
    examples = [
        Example("x1", "true", 0.0),
        Example("x2", "false", 1.0),
        Example("x3", "true", 2.0),
        Example("x4", "false", 3.0),
        Example("same", "false", 4.0),
        Example("same", "true", 4.0),
    ]
    data = build_dataset("s", "boolean", BOOL, examples, 2, 0.25)
    kept = {item.text: item.label for item in data.train + data.holdout}
    assert kept["same"] == 1


def test_predict_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="argument 2 is shorter"):
        predict([[1.0, 0.0], [1.0, 0.0]], [0], temperature=1.0)


def test_ece_uses_fifteen_bins_by_default():
    predictions = [Prediction(0, 0, 0.61), Prediction(0, 1, 0.66)]
    assert ece(predictions) == pytest.approx(reference_ece(predictions, 15), abs=1e-9)


def test_ece_clamps_an_out_of_range_confidence_to_the_top_bin():
    predictions = [Prediction(0, 0, 1.5), Prediction(0, 1, 14.5 / 15)]
    assert ece(predictions) == pytest.approx(reference_ece(predictions, 15), abs=1e-9)
