from collections import Counter
from dataclasses import replace

import pytest
from decisions_helpers import FakeVerdict, site_model

from benchmarks.encoders import (
    Example,
    Measured,
    Result,
    captures,
    format_table,
    latency,
    massive_config,
    measure,
    stratified_sample,
)
from stuntd.settings import Settings
from stuntd.store.db import Example as StoredExample
from stuntd.train.dataset import build_dataset

LABELS = ("billing", "other", "shipping")


def examples(**counts):
    return [
        Example(f"{label} {number}", label)
        for number in range(max(counts.values()))
        for label, count in counts.items()
        if number < count
    ]


def verdict(label, confidence, novelty=0.1):
    return FakeVerdict(LABELS.index(label), confidence, 1, (), novelty)


def gated_model():
    return replace(site_model("s"), novelty_cutoff=0.3)


def test_sample_keeps_every_label_in_proportion():
    sample = stratified_sample(examples(billing=600, other=300, shipping=100), 100, 0)
    assert Counter(example.label for example in sample) == {
        "billing": 60,
        "other": 30,
        "shipping": 10,
    }


def test_sample_gives_the_leftover_places_to_the_largest_remainders():
    sample = stratified_sample(examples(billing=5, other=3, shipping=2), 4, 0)
    assert Counter(example.label for example in sample) == {"billing": 2, "other": 1, "shipping": 1}


def test_sample_is_the_same_for_the_same_seed_and_not_for_another():
    rows = examples(billing=50, other=50)
    assert stratified_sample(rows, 20, 0) == stratified_sample(rows, 20, 0)
    assert stratified_sample(rows, 20, 0) != stratified_sample(rows, 20, 1)


def test_sample_has_no_row_twice():
    sample = stratified_sample(examples(billing=50, other=50), 60, 0)
    assert len(set(sample)) == 60


def test_sample_larger_than_the_data_is_all_of_it():
    rows = examples(billing=3, other=2)
    assert sorted(stratified_sample(rows, 10, 0), key=lambda row: row.text) == sorted(
        rows, key=lambda row: row.text
    )


def test_captures_make_a_site_stuntd_can_train():
    stored = captures("massive-ru", LABELS, examples(billing=10, other=10, shipping=10))
    dataset = build_dataset(
        "massive-ru",
        stored[0].kind,
        stored[0].schema_canonical,
        [StoredExample(one.input_text, one.answer, float(at)) for at, one in enumerate(stored)],
        min_examples=10,
        holdout=0.2,
    )
    assert dataset.labels == LABELS
    assert dataset.field == "intent"
    assert len(dataset.train) + len(dataset.holdout) == 30


def test_measure_counts_each_answer_against_the_threshold_and_the_gate():
    verdicts = [
        verdict("billing", 0.9),
        verdict("billing", 0.8),
        verdict("other", 0.5),
        verdict("shipping", 0.9, novelty=0.5),
    ]
    expected = ["billing", "other", "other", "shipping"]
    assert measure(gated_model(), Settings(), verdicts, expected) == Measured(
        0.75, 0.75, 2 / 3, 0.5
    )


def test_measure_with_the_gate_off_answers_a_novel_request_locally():
    verdicts = [verdict("billing", 0.9, novelty=0.5)]
    measured = measure(gated_model(), Settings(novelty_gate=False), verdicts, ["billing"])
    assert measured.local_share == 1.0


def test_measure_without_a_threshold_covers_nothing():
    model = replace(gated_model(), threshold=None)
    assert measure(model, Settings(), [verdict("billing", 0.99)], ["billing"]) == Measured(
        1.0, None, None, 0.0
    )


def test_measure_with_nothing_above_the_threshold_has_no_accuracy_there():
    measured = measure(gated_model(), Settings(), [verdict("billing", 0.1)], ["billing"])
    assert measured == Measured(1.0, 0.0, None, 0.0)


def test_latency_is_the_median_and_95th_percentile_of_each_call():
    times = iter([0.0, 0.01, 1.0, 1.02, 2.0, 2.03, 3.0, 3.04, 4.0, 4.05])
    p50, p95 = latency(lambda text: None, list("abcde"), lambda: next(times))
    assert (p50, p95) == (pytest.approx(30.0), pytest.approx(48.0))


@pytest.mark.parametrize(
    ("language", "config"),
    [("ru", "ru"), ("en", "en"), ("zh", "zh-CN")],
    ids=["same", "english", "chinese"],
)
def test_massive_names_chinese_by_its_script(language, config):
    assert massive_config(language) == config


def test_table_has_a_row_per_result_and_a_dash_where_nothing_was_measured():
    results = [
        Result("massive-ru", "laya", Measured(0.8123, 0.5, 0.9876, 0.4), 12.34, 21.04, 30.55),
        Result("banking77", "e5", Measured(0.9, None, None, 0.0), 3.0, 5.0, 6.0),
    ]
    assert format_table(results).splitlines() == [
        "| Dataset | Encoder | Accuracy | Coverage | Accuracy at coverage | Local | Train s | p50 ms | p95 ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| massive-ru | laya | 81.2% | 50.0% | 98.8% | 40.0% | 12.3 | 21.0 | 30.6 |",
        "| banking77 | e5 | 90.0% | - | - | 0.0% | 3.0 | 5.0 | 6.0 |",
    ]
