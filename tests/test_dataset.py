import pytest

from stuntd.store.db import Example
from stuntd.train.dataset import MAX_NUMBER_LABELS, NotTrainable, build_dataset

CHOICE = (
    '{"properties":{"verdict":{"enum":["allow","review","block"],"type":"string"}},"type":"object"}'
)
BOOL = '{"properties":{"spam":{"type":"boolean"}},"type":"object"}'
NUMBER = '{"properties":{"priority":{"type":"integer"}},"type":"object"}'


def examples(pairs):
    return [Example(text, answer, float(i)) for i, (text, answer) in enumerate(pairs)]


def test_choice_labels_follow_the_schema():
    data = build_dataset(
        "s",
        "choice",
        CHOICE,
        examples([("a", "block"), ("b", "allow"), ("c", "allow"), ("d", "review"), ("e", "allow")]),
        2,
        0.25,
    )
    assert data.field == "verdict" and data.labels == ("allow", "review", "block")
    assert [i.label for i in data.train] == [2, 0, 0]
    assert [i.label for i in data.holdout] == [1, 0]


def test_holdout_is_the_latest_slice():
    pairs = [(f"t{i}", "true" if i % 2 else "false") for i in range(10)]
    data = build_dataset("s", "boolean", BOOL, examples(pairs), 2, 0.2)
    assert len(data.holdout) == 2 and len(data.train) == 8
    assert max(i.created_at for i in data.train) < min(i.created_at for i in data.holdout)


def test_duplicates_keep_the_latest_answer():
    pairs = [
        ("same", "true"),
        ("x", "false"),
        ("y", "true"),
        ("same", "false"),
        ("z", "true"),
        ("w", "false"),
    ]
    data = build_dataset("s", "boolean", BOOL, examples(pairs), 2, 0.25)
    assert data.deduplicated == 1 and len(data.train) == 3 and len(data.holdout) == 2
    kept = {i.text: i.label for i in data.train + data.holdout}
    assert kept["same"] == 0


def test_answers_outside_the_labels_are_dropped():
    pairs = [
        ("a", "allow"),
        ("b", "maybe"),
        ("c", "block"),
        ("d", "allow"),
        ("e", "block"),
        ("f", "allow"),
    ]
    data = build_dataset("s", "choice", CHOICE, examples(pairs), 2, 0.25)
    assert data.dropped == 1 and len(data.train) + len(data.holdout) == 5


def test_numbers_become_sorted_choices():
    data = build_dataset(
        "s",
        "number",
        NUMBER,
        examples([("a", "10"), ("b", "2"), ("c", "10"), ("d", "3"), ("e", "2")]),
        2,
        0.25,
    )
    assert data.labels == ("2", "3", "10")


def test_number_spellings_of_one_value_share_a_label():
    pairs = [("a", "1"), ("b", "2"), ("c", "1.0"), ("d", "2"), ("e", "1")]
    data = build_dataset("s", "number", NUMBER, examples(pairs), 2, 0.25)
    assert data.labels == ("1", "2") and data.dropped == 0
    labelled = {i.text: i.label for i in data.train + data.holdout}
    assert labelled["a"] == labelled["c"] == 0


def test_non_finite_numbers_are_dropped():
    pairs = [("a", "1"), ("b", "nan"), ("c", "2"), ("d", "inf"), ("e", "2"), ("f", "1"), ("g", "2")]
    data = build_dataset("s", "number", NUMBER, examples(pairs), 2, 0.25)
    assert data.labels == ("1", "2") and data.dropped == 2
    assert len(data.train) + len(data.holdout) == 5


def test_replaced_duplicates_leave_no_number_label():
    pairs = [("a", "7"), ("b", "1"), ("c", "2"), ("a", "1"), ("d", "2"), ("e", "1")]
    data = build_dataset("s", "number", NUMBER, examples(pairs), 2, 0.25)
    assert data.labels == ("1", "2") and data.deduplicated == 1


def test_too_many_numbers_are_not_trainable():
    pairs = [(f"t{i}", str(i)) for i in range(MAX_NUMBER_LABELS + 1)] * 2
    with pytest.raises(NotTrainable, match="distinct numbers"):
        build_dataset("s", "number", NUMBER, examples(pairs), 2, 0.2)


@pytest.mark.parametrize(
    ("pairs", "message"),
    [
        ([("a", "true")] * 3, "examples after deduplication"),
        ([("a", "true"), ("b", "true"), ("c", "true"), ("d", "false")], "same answer"),
        ([("a", "true"), ("b", "false"), ("c", "true"), ("d", "true")], "single example"),
        (
            [
                ("a", "true"),
                ("b", "false"),
                ("c", "true"),
                ("d", "false"),
                ("e", "true"),
                ("f", "false"),
                ("g", "true"),
                ("h", "true"),
            ],
            "held-out examples",
        ),
    ],
    ids=["too-few", "one-class-train", "one-row-holdout", "one-class-holdout"],
)
def test_thin_sites_are_not_trainable(pairs, message):
    with pytest.raises(NotTrainable, match=message):
        build_dataset("s", "boolean", BOOL, examples(pairs), 3, 0.25)


def test_bad_schema_is_not_trainable():
    with pytest.raises(NotTrainable, match="single typed field"):
        build_dataset("s", "choice", '{"type":"object"}', examples([("a", "x")] * 4), 2, 0.25)


JEV_CHOICE = (
    '{"criteria":{"angry":null,"calm":null},"instructions":"What is the tone?","type":"choice"}'
)
JEV_NOUL = '{"criteria":null,"instructions":null,"type":"noul"}'
JEV_SCORE = '{"criteria":["low","mid","high"],"instructions":"How urgent?","type":"score"}'


def test_a_jev_choice_canonical_names_its_criteria():
    pairs = [("a", "angry"), ("b", "calm"), ("c", "angry"), ("d", "calm"), ("e", "angry")]
    data = build_dataset("tone", "choice", JEV_CHOICE, examples(pairs), 2, 0.25)
    assert data.field == "tone" and data.labels == ("angry", "calm")
    assert [i.label for i in data.train] == [0, 1, 0]
    assert [i.label for i in data.holdout] == [1, 0]


def test_a_jev_noul_canonical_carries_the_two_boolean_labels():
    pairs = [(f"t{i}", "true" if i % 2 else "false") for i in range(8)]
    data = build_dataset("billing", "boolean", JEV_NOUL, examples(pairs), 2, 0.25)
    assert data.field == "billing" and data.labels == ("false", "true")


def test_a_jev_score_canonical_keeps_every_level():
    pairs = [("a", "0"), ("b", "2"), ("c", "0"), ("d", "2"), ("e", "0"), ("f", "2")]
    data = build_dataset("urgency", "number", JEV_SCORE, examples(pairs), 2, 0.25)
    assert data.labels == ("0", "1", "2")
    assert [i.label for i in data.train] == [0, 2, 0, 2]


def test_a_canonical_that_is_neither_schema_nor_jev_is_not_trainable():
    with pytest.raises(NotTrainable, match="single typed field"):
        build_dataset("s", "choice", '{"type":"choice"}', examples([("a", "x")] * 4), 2, 0.25)
