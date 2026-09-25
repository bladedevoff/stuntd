import zlib
from itertools import pairwise
from types import SimpleNamespace

import pytest

from stuntd.train.artifacts import SiteModel
from stuntd.train.dataset import Item, NotTrainable, SiteDataset
from stuntd.train.layout import Layout, choose_layout, question_for

CHECKPOINT = Layout(max_len=512, head_max_len=192, spaced_labels=False)
BANKING = [f"card_payment_not_recognised_{index}" for index in range(77)]
LONG_FIELD = " ".join(["long"] * 40)
SEP = 3


def words(text):
    return len(text.split())


class WordTokenizer:
    mask_token = "[MASK]"
    pad_token_id, mask_token_id, cls_token_id, sep_token_id = 0, 1, 2, 3

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [zlib.crc32(word.encode()) + 4 for word in text.split()]}


def choose(labels, max_option_tokens=1024, field="intent", max_positions=8192):
    return choose_layout(field, labels, words, CHECKPOINT, max_option_tokens, max_positions)


def test_layout_for_short_labels_keeps_the_checkpoint_lengths():
    assert choose(["allow", "block"]) == Layout(512, 192, spaced_labels=True)


def test_layout_for_a_long_label_list_widens_both_lengths_alike():
    assert choose(BANKING) == Layout(512 + 286, 192 + 286, spaced_labels=True)


@pytest.mark.parametrize(
    ("cap", "expected"),
    [
        (478, Layout(798, 478, True)),
        (477, Layout(797, 477, True)),
        (300, Layout(620, 300, True)),
        (100, Layout(512, 192, True)),
    ],
    ids=["exactly the need", "one below the need", "above the checkpoint", "below the checkpoint"],
)
def test_layout_stops_at_the_option_cap(cap, expected):
    assert choose(BANKING, max_option_tokens=cap) == expected


@pytest.mark.parametrize(
    ("max_positions", "expected"),
    [(700, Layout(700, 380, True)), (798, Layout(798, 478, True)), (400, Layout(512, 192, True))],
    ids=["below the need", "exactly the need", "below the checkpoint"],
)
def test_layout_stays_within_the_encoder_positions(max_positions, expected):
    assert choose(BANKING, max_positions=max_positions) == expected


def test_layout_counts_a_long_option_at_laya_cut():
    labels = [" ".join([f"w{index}"] * 60) for index in range(10)]
    assert choose(labels).head_max_len == 10 * 49 + 16


@pytest.mark.parametrize(
    ("field", "expected"),
    [("", 77 * 6 + 16), ("intent", 77 * 6 + 16), (LONG_FIELD, 77 * 6 + 43)],
    ids=["empty field", "short field", "long field"],
)
def test_layout_makes_room_for_the_instruction(field, expected):
    assert choose(BANKING, field=field).head_max_len == expected


@pytest.mark.parametrize(
    ("labels", "spaced"),
    [
        (["a_b", "c_d"], True),
        (["a_b", "a b"], False),
        (["a_b c", "a b_c"], False),
        (["plain", "words"], True),
    ],
    ids=["distinct", "colliding", "colliding once both are spaced", "no underscores"],
)
def test_layout_spaces_labels_only_when_they_stay_distinct(labels, spaced):
    assert choose(labels).spaced_labels is spaced


@pytest.mark.parametrize(
    ("spaced", "options"),
    [(True, ["refund request", "other"]), (False, ["refund_request", "other"])],
    ids=["spaced", "raw"],
)
def test_question_shows_the_labels_the_layout_asks_for(spaced, options):
    question = question_for("intent", ["refund_request", "other"], spaced)
    assert question == {"t": "choice", "ins": "Choose intent", "crit": dict.fromkeys(options)}


@pytest.fixture(scope="module")
def site_row():
    pytest.importorskip("laya")
    from stuntd.train.trainer import site_row

    return site_row


def option_lengths(row):
    markers = row["markers"]
    closing = row["ids"].index(SEP, markers[-1])
    return [after - before for before, after in pairwise([*markers, closing])]


def narrower(layout):
    return Layout(layout.max_len - 1, layout.head_max_len - 1, layout.spaced_labels)


def test_chosen_layout_keeps_every_option_whole(site_row):
    layout = choose(BANKING)
    row = site_row(WordTokenizer(), "user: hi", "intent", BANKING, layout)
    assert option_lengths(row) == [6] * 77
    cut = site_row(WordTokenizer(), "user: hi", "intent", BANKING, narrower(layout))
    assert max(option_lengths(cut)) < 6


def test_chosen_layout_keeps_a_long_instruction_whole(site_row):
    instruction = WordTokenizer()(f"choice question: Choose {LONG_FIELD}")["input_ids"]
    layout = choose(BANKING, field=LONG_FIELD)
    row = site_row(WordTokenizer(), "user: hi", LONG_FIELD, BANKING, layout)
    assert row["ids"][1 : len(instruction) + 2] == [*instruction, SEP]
    cut = site_row(WordTokenizer(), "user: hi", LONG_FIELD, BANKING, narrower(layout))
    assert cut["ids"][1 : len(instruction) + 1] == [*instruction[:-1], SEP]


def site_model(labels, max_len=None, head_max_len=None, spaced_labels=False):
    return SiteModel(
        site="bank",
        kind="choice",
        field="intent",
        labels=list(labels),
        base_model="laya",
        temperature=1.0,
        threshold=None,
        target_agreement=0.95,
        trained_at=0.0,
        n_train=1,
        n_holdout=1,
        agreement=1.0,
        coverage=None,
        covered_agreement=None,
        ece=0.0,
        per_class={},
        confident_errors=[],
        curve=[],
        max_len=max_len,
        head_max_len=head_max_len,
        spaced_labels=spaced_labels,
    )


def fake_agent(max_len=512, head_max_len=192, max_positions=8192):
    config = SimpleNamespace(max_position_embeddings=max_positions)
    return SimpleNamespace(
        tok=WordTokenizer(),
        cfg={"max_len": max_len, "head_max_len": head_max_len},
        model=SimpleNamespace(encoder=SimpleNamespace(config=config)),
    )


@pytest.mark.parametrize(
    ("model", "layout"),
    [
        (site_model(BANKING, 798, 478, True), Layout(798, 478, True)),
        (site_model(BANKING), CHECKPOINT),
    ],
    ids=["trained with a layout", "written by 0.1.0"],
)
def test_decider_builds_rows_with_the_site_layout(site_row, model, layout):
    from stuntd.serve.decider import Decider

    decider = Decider.__new__(Decider)
    decider._agent = fake_agent()
    expected = site_row(WordTokenizer(), "user: hi", "intent", BANKING, layout)
    assert decider._row("user: hi", model) == expected


def trainer_for(agent, max_option_tokens):
    pytest.importorskip("laya")
    from stuntd.train.trainer import LayaTrainer

    trainer = LayaTrainer.__new__(LayaTrainer)
    trainer._agent = agent
    trainer._max_option_tokens = max_option_tokens
    return trainer


def dataset(labels):
    items = tuple(Item(f"user: message {index}", index % 2, 0.0) for index in range(4))
    return SiteDataset("bank", "choice", "intent", tuple(labels), items[:3], items[3:], 0, 0)


def test_trainer_rows_follow_the_layout_it_chose(site_row):
    trainer = trainer_for(fake_agent(), 1024)
    layout = trainer._layout(dataset(BANKING))
    train, holdout = trainer._items(dataset(BANKING), layout)
    assert layout == choose(BANKING)
    assert holdout == [
        {**site_row(WordTokenizer(), "user: message 3", "intent", BANKING, layout), "label": 1}
    ]
    assert all(option_lengths(row) == [6] * 77 for row in train)


def test_trainer_widens_no_further_than_the_encoder_positions():
    trainer = trainer_for(fake_agent(max_positions=700), 1024)
    assert trainer._layout(dataset(BANKING)) == Layout(700, 380, True)


def test_trainer_refuses_labels_that_outgrow_the_option_cap(site_row):
    labels = [f"label {index}" for index in range(30)]
    trainer = trainer_for(fake_agent(max_len=64, head_max_len=32), 32)
    with pytest.raises(NotTrainable, match="training.max_option_tokens"):
        trainer._items(dataset(labels), trainer._layout(dataset(labels)))


def test_question_for_keeps_its_0_1_0_import_and_call():
    pytest.importorskip("laya")
    from stuntd.train.trainer import question_for as from_trainer

    assert from_trainer("intent", ["card_arrival", "age_limit"]) == {
        "t": "choice",
        "ins": "Choose intent",
        "crit": {"card_arrival": None, "age_limit": None},
    }
