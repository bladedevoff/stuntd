import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from stuntd.cli import main
from stuntd.store.db import Capture, Store
from stuntd.store.redact import Redactor
from stuntd.train.artifacts import SiteModel, save_model
from stuntd.train.gold import GoldRow, render_gold, score_gold

LABELS = ["refund", "deny"]
JEV_QUESTION = {
    "criteria": {"deny": None, "refund": None},
    "instructions": "Refund?",
    "type": "choice",
}
JEV_CANONICAL = json.dumps(JEV_QUESTION, sort_keys=True, separators=(",", ":"))


def site_model(threshold=0.8):
    return SiteModel(
        site="s1",
        kind="choice",
        field="s1",
        labels=LABELS,
        base_model="convaiinnovations/laya",
        temperature=1.0,
        threshold=threshold,
        target_agreement=0.99,
        trained_at=1_700_000_000.0,
        n_train=80,
        n_holdout=20,
        agreement=0.95,
        coverage=0.7,
        covered_agreement=0.99,
        ece=0.02,
        per_class={},
        confident_errors=[],
        curve=[],
    )


class FakeDecider:
    def __init__(self, verdicts, zero_shot=None, novelty=None):
        self.verdicts = verdicts
        self.zero_shot = zero_shot
        self.novelty = novelty or {}
        self.asked = []
        self.questions = []

    def decide(self, model, head_path, text):
        self.asked.append(text)
        label, confidence = self.verdicts[text]
        return SimpleNamespace(
            label=LABELS.index(label), confidence=confidence, novelty=self.novelty.get(text)
        )

    def answer(self, state, questions):
        self.questions.append(questions)
        return {
            "answers": {
                name: {"type": "choice", "choice": self.zero_shot[state]} for name in questions
            },
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }


def row(text, answer):
    return json.dumps({"text": text, "answer": answer})


def write_gold(tmp_path, lines):
    path = tmp_path / "gold.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def record(data_dir, *captures, schema="{}"):
    store = Store(data_dir / "captures.sqlite", Redactor())
    for text, answer in captures:
        store.record(Capture("s1", schema, "choice", text, answer, "gpt-x", 100, 1, 1))
    store.close()


@pytest.fixture
def trained(data_dir):
    save_model(data_dir / "models", site_model())


@pytest.fixture
def decider(monkeypatch):
    fake = FakeDecider(
        {
            "user: a": ("deny", 0.9),
            "user: b": ("deny", 0.5),
            "user: mail [email]": ("refund", 0.95),
        },
        {"user: a": "refund", "user: b": "refund", "user: mail [email]": "deny"},
    )
    monkeypatch.setattr("stuntd.cli._make_decider", lambda settings: fake)
    return fake


@pytest.fixture
def gold(tmp_path):
    return write_gold(
        tmp_path,
        [
            row("user: a", "deny"),
            row("user: b", "refund"),
            row("user: mail me@example.com", "refund"),
            row("user: d", "maybe"),
        ],
    )


def test_score_gold_counts_each_cell():
    rows = [
        GoldRow("refund", "refund", 0.9, "refund"),
        GoldRow("refund", "deny", 0.9, "deny"),
        GoldRow("refund", "refund", 0.9, "deny"),
        GoldRow("refund", "deny", 0.9, "refund"),
    ]
    score = score_gold(rows, 0.8, 0)
    assert (score.both_right, score.both_wrong, score.head_only, score.teacher_only) == (1, 1, 1, 1)
    assert score.head_accuracy == 0.5 and score.teacher_accuracy == 0.5
    assert score.teacher_rows == 4
    assert score.unseen_rows == 0 and score.unseen_head_accuracy is None


def test_score_gold_leaves_rows_without_a_teacher_out_of_served_accuracy():
    rows = [
        GoldRow("refund", "deny", 0.3, None),
        GoldRow("refund", "deny", 0.3, "refund"),
        GoldRow("deny", "deny", 0.9, None),
    ]
    score = score_gold(rows, 0.8, 0)
    assert score.unserved == 1
    assert score.served_accuracy == 1.0
    assert score.teacher_rows == 1 and score.teacher_accuracy == 1.0
    assert score.head_accuracy == pytest.approx(1 / 3)
    assert score.unseen_rows == 2 and score.unseen_head_accuracy == 0.5
    assert (score.both_right, score.both_wrong, score.head_only, score.teacher_only) == (0, 0, 0, 1)


@pytest.mark.parametrize(
    ("confidence", "local_share", "served_accuracy"),
    [(0.8, 1.0, 1.0), (0.79, 0.0, 0.0)],
    ids=["at-threshold", "just-below"],
)
def test_score_gold_threshold_boundary(confidence, local_share, served_accuracy):
    score = score_gold([GoldRow("refund", "refund", confidence, "deny")], 0.8, 0)
    assert score.local_share == local_share
    assert score.served_accuracy == served_accuracy


def test_score_gold_without_threshold_never_serves_the_head():
    rows = [GoldRow("refund", "refund", 1.0, "deny"), GoldRow("deny", "deny", 1.0, None)]
    score = score_gold(rows, None, 0)
    assert score.local_share == 0.0
    assert score.served_accuracy == 0.0
    assert score.unserved == 1
    assert score.head_accuracy == 1.0
    assert score.unseen_rows == 1 and score.unseen_head_accuracy == 1.0


def test_score_gold_counts_skipped_rows():
    score = score_gold([GoldRow("refund", "refund", 0.9, None)], 0.8, 3)
    assert (score.read, score.used, score.skipped) == (4, 1, 3)


def test_score_gold_without_usable_rows_has_no_rates():
    score = score_gold([], 0.8, 2)
    assert (score.read, score.used, score.skipped) == (2, 0, 2)
    assert score.head_accuracy is None and score.served_accuracy is None
    assert score.local_share is None and score.teacher_accuracy is None
    assert score.unseen_head_accuracy is None


def test_score_gold_serves_the_zero_shot_answer_below_the_threshold():
    rows = [
        GoldRow("refund", "deny", 0.3, None, zero_shot="refund"),
        GoldRow("deny", "deny", 0.9, None, zero_shot="refund"),
    ]
    score = score_gold(rows, 0.8, 0, "zeroshot")
    assert score.served_accuracy == 1.0
    assert score.local_share == 0.5
    assert score.unserved == 0


def test_score_gold_serves_the_head_below_the_threshold_when_the_fallback_is_the_head():
    rows = [
        GoldRow("refund", "deny", 0.3, None, zero_shot="refund"),
        GoldRow("deny", "deny", 0.9, None, zero_shot="refund"),
    ]
    score = score_gold(rows, 0.8, 0, "head")
    assert score.served_accuracy == 0.5
    assert score.local_share == 0.5


def test_score_gold_sends_a_novel_row_to_the_fallback():
    rows = [GoldRow("refund", "deny", 0.9, "refund", novel=True)]
    score = score_gold(rows, 0.8, 0)
    assert score.local_share == 0.0
    assert score.served_accuracy == 1.0


def test_render_gold_names_the_fallback():
    score = score_gold([GoldRow("deny", "deny", 0.9, None)], 0.8, 0, "zeroshot")
    assert (
        "served accuracy 1.000 at threshold 0.80: 100% answered by the head,"
        " the rest by the zeroshot fallback" in render_gold(site_model(), score)
    )


def test_render_gold_without_threshold_counts_rows_with_no_teacher():
    score = score_gold([GoldRow("deny", "deny", 1.0, None)], None, 0)
    assert (
        "served accuracy - at no threshold: 0% answered locally, 1 with no teacher answer"
        in render_gold(site_model(threshold=None), score)
    )


def test_report_gold_prints_the_scores(data_dir, trained, decider, gold, capsys):
    record(data_dir, ("user: a", "refund"), ("user: a", "deny"), ("user: b", "refund"))
    assert main(["report", "s1", "--gold", gold]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "site s1  gold rows 4: 3 used, 1 skipped",
        "head accuracy 0.667",
        "head accuracy 1.000 on 1 rows the store has not seen",
        "served accuracy 1.000 at threshold 0.80: 67% answered locally,"
        " 0 below it with no teacher answer",
        "teacher accuracy 1.000 on 2 rows",
        "head and teacher: both right 1, both wrong 0, head only 0, teacher only 1",
    ]
    assert decider.asked == ["user: a", "user: b", "user: mail [email]"]


def test_report_gold_in_local_mode_scores_the_zero_shot_fallback(
    data_dir, trained, decider, gold, capsys
):
    record(data_dir, ("user: a", "refund"), schema=JEV_CANONICAL)
    assert main(["report", "s1", "--gold", gold]) == 0
    assert capsys.readouterr().out.splitlines()[3] == (
        "served accuracy 1.000 at threshold 0.80: 67% answered by the head,"
        " the rest by the zeroshot fallback"
    )
    assert decider.questions == [{"s1": JEV_QUESTION}] * 3


def test_report_gold_in_local_mode_scores_the_head_fallback(
    data_dir, trained, decider, gold, capsys
):
    record(data_dir, ("user: a", "refund"), schema=JEV_CANONICAL)
    (data_dir / "stuntd.toml").write_text('[serving]\nlocal_fallback = "head"\n', encoding="utf-8")
    assert main(["report", "s1", "--gold", gold]) == 0
    assert capsys.readouterr().out.splitlines()[3] == (
        "served accuracy 0.667 at threshold 0.80: 67% answered by the head,"
        " the rest by the head fallback"
    )
    assert decider.questions == []


def test_report_gold_in_local_mode_sends_a_novel_request_to_the_fallback(
    data_dir, decider, gold, capsys
):
    save_model(data_dir / "models", replace(site_model(), novelty_cutoff=0.3))
    decider.novelty = {"user: a": 0.5}
    record(data_dir, ("user: a", "refund"), schema=JEV_CANONICAL)
    assert main(["report", "s1", "--gold", gold, "--json"]) == 0
    score = json.loads(capsys.readouterr().out)
    assert score["local_share"] == pytest.approx(1 / 3)
    assert score["served_accuracy"] == pytest.approx(2 / 3)


def test_report_gold_with_a_jev_provider_scores_the_teacher(
    data_dir, trained, decider, gold, capsys
):
    record(data_dir, ("user: a", "refund"), ("user: b", "refund"), schema=JEV_CANONICAL)
    (data_dir / "stuntd.toml").write_text('[jev]\nupstream = "http://jev"\n', encoding="utf-8")
    assert main(["report", "s1", "--gold", gold]) == 0
    assert "answered locally" in capsys.readouterr().out
    assert decider.questions == []


def test_report_gold_json_gives_the_same_numbers(data_dir, trained, decider, gold, capsys):
    record(data_dir, ("user: a", "refund"), ("user: a", "deny"), ("user: b", "refund"))
    assert main(["report", "s1", "--gold", gold, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "site": "s1",
        "threshold": 0.8,
        "read": 4,
        "used": 3,
        "skipped": 1,
        "head_accuracy": pytest.approx(2 / 3),
        "unseen_rows": 1,
        "unseen_head_accuracy": 1.0,
        "fallback": None,
        "served_accuracy": 1.0,
        "local_share": pytest.approx(2 / 3),
        "unserved": 0,
        "teacher_rows": 2,
        "teacher_accuracy": 1.0,
        "both_right": 1,
        "both_wrong": 0,
        "head_only": 0,
        "teacher_only": 1,
    }


def test_report_gold_without_a_database_has_no_teacher(data_dir, trained, decider, gold, capsys):
    assert main(["report", "s1", "--gold", gold, "--json"]) == 0
    score = json.loads(capsys.readouterr().out)
    assert score["teacher_rows"] == 0 and score["unserved"] == 1
    assert not (data_dir / "captures.sqlite").exists()


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("{oops}", "line 2: invalid JSON"),
        ("[1]", "line 2: expected an object"),
        (row("", "deny"), "line 2: text must be a non-empty string"),
        (json.dumps({"text": "a", "answer": 1}), "line 2: answer must be a string"),
    ],
    ids=["broken-json", "not-an-object", "empty-text", "answer-not-string"],
)
def test_report_gold_names_the_invalid_line(
    data_dir, trained, decider, tmp_path, capsys, line, message
):
    path = write_gold(tmp_path, [row("user: a", "deny"), line])
    assert main(["report", "s1", "--gold", path]) == 1
    assert message in capsys.readouterr().err
    assert decider.asked == []


def test_report_gold_without_a_site_is_refused(data_dir, trained, gold, capsys):
    assert main(["report", "--gold", gold]) == 2
    assert "stuntd: report --gold needs the SITE to score" in capsys.readouterr().err


def test_report_gold_without_torch_explains_the_extra(data_dir, trained, gold, capsys, monkeypatch):
    def missing(settings):
        raise ImportError("No module named 'laya'")

    monkeypatch.setattr("stuntd.cli._make_decider", missing)
    assert main(["report", "s1", "--gold", gold]) == 2
    assert (
        'stuntd: report --gold needs the train extra: pip install "stuntd[train]"'
        in capsys.readouterr().err
    )


def test_report_gold_with_curve_is_refused(data_dir, trained, gold, capsys):
    assert main(["report", "s1", "--gold", gold, "--curve"]) == 2
    assert "stuntd: --curve and --gold do not combine" in capsys.readouterr().err


def test_report_gold_without_usable_rows_loads_no_decider(
    data_dir, trained, tmp_path, capsys, monkeypatch
):
    def missing(settings):
        raise ImportError("No module named 'laya'")

    monkeypatch.setattr("stuntd.cli._make_decider", missing)
    path = write_gold(tmp_path, [row("user: a", "maybe")])
    assert main(["report", "s1", "--gold", path]) == 0
    assert capsys.readouterr().out.startswith("site s1  gold rows 1: 0 used, 1 skipped")


def test_report_gold_without_a_model_is_refused(data_dir, gold, capsys):
    assert main(["report", "s1", "--gold", gold]) == 2
    assert "stuntd: no model for s1" in capsys.readouterr().err


def test_report_gold_missing_file_is_refused(data_dir, trained, tmp_path, capsys):
    assert main(["report", "s1", "--gold", str(tmp_path / "absent.jsonl")]) == 2
    assert "stuntd:" in capsys.readouterr().err
