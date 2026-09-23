import sqlite3
import time

import pytest

from stuntd.store.db import PRUNE_EVERY, Capture, Decision, Store
from stuntd.store.redact import Redactor


def make(site="s1", text="user: hi", answer="allow"):
    return Capture(
        site=site,
        schema_canonical="{}",
        kind="choice",
        input_text=text,
        answer=answer,
        model="gpt-x",
        latency_ms=120,
        prompt_tokens=10,
        completion_tokens=1,
    )


def decision(site="s1", mode="shadow", answer="true", confidence=0.9, agree=True):
    return Decision(site, mode, answer, confidence, agree, 12, 0.0)


def test_record_and_stats(tmp_path):
    store = Store(tmp_path / "c.sqlite", Redactor())
    store.record(make())
    store.record(make(answer="block"))
    store.record(make(site="s2"))
    stats = {row["site"]: row for row in store.stats()}
    assert stats["s1"]["count"] == 2 and stats["s2"]["count"] == 1
    assert stats["s1"]["kind"] == "choice"
    store.close()


def test_redaction_happens_before_write(tmp_path):
    store = Store(tmp_path / "c.sqlite", Redactor())
    store.record(make(text="user: contact bob@example.com"))
    store.close()
    reader = sqlite3.connect(tmp_path / "c.sqlite")
    try:
        row = reader.execute("select input_text from captures").fetchone()
    finally:
        reader.close()
    assert row[0] == "user: contact [email]"


def test_prune_by_rows(tmp_path):
    store = Store(tmp_path / "c.sqlite", Redactor(), max_rows=3)
    for i in range(5):
        store.record(make(text=f"user: {i}"))
    store._prune()
    store._conn.commit()
    store.close()
    reader = sqlite3.connect(tmp_path / "c.sqlite")
    try:
        rows = reader.execute("select input_text from captures order by id").fetchall()
    finally:
        reader.close()
    assert [r[0] for r in rows] == ["user: 2", "user: 3", "user: 4"]


def test_prune_by_age(tmp_path):
    store = Store(tmp_path / "c.sqlite", Redactor(), max_age_days=1)
    store.record(make(text="user: old"))
    store._conn.execute("update captures set created_at = ?", (time.time() - 3 * 86400,))
    store.record(make(text="user: new"))
    store._prune()
    store._conn.commit()
    store.close()
    reader = sqlite3.connect(tmp_path / "c.sqlite")
    try:
        rows = reader.execute("select input_text from captures").fetchall()
    finally:
        reader.close()
    assert [r[0] for r in rows] == ["user: new"]


def test_prune_runs_periodically(tmp_path):
    store = Store(tmp_path / "c.sqlite", Redactor(), max_rows=3)
    counts = []
    for i in range(PRUNE_EVERY + 1):
        store.record(make(text=f"user: {i}"))
        counts.append(store._conn.execute("select count(*) from captures").fetchone()[0])
    assert counts[PRUNE_EVERY - 1] == PRUNE_EVERY
    assert counts[PRUNE_EVERY] == 3
    store.close()


def test_answer_is_stored_verbatim(tmp_path):
    store = Store(tmp_path / "c.sqlite", Redactor())
    store.record(make(text="user: 123456789", answer="123456789"))
    store.close()
    reader = sqlite3.connect(tmp_path / "c.sqlite")
    try:
        row = reader.execute("select input_text, answer from captures").fetchone()
    finally:
        reader.close()
    assert row == ("user: [phone]", "123456789")


def test_examples_come_back_in_time_order(tmp_path):
    store = Store(tmp_path / "c.sqlite", Redactor())
    store.record(make(text="user: first", answer="allow"))
    store.record(make(text="user: second", answer="block"))
    store.record(make(site="s2", text="user: other"))
    examples = store.examples("s1")
    store.close()
    assert [e.input_text for e in examples] == ["user: first", "user: second"]
    assert [e.answer for e in examples] == ["allow", "block"]
    assert examples[0].created_at <= examples[1].created_at


def test_examples_of_unknown_site_are_empty(tmp_path):
    store = Store(tmp_path / "c.sqlite", Redactor())
    assert store.examples("nope") == []
    store.close()


def test_sites_summarise_each_site_once(tmp_path):
    store = Store(tmp_path / "c.sqlite", Redactor())
    store.record(make(site="b"))
    store.record(make(site="a"))
    store.record(make(site="a", answer="block"))
    sites = store.sites()
    store.close()
    assert [(s.site, s.count) for s in sites] == [("a", 2), ("b", 1)]
    assert sites[0].kind == "choice" and sites[0].schema_canonical == "{}"


def test_decisions_come_back_newest_first(tmp_path):
    store = Store(tmp_path / "c.sqlite", Redactor())
    store.record_decision(decision(confidence=0.5))
    store.record_decision(decision(confidence=0.7))
    store.record_decision(decision(site="s2"))
    rows = store.decisions("s1", limit=10)
    store.close()
    assert [r.confidence for r in rows] == [0.7, 0.5]
    assert rows[0].created_at >= rows[1].created_at


def test_decisions_limit_and_none_agree(tmp_path):
    store = Store(tmp_path / "c.sqlite", Redactor())
    for _ in range(5):
        store.record_decision(decision(mode="live", agree=None))
    rows = store.decisions("s1", limit=2)
    store.close()
    assert len(rows) == 2 and rows[0].agree is None and rows[0].mode == "live"


def test_decision_counts_group_by_site_and_mode(tmp_path):
    store = Store(tmp_path / "c.sqlite", Redactor())
    store.record_decision(decision())
    store.record_decision(decision(mode="check", agree=False))
    store.record_decision(decision(site="s2", mode="live", agree=None))
    counts = store.decision_counts()
    store.close()
    assert counts == {
        "s1": {"shadow": 1, "check": 1, "live": 0},
        "s2": {"shadow": 0, "check": 0, "live": 1},
    }


def test_decisions_are_pruned_with_captures(tmp_path):
    store = Store(tmp_path / "c.sqlite", Redactor(), max_rows=3)
    for _ in range(PRUNE_EVERY + 5):
        store.record_decision(decision())
    store._prune()
    assert len(store.decisions("s1", limit=100)) == 3
    store.close()


def test_comparisons_skip_the_answers_nothing_compared(tmp_path):
    store = Store(tmp_path / "c.sqlite", Redactor())
    store.record_decision(decision(mode="shadow"))
    for _ in range(5):
        store.record_decision(decision(mode="live", agree=None))
    store.record_decision(decision(mode="check", agree=False))
    rows = store.comparisons("s1", limit=10, since=0.0)
    store.close()
    assert [(row.mode, row.agree) for row in rows] == [("check", False), ("shadow", True)]


def test_comparisons_skip_the_rows_recorded_before_since(tmp_path):
    store = Store(tmp_path / "c.sqlite", Redactor())
    store.record_decision(decision(mode="shadow"))
    store._conn.execute("update decisions set created_at = ?", (100.0,))
    store.record_decision(decision(mode="check", agree=False))
    rows = store.comparisons("s1", limit=10, since=200.0)
    store.close()
    assert [(row.mode, row.agree) for row in rows] == [("check", False)]


def test_comparisons_count_the_limit_in_comparisons(tmp_path):
    store = Store(tmp_path / "c.sqlite", Redactor())
    for index in range(3):
        store.record_decision(decision(mode="shadow", confidence=0.1 * index))
        for _ in range(4):
            store.record_decision(decision(mode="live", agree=None))
    rows = store.comparisons("s1", limit=2, since=0.0)
    store.close()
    assert [row.confidence for row in rows] == [0.2, 0.1]


def test_comparisons_reject_zero_limit(tmp_path):
    store = Store(tmp_path / "c.sqlite", Redactor())
    with pytest.raises(ValueError):
        store.comparisons("s1", limit=0, since=0.0)
    store.close()


def test_decisions_reject_zero_limit_and_read_back_unknown_site_and_false_agree(tmp_path):
    store = Store(tmp_path / "c.sqlite", Redactor())
    store.record_decision(decision(mode="check", agree=False))
    with pytest.raises(ValueError):
        store.decisions("s1", limit=0)
    assert store.decisions("unknown", 5) == []
    assert store.decisions("s1", 5)[0].agree is False
    store.close()
