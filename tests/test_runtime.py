from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import pytest

from stuntd.serve.modes import (
    MODE_COLLECT,
    MODE_FILE,
    MODE_LIVE,
    MODE_SHADOW,
    read_mode,
    write_mode,
)
from stuntd.serve.runtime import Outcome, Runtime
from stuntd.settings import Settings
from stuntd.store.db import Decision, Store
from stuntd.store.redact import Redactor
from stuntd.train.artifacts import META_FILE, SiteModel, save_model
from stuntd.train.metrics import ClassStats

NOW = 1000.0


@dataclass(frozen=True)
class FakeVerdict:
    label: int
    confidence: float
    latency_ms: int
    probabilities: tuple[float, ...] = ()


SURE = FakeVerdict(1, 0.9, 7)


class FakeDecider:
    def __init__(self, verdict=SURE, error=None):
        self.verdict = verdict
        self.error = error
        self.calls = []

    def decide(self, model, head_path, text):
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return self.verdict


def model(site="s1", threshold=0.6, covered_agreement=1.0, trained_at=100.0):
    return SiteModel(
        site=site,
        kind="boolean",
        field="spam",
        labels=["false", "true"],
        base_model="b",
        temperature=1.0,
        threshold=threshold,
        target_agreement=0.9,
        trained_at=trained_at,
        n_train=8,
        n_holdout=2,
        agreement=1.0,
        coverage=1.0,
        covered_agreement=covered_agreement,
        ece=0.0,
        per_class={"false": ClassStats(1, 1.0), "true": ClassStats(1, 1.0)},
        confident_errors=[],
        curve=[],
    )


@pytest.fixture
def models(tmp_path):
    return tmp_path / "models"


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "decisions.sqlite", Redactor())
    yield store
    store.close()


def settings_for(models, **overrides):
    values = {
        "models_dir": models,
        "check_share": 0.0,
        "window": 10,
        "min_window": 2,
        "target_agreement": 0.9,
        "cache_size": 10,
    }
    return Settings(**{**values, **overrides})


def runtime_for(models, store, decider=None, **overrides):
    return Runtime(settings_for(models, **overrides), store, decider, now=lambda: NOW)


def decided(site="s1", mode="shadow", agree=True, confidence=0.9):
    return Decision(site, mode, "true", confidence, agree, 5, 0.0)


def test_state_without_a_model_is_collect(models, store):
    state = runtime_for(models, store).state("s1")
    assert state.mode == MODE_COLLECT and state.model is None


def test_state_with_a_model_and_no_mode_file_is_shadow(models, store):
    save_model(models, model())
    state = runtime_for(models, store).state("s1")
    assert state.mode == MODE_SHADOW and state.model is not None


def test_state_follows_the_mode_file_after_it_changes(models, store):
    folder = save_model(models, model())
    runtime = runtime_for(models, store)
    assert runtime.state("s1").mode == MODE_SHADOW
    write_mode(folder, MODE_LIVE, now=NOW)
    stamp = (folder / META_FILE).stat().st_mtime_ns + 1_000_000_000
    os.utime(folder / META_FILE, ns=(stamp, stamp))
    assert runtime.state("s1").mode == MODE_LIVE


def test_state_falls_back_to_collect_once_a_file_is_torn(models, store, caplog):
    folder = save_model(models, model())
    write_mode(folder, MODE_LIVE, now=1.0)
    (folder / MODE_FILE).write_bytes(b"")
    runtime = runtime_for(models, store)
    with caplog.at_level(logging.ERROR):
        torn = [runtime.state("s1") for _ in range(2)]
    assert [(state.mode, state.model) for state in torn] == [(MODE_COLLECT, None)] * 2
    assert runtime.state_reason("s1") == "state-error"
    assert caplog.text.count("s1 state unreadable") == 1
    write_mode(folder, MODE_LIVE, now=2.0)
    assert runtime.state("s1").mode == MODE_LIVE
    assert runtime.state_reason("s1") is None


def test_state_is_cached_while_the_files_hold_still(models, store):
    save_model(models, model())
    runtime = runtime_for(models, store)
    assert runtime.state("s1") is runtime.state("s1")


def test_state_notices_a_rewrite_the_clock_did_not(models, store):
    folder = save_model(models, model())
    write_mode(folder, MODE_LIVE, now=1.0)
    before = (folder / MODE_FILE).stat()
    runtime = runtime_for(models, store)
    assert runtime.state("s1").mode == MODE_LIVE
    write_mode(folder, MODE_SHADOW, now=1.0)
    os.utime(folder / MODE_FILE, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert runtime.state("s1").mode == MODE_SHADOW


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("big_answer", "agree"), [("true", True), ("false", False)], ids=["agrees", "differs"]
)
async def test_shadow_records_how_the_model_compared(models, store, big_answer, agree):
    save_model(models, model())
    decider = FakeDecider()
    runtime = runtime_for(models, store, decider)
    await runtime.shadow(runtime.state("s1"), "user: hi", big_answer)
    rows = store.decisions("s1", 10)
    assert decider.calls == ["user: hi"]
    assert [(row.mode, row.answer, row.confidence, row.agree, row.latency_ms) for row in rows] == [
        ("shadow", "true", 0.9, agree, 7)
    ]


@pytest.mark.anyio
async def test_shadow_logs_a_model_error_instead_of_raising(models, store, caplog):
    save_model(models, model())
    runtime = runtime_for(models, store, FakeDecider(error=RuntimeError("boom")))
    with caplog.at_level(logging.ERROR):
        await runtime.shadow(runtime.state("s1"), "user: hi", "true")
    assert store.decisions("s1", 10) == []
    assert "shadow decision failed" in caplog.text


@pytest.mark.anyio
async def test_shadow_without_a_decider_records_nothing(models, store):
    save_model(models, model())
    runtime = runtime_for(models, store)
    await runtime.shadow(runtime.state("s1"), "user: hi", "true")
    assert store.decisions("s1", 10) == []


@pytest.mark.anyio
async def test_the_head_is_asked_the_redacted_text(models, store):
    save_model(models, model())
    decider = FakeDecider()
    runtime = runtime_for(models, store, decider)
    state = runtime.state("s1")
    outcome = await runtime.live(state, "user: write to ann@example.com")
    assert decider.calls == ["user: write to [email]"]
    assert outcome.mode == MODE_LIVE
    assert store.decisions("s1", 10) == []


@pytest.mark.anyio
async def test_live_answers_when_the_model_is_confident(models, store):
    save_model(models, model())
    runtime = runtime_for(models, store, FakeDecider())
    state = runtime.state("s1")
    outcome = await runtime.live(state, "user: hi")
    assert outcome == Outcome(MODE_LIVE, None, "true", 0.9, 7, ())
    runtime.record_live(state, outcome)
    rows = store.decisions("s1", 10)
    assert [(row.mode, row.answer, row.agree, row.latency_ms) for row in rows] == [
        ("live", "true", None, 7)
    ]


@pytest.mark.anyio
async def test_live_collects_when_the_model_is_unsure(models, store):
    save_model(models, model())
    runtime = runtime_for(models, store, FakeDecider(FakeVerdict(0, 0.4, 7)))
    outcome = await runtime.live(runtime.state("s1"), "user: hi")
    assert outcome == Outcome(MODE_COLLECT, "low-confidence", "false", 0.4, 7, ())


@pytest.mark.anyio
async def test_record_live_ignores_an_answer_the_provider_gave(models, store):
    save_model(models, model())
    runtime = runtime_for(models, store, FakeDecider(FakeVerdict(0, 0.4, 7)))
    state = runtime.state("s1")
    runtime.record_live(state, await runtime.live(state, "user: hi"))
    assert store.decisions("s1", 10) == []


def test_record_live_ignores_an_outcome_that_was_never_timed(models, store):
    save_model(models, model())
    runtime = runtime_for(models, store)
    runtime.record_live(runtime.state("s1"), Outcome(MODE_LIVE, None, "true", 0.9, None))
    assert store.decisions("s1", 10) == []


@pytest.mark.anyio
async def test_live_collects_the_sampled_check(models, store):
    save_model(models, model())
    runtime = runtime_for(models, store, FakeDecider(), check_share=0.9999)
    state = runtime.state("s1")
    outcome = await runtime.live(state, "user: hi")
    assert outcome == Outcome(MODE_COLLECT, "check", "true", 0.9, 7, ())
    runtime.compare(state, outcome, "false")
    rows = store.decisions("s1", 10)
    assert [(row.mode, row.answer, row.confidence, row.agree, row.latency_ms) for row in rows] == [
        ("check", "true", 0.9, False, 7)
    ]


@pytest.mark.anyio
async def test_live_collects_without_a_decider(models, store):
    save_model(models, model())
    runtime = runtime_for(models, store)
    outcome = await runtime.live(runtime.state("s1"), "user: hi")
    assert outcome == Outcome(MODE_COLLECT, "no-runtime", None, None, None)


@pytest.mark.anyio
async def test_live_collects_after_a_model_error(models, store, caplog):
    save_model(models, model())
    runtime = runtime_for(models, store, FakeDecider(error=RuntimeError("boom")))
    with caplog.at_level(logging.ERROR):
        outcome = await runtime.live(runtime.state("s1"), "user: hi")
    assert outcome == Outcome(MODE_COLLECT, "model-error", None, None, None)
    assert "live decision failed" in caplog.text


@pytest.mark.anyio
async def test_live_collects_when_the_model_has_no_threshold(models, store):
    save_model(models, model(threshold=None))
    decider = FakeDecider()
    runtime = runtime_for(models, store, decider)
    outcome = await runtime.live(runtime.state("s1"), "user: hi")
    assert outcome == Outcome(MODE_COLLECT, "low-confidence", None, None, None)
    assert decider.calls == []


@pytest.mark.anyio
async def test_live_answers_a_repeated_request_from_the_cache(models, store):
    save_model(models, model())
    decider = FakeDecider()
    runtime = runtime_for(models, store, decider)
    state = runtime.state("s1")
    outcomes = []
    for _ in range(2):
        outcome = await runtime.live(state, "user: hi")
        outcomes.append(outcome)
        runtime.record_live(state, outcome)
    assert outcomes == [
        Outcome(MODE_LIVE, None, "true", 0.9, 7, ()),
        Outcome(MODE_LIVE, None, "true", 0.9, 0, ()),
    ]
    assert decider.calls == ["user: hi"]
    assert [(row.mode, row.latency_ms) for row in store.decisions("s1", 10)] == [
        ("live", 0),
        ("live", 7),
    ]


@pytest.mark.anyio
async def test_live_decides_every_time_when_the_cache_is_off(models, store):
    save_model(models, model())
    decider = FakeDecider()
    runtime = runtime_for(models, store, decider, cache_size=0)
    state = runtime.state("s1")
    for _ in range(2):
        assert await runtime.live(state, "user: hi") == Outcome(MODE_LIVE, None, "true", 0.9, 7, ())
    assert decider.calls == ["user: hi", "user: hi"]


@pytest.mark.anyio
async def test_live_decides_again_once_the_site_is_retrained(models, store):
    save_model(models, model())
    decider = FakeDecider()
    runtime = runtime_for(models, store, decider)
    await runtime.live(runtime.state("s1"), "user: hi")
    folder = save_model(models, model(trained_at=200.0))
    stamp = (folder / META_FILE).stat().st_mtime_ns + 1_000_000_000
    os.utime(folder / META_FILE, ns=(stamp, stamp))
    await runtime.live(runtime.state("s1"), "user: hi")
    assert decider.calls == ["user: hi", "user: hi"]


class RetrainingDecider:
    def __init__(self, models):
        self.models = models

    def decide(self, site_model, head_path, text):
        folder = save_model(self.models, model(trained_at=200.0))
        stamp = (folder / META_FILE).stat().st_mtime_ns + 1_000_000_000
        os.utime(folder / META_FILE, ns=(stamp, stamp))
        return SURE


@pytest.mark.anyio
async def test_live_collects_an_answer_the_retraining_overtook(models, store):
    save_model(models, model())
    runtime = runtime_for(models, store, RetrainingDecider(models))
    state = runtime.state("s1")
    outcome = await runtime.live(state, "user: hi")
    assert outcome == Outcome(MODE_COLLECT, "retrained", None, None, None)
    assert store.decisions("s1", 10) == []


@pytest.mark.anyio
async def test_live_evicts_the_oldest_cached_answer(models, store):
    save_model(models, model())
    decider = FakeDecider()
    runtime = runtime_for(models, store, decider, cache_size=1)
    state = runtime.state("s1")
    for text in ("first", "second", "first"):
        await runtime.live(state, text)
    assert decider.calls == ["first", "second", "first"]


@pytest.mark.anyio
async def test_live_evicts_the_answer_it_has_reused_least_recently(models, store):
    save_model(models, model())
    decider = FakeDecider()
    runtime = runtime_for(models, store, decider, cache_size=2)
    state = runtime.state("s1")
    for text in ("first", "second", "first", "third", "first"):
        await runtime.live(state, text)
    assert decider.calls == ["first", "second", "third"]


@pytest.mark.parametrize("disagreements", [2, 1], ids=["full-window", "short-window"])
def test_demote_needs_a_losing_window(models, store, disagreements):
    folder = save_model(models, model())
    write_mode(folder, MODE_LIVE, now=1.0)
    for _ in range(disagreements):
        store.record_decision(decided(agree=False))
    runtime = runtime_for(models, store)
    demoted = runtime.demote_if_needed(runtime.state("s1"))
    assert demoted is (disagreements == 2)
    assert read_mode(folder) == ((MODE_SHADOW, NOW) if demoted else (MODE_LIVE, 1.0))


def test_demote_reads_past_the_answers_nothing_compared(models, store):
    folder = save_model(models, model())
    write_mode(folder, MODE_LIVE, now=1.0)
    for _ in range(2):
        store.record_decision(decided(agree=False))
    for _ in range(20):
        store.record_decision(decided(mode="live", agree=None))
    runtime = runtime_for(models, store, window=3)
    assert runtime.demote_if_needed(runtime.state("s1")) is True
    assert read_mode(folder) == (MODE_SHADOW, NOW)


def test_demote_takes_effect_on_a_clock_too_coarse_to_notice(models, store):
    folder = save_model(models, model())
    write_mode(folder, MODE_LIVE, now=1.0)
    before = (folder / MODE_FILE).stat()
    runtime = runtime_for(models, store)
    assert runtime.state("s1").mode == MODE_LIVE
    for _ in range(2):
        store.record_decision(decided(agree=False))
    assert runtime.demote_if_needed(runtime.state("s1")) is True
    os.utime(folder / MODE_FILE, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert runtime.state("s1").mode == MODE_SHADOW


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("check", "mode", "reason"),
    [(True, MODE_COLLECT, "check"), (False, MODE_LIVE, None)],
    ids=["sampled", "not-sampled"],
)
async def test_live_samples_a_check_only_when_it_is_asked_to(models, store, check, mode, reason):
    save_model(models, model())
    runtime = runtime_for(models, store, FakeDecider(), check_share=0.9999)
    outcome = await runtime.live(runtime.state("s1"), "user: hi", check)
    assert (outcome.mode, outcome.reason, outcome.answer) == (mode, reason, "true")


def backdate(store, created_at):
    store._conn.execute("update decisions set created_at = ?", (created_at,))


def test_demote_ignores_the_comparisons_of_an_earlier_mode(models, store):
    folder = save_model(models, model(trained_at=0.0))
    for _ in range(100):
        store.record_decision(decided(agree=False))
    backdate(store, NOW - 1)
    write_mode(folder, MODE_LIVE, now=NOW)
    runtime = runtime_for(models, store)
    state = runtime.state("s1")
    assert runtime.demote_if_needed(state) is False
    for _ in range(2):
        store.record_decision(decided(mode="check", agree=False))
    assert runtime.demote_if_needed(state) is True
    assert read_mode(folder) == (MODE_SHADOW, NOW)


def test_demote_ignores_the_comparisons_of_an_earlier_head(models, store):
    folder = save_model(models, model(trained_at=NOW))
    write_mode(folder, MODE_LIVE, now=1.0)
    for _ in range(100):
        store.record_decision(decided(agree=False))
    backdate(store, NOW - 1)
    runtime = runtime_for(models, store)
    assert runtime.demote_if_needed(runtime.state("s1")) is False
    assert read_mode(folder) == (MODE_LIVE, 1.0)


def test_promote_is_skipped_when_auto_promote_is_off(models, store, monkeypatch):
    save_model(models, model())
    reads = []
    monkeypatch.setattr(store, "comparisons", lambda *args: reads.append(args) or [])
    runtime = runtime_for(models, store)
    assert runtime.promote_if_allowed(runtime.state("s1")) is False
    assert reads == []


def test_promote_sends_a_steady_site_live(models, store):
    folder = save_model(models, model())
    for _ in range(2):
        store.record_decision(decided())
    runtime = runtime_for(models, store, auto_promote=True, auto_promote_after_hours=0)
    assert runtime.promote_if_allowed(runtime.state("s1")) is True
    assert read_mode(folder) == (MODE_LIVE, NOW)


@pytest.mark.anyio
async def test_compare_logs_a_store_failure_instead_of_raising(models, store, caplog):
    save_model(models, model())
    runtime = runtime_for(models, store, FakeDecider(), check_share=0.9999)
    state = runtime.state("s1")
    outcome = await runtime.live(state, "user: hi")
    store.close()
    with caplog.at_level(logging.ERROR):
        runtime.compare(state, outcome, "false")
    assert "check decision not recorded" in caplog.text


def test_record_live_logs_a_store_failure_instead_of_raising(models, store, caplog):
    save_model(models, model())
    runtime = runtime_for(models, store)
    store.close()
    with caplog.at_level(logging.ERROR):
        runtime.record_live(runtime.state("s1"), Outcome(MODE_LIVE, None, "true", 0.9, 7))
    assert "live decision not recorded" in caplog.text


def test_demote_logs_a_store_failure_instead_of_raising(models, store, caplog):
    folder = save_model(models, model())
    write_mode(folder, MODE_LIVE, now=1.0)
    runtime = runtime_for(models, store)
    state = runtime.state("s1")
    store.close()
    with caplog.at_level(logging.ERROR):
        assert runtime.demote_if_needed(state) is False
    assert "demotion check failed" in caplog.text
    assert read_mode(folder) == (MODE_LIVE, 1.0)


def refuse_write(*args, **kwargs):
    raise OSError("models directory is read-only")


@pytest.mark.anyio
async def test_compare_records_when_the_mode_file_cannot_be_rewritten(
    models, store, monkeypatch, caplog
):
    folder = save_model(models, model())
    write_mode(folder, MODE_LIVE, now=1.0)
    store.record_decision(decided(mode="check", agree=False))
    runtime = runtime_for(models, store, FakeDecider(), check_share=0.9999)
    state = runtime.state("s1")
    outcome = await runtime.live(state, "user: hi")
    monkeypatch.setattr("stuntd.serve.runtime.write_mode", refuse_write)
    with caplog.at_level(logging.ERROR):
        runtime.compare(state, outcome, "false")
        assert runtime.demote_if_needed(state) is False
    assert [(row.mode, row.agree) for row in store.decisions("s1", 10)] == [
        ("check", False),
        ("check", False),
    ]
    assert "demotion check failed" in caplog.text
    assert read_mode(folder) == (MODE_LIVE, 1.0)


def test_promote_logs_a_file_failure_instead_of_raising(models, store, monkeypatch, caplog):
    folder = save_model(models, model())
    for _ in range(2):
        store.record_decision(decided())
    runtime = runtime_for(models, store, auto_promote=True, auto_promote_after_hours=0)
    monkeypatch.setattr("stuntd.serve.runtime.write_mode", refuse_write)
    with caplog.at_level(logging.ERROR):
        assert runtime.promote_if_allowed(runtime.state("s1")) is False
    assert "promotion check failed" in caplog.text
    assert read_mode(folder) == (MODE_SHADOW, None)
