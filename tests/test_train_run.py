import json
from pathlib import Path

import pytest

from stuntd.settings import Settings, models_path
from stuntd.store.db import Capture, Store
from stuntd.store.redact import Redactor
from stuntd.train.artifacts import HEAD_FILE, META_FILE, load_model
from stuntd.train.dataset import NotTrainable
from stuntd.train.run import train_sites

SCHEMA = '{"properties":{"spam":{"type":"boolean"}},"type":"object"}'


def capture(site, text, answer):
    return Capture(site, SCHEMA, "boolean", text, answer, "gpt-x", 10, 5, 1)


@pytest.fixture
def store(data_dir):
    data_dir.mkdir(parents=True)
    store = Store(data_dir / "c.sqlite", Redactor())
    for i in range(10):
        store.record(capture("s1", f"user: message {i}", "true" if i % 2 else "false"))
    store.record(capture("thin", "user: only", "true"))
    yield store
    store.close()


def settings():
    return Settings(min_examples=4, holdout=0.3, target_agreement=0.9)


def perfect_trainer(dataset, head_path):
    head_path.write_bytes(b"head")
    return [[3.0, 0.0] if item.label == 0 else [0.0, 3.0] for item in dataset.holdout]


def failing_trainer(dataset, head_path):
    raise RuntimeError("training failed: boom")


def refusing_trainer(dataset, head_path):
    raise NotTrainable("labels do not fit")


def test_trained_site_gets_a_model_and_the_thin_one_a_reason(store, data_dir):
    results = {r.site: r for r in train_sites(store, settings(), perfect_trainer)}
    model = results["s1"].model
    assert model is not None and model.labels == ["false", "true"]
    assert model.agreement == 1.0 and model.coverage == 1.0 and model.threshold is not None
    assert model.n_train == 7 and model.n_holdout == 3
    assert (models_path(settings()) / "s1" / HEAD_FILE).read_bytes() == b"head"
    assert load_model(models_path(settings()), "s1") == model
    assert (
        results["thin"].model is None and "examples after deduplication" in results["thin"].reason
    )


def test_only_named_sites_are_trained(store):
    results = train_sites(store, settings(), perfect_trainer, sites=["s1", "ghost"])
    assert [(r.site, r.model is not None, r.reason) for r in results] == [
        ("s1", True, None),
        ("ghost", False, "no captures for this site"),
    ]


def test_failed_training_keeps_the_previous_model(store):
    train_sites(store, settings(), perfect_trainer, sites=["s1"])
    before = (models_path(settings()) / "s1" / META_FILE).read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="boom"):
        train_sites(store, settings(), failing_trainer, sites=["s1"])
    folder = models_path(settings())
    work = folder.with_name(folder.name + ".work")
    assert (folder / "s1" / META_FILE).read_text(encoding="utf-8") == before
    assert not (work / "s1").exists() and not (work / "s1.old").exists()


def test_retraining_replaces_the_model_in_place(store):
    train_sites(store, settings(), perfect_trainer, sites=["s1"])
    results = train_sites(store, settings(), perfect_trainer, sites=["s1"], now=lambda: 42.0)
    assert results[0].model is not None and results[0].model.trained_at == 42.0
    raw = json.loads((models_path(settings()) / "s1" / META_FILE).read_text(encoding="utf-8"))
    assert raw["trained_at"] == 42.0
    assert sorted(p.name for p in models_path(settings()).iterdir()) == ["s1"]


def test_a_site_named_like_a_parent_directory_is_skipped(store):
    store.record(capture("..", "user: message 0", "true"))
    results = {r.site: r for r in train_sites(store, settings(), perfect_trainer)}
    assert results[".."].model is None and "invalid site name" in results[".."].reason
    assert results["s1"].model is not None


def test_a_failing_rollback_reports_the_original_error(store, monkeypatch):
    train_sites(store, settings(), perfect_trainer, sites=["s1"])
    rename, renamed = Path.rename, []

    def once(self, target):
        renamed.append(self.name)
        if len(renamed) > 1:
            raise OSError("rename blocked")
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", once)
    with pytest.raises(OSError, match="rename blocked"):
        train_sites(store, settings(), perfect_trainer, sites=["s1"])
    work = models_path(settings()).with_name("models.work")
    assert renamed == ["s1", "s1", "s1.old"]
    assert (work / "s1.old" / META_FILE).is_file() and not (work / "s1").exists()


def test_temperature_and_curve_are_recorded(store):
    result = train_sites(store, settings(), perfect_trainer, sites=["s1"])[0]
    assert result.model is not None
    assert result.model.temperature > 0 and len(result.model.curve) >= 1
    assert result.model.ece >= 0.0 and set(result.model.per_class) == {"false", "true"}


def test_a_trainer_refusing_the_labels_reports_the_reason(store):
    for i in range(10):
        store.record(capture("s2", f"user: note {i}", "true" if i % 2 else "false"))
    results = {r.site: r for r in train_sites(store, settings(), refusing_trainer)}
    assert [r.reason for r in (results["s1"], results["s2"])] == ["labels do not fit"] * 2
    assert results["s1"].model is None and results["s2"].model is None
    work = models_path(settings()).with_name("models.work")
    assert not (work / "s1").exists() and not (work / "s2").exists()


def test_an_interrupted_publish_keeps_the_previous_model(store, monkeypatch):
    train_sites(store, settings(), perfect_trainer, sites=["s1"])
    before = (models_path(settings()) / "s1" / META_FILE).read_text(encoding="utf-8")
    rename, renamed = Path.rename, []

    def interrupt_the_swap(self, target):
        renamed.append(self.name)
        if len(renamed) == 2:
            raise KeyboardInterrupt
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", interrupt_the_swap)
    with pytest.raises(KeyboardInterrupt):
        train_sites(store, settings(), perfect_trainer, sites=["s1"])
    assert (models_path(settings()) / "s1" / META_FILE).read_text(encoding="utf-8") == before


def test_publish_starts_the_model_in_shadow(store):
    train_sites(store, settings(), perfect_trainer, sites=["s1"], now=lambda: 7.0)
    folder = models_path(settings()) / "s1"
    assert json.loads((folder / "mode.json").read_text(encoding="utf-8")) == {
        "mode": "shadow",
        "changed_at": 7.0,
    }


def test_retraining_resets_live_to_shadow(store):
    train_sites(store, settings(), perfect_trainer, sites=["s1"])
    from stuntd.serve.modes import MODE_LIVE, read_mode, write_mode

    write_mode(models_path(settings()) / "s1", MODE_LIVE, now=1.0)
    train_sites(store, settings(), perfect_trainer, sites=["s1"])
    assert read_mode(models_path(settings()) / "s1")[0] == "shadow"
