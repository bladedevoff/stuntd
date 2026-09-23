import json
import os
import stat
import sys
from dataclasses import asdict, replace

import pytest

from stuntd.train.artifacts import (
    HEAD_FILE,
    META_FILE,
    SiteModel,
    list_models,
    load_model,
    save_model,
    site_dir,
    write_meta,
)
from stuntd.train.metrics import ClassStats, ConfidentError, Operating


def model(site="s1"):
    return SiteModel(
        site=site,
        kind="choice",
        field="verdict",
        labels=["allow", "block"],
        base_model="convaiinnovations/laya",
        temperature=1.5,
        threshold=0.62,
        target_agreement=0.99,
        trained_at=1_700_000_000.0,
        n_train=240,
        n_holdout=60,
        agreement=0.95,
        coverage=0.8,
        covered_agreement=0.99,
        ece=0.02,
        per_class={
            "allow": ClassStats(40, 0.975),
            "block": ClassStats(20, 0.9),
            "x": ClassStats(0, None),
        },
        confident_errors=[ConfidentError("user: hi", 1, 0, 0.97)],
        curve=[Operating(0.9, 0.5, 1.0), Operating(0.62, 0.8, 0.99)],
    )


def test_round_trip(tmp_path):
    folder = save_model(tmp_path / "models", model())
    assert folder == tmp_path / "models" / "s1" and (folder / META_FILE).exists()
    assert load_model(tmp_path / "models", "s1") == model()


def test_list_models_is_sorted_and_tolerates_missing_dir(tmp_path):
    assert list_models(tmp_path / "none") == []
    save_model(tmp_path / "models", model("b"))
    save_model(tmp_path / "models", model("a"))
    assert [m.site for m in list_models(tmp_path / "models")] == ["a", "b"]


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "models", "s1")


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "a/b", "a\\b", "c:evil", "...", "a" * 65],
    ids=["empty", "dot", "dotdot", "slash", "backslash", "drive", "dots", "too-long"],
)
def test_site_dir_rejects_path_tricks(tmp_path, name):
    with pytest.raises(ValueError):
        site_dir(tmp_path, name)


@pytest.mark.parametrize("name", ["s1.tmp", "s1.old", "a" * 64], ids=["tmp", "old", "longest"])
def test_site_dir_accepts_usable_names(tmp_path, name):
    assert site_dir(tmp_path, name) == tmp_path / name


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_artifacts_are_private(tmp_path):
    folder = save_model(tmp_path / "models", model())
    assert stat.S_IMODE(os.stat(folder).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(folder / META_FILE).st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_created_parents_are_private(tmp_path):
    save_model(tmp_path / "outer" / "models", model())
    assert stat.S_IMODE(os.stat(tmp_path / "outer").st_mode) == 0o700
    assert stat.S_IMODE(os.stat(tmp_path / "outer" / "models").st_mode) == 0o700


def test_head_file_name_is_stable():
    assert HEAD_FILE == "head.safetensors"


def test_meta_is_plain_json(tmp_path):
    folder = save_model(tmp_path / "models", model())
    raw = json.loads((folder / META_FILE).read_text(encoding="utf-8"))
    assert raw["labels"] == ["allow", "block"] and raw["per_class"]["x"]["agreement"] is None


def test_non_finite_metric_is_refused_before_anything_is_written(tmp_path):
    with pytest.raises(ValueError):
        save_model(tmp_path / "models", replace(model(), ece=float("nan")))
    assert not (tmp_path / "models").exists()


def refuse_replace(*args, **kwargs):
    raise OSError("replace refused")


def test_meta_survives_a_failed_replace(tmp_path, monkeypatch):
    folder = save_model(tmp_path / "models", model())
    monkeypatch.setattr(os, "replace", refuse_replace)
    with pytest.raises(OSError):
        save_model(tmp_path / "models", replace(model(), agreement=0.5))
    assert load_model(tmp_path / "models", "s1") == model()
    assert [path.name for path in folder.iterdir()] == [META_FILE]


def test_corrupt_meta_names_the_file(tmp_path):
    folder = save_model(tmp_path / "models", model())
    (folder / META_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError) as failure:
        load_model(tmp_path / "models", "s1")
    assert str(folder / META_FILE) in str(failure.value)


def test_write_meta_produces_a_loadable_folder(tmp_path):
    write_meta(tmp_path / "x", model("x"))
    raw = json.loads((tmp_path / "x" / META_FILE).read_text(encoding="utf-8"))
    assert raw == asdict(model("x"))
    assert load_model(tmp_path, "x") == model("x")
