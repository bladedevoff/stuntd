import json
import os
import re

import pytest

from stuntd.serve.modes import (
    MODE_COLLECT,
    MODE_LIVE,
    MODE_SHADOW,
    read_mode,
    site_state,
    site_states,
    write_mode,
)
from stuntd.train.artifacts import SiteModel, save_model
from stuntd.train.metrics import ClassStats


def model(site="s1"):
    return SiteModel(
        site=site,
        kind="boolean",
        field="spam",
        labels=["false", "true"],
        base_model="b",
        temperature=1.0,
        threshold=0.6,
        target_agreement=0.99,
        trained_at=100.0,
        n_train=8,
        n_holdout=2,
        agreement=1.0,
        coverage=1.0,
        covered_agreement=1.0,
        ece=0.0,
        per_class={"false": ClassStats(1, 1.0), "true": ClassStats(1, 1.0)},
        confident_errors=[],
        curve=[],
    )


def test_mode_defaults_to_shadow_when_the_file_is_missing(tmp_path):
    folder = save_model(tmp_path, model())
    assert read_mode(folder) == (MODE_SHADOW, None)


def test_mode_round_trip(tmp_path):
    folder = save_model(tmp_path, model())
    write_mode(folder, MODE_LIVE, now=42.0)
    assert read_mode(folder) == (MODE_LIVE, 42.0)
    assert json.loads((folder / "mode.json").read_text(encoding="utf-8")) == {
        "mode": "live",
        "changed_at": 42.0,
    }


def refuse_replace(*args, **kwargs):
    raise OSError("replace refused")


def test_mode_survives_a_failed_replace(tmp_path, monkeypatch):
    folder = save_model(tmp_path, model())
    write_mode(folder, MODE_LIVE, now=42.0)
    monkeypatch.setattr(os, "replace", refuse_replace)
    with pytest.raises(OSError):
        write_mode(folder, MODE_SHADOW, now=43.0)
    assert read_mode(folder) == (MODE_LIVE, 42.0)
    assert sorted(path.name for path in folder.iterdir()) == ["meta.json", "mode.json"]


def test_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        write_mode(tmp_path, "turbo", now=1.0)


def test_site_states_cover_every_model(tmp_path):
    save_model(tmp_path, model("a"))
    write_mode(tmp_path / "b", MODE_LIVE, now=1.0)
    save_model(tmp_path, model("b"))
    states = {s.site: s for s in site_states(tmp_path)}
    assert states["a"].mode == MODE_SHADOW and states["a"].model is not None
    assert states["b"].mode == MODE_LIVE and states["b"].changed_at == 1.0


def test_site_without_model_is_collecting(tmp_path):
    state = site_state(tmp_path, "nope")
    assert state.mode == MODE_COLLECT and state.model is None


@pytest.mark.parametrize(
    "contents",
    [
        '{"mode": "live"',
        '{"mode": "turbo", "changed_at": 1.0}',
        '{"mode": "live", "changed_at": "soon"}',
        '{"mode": "live", "changed_at": true}',
    ],
    ids=["not-json", "unknown-mode", "changed-at-not-a-number", "changed-at-a-bool"],
)
def test_corrupt_mode_names_the_file(tmp_path, contents):
    folder = save_model(tmp_path, model())
    path = folder / "mode.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(str(path))):
        read_mode(folder)


def test_site_states_propagates_a_corrupt_mode(tmp_path):
    folder = save_model(tmp_path, model("a"))
    (folder / "mode.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(str(folder / "mode.json"))):
        site_states(tmp_path)
