import json

from stuntd.train.artifacts import SiteModel
from stuntd.train.metrics import ClassStats, ConfidentError, Operating
from stuntd.train.report import render, render_json


def model(threshold=0.62):
    return SiteModel(
        site="s1",
        kind="choice",
        field="verdict",
        labels=["allow", "block"],
        base_model="convaiinnovations/laya",
        temperature=1.5,
        threshold=threshold,
        target_agreement=0.99,
        trained_at=1_700_000_000.0,
        n_train=240,
        n_holdout=60,
        agreement=0.95,
        coverage=None if threshold is None else 0.8,
        covered_agreement=None if threshold is None else 0.99,
        ece=0.021,
        per_class={"allow": ClassStats(40, 0.975), "block": ClassStats(0, None)},
        confident_errors=[ConfidentError("user: " + "x" * 100, 1, 0, 0.97)],
        curve=[Operating(0.9, 0.5, 1.0), Operating(0.62, 0.8, 0.99)],
    )


def test_render_shows_the_headline_numbers():
    text = render(model())
    assert text.startswith("site s1  choice verdict  trained ")
    assert "examples: 240 train, 60 holdout" in text
    assert "holdout agreement 0.950  ece 0.021  temperature 1.50" in text
    assert "at threshold 0.62: coverage 0.80, agreement 0.990" in text
    assert "allow" in text and "block" in text and "-" in text
    assert "0.97  expected block, got allow: user: " in text
    assert "x" * 100 in text
    assert "threshold  coverage  agreement" not in text


def test_render_without_threshold_says_so():
    assert "threshold: none reaches 0.99" in render(model(threshold=None))


def test_render_curve_lists_every_point():
    text = render(model(), curve=True)
    assert "threshold  coverage  agreement" in text and "0.62" in text and "0.90" in text


def test_render_json_is_a_list():
    data = json.loads(render_json([model()]))
    assert data[0]["site"] == "s1" and data[0]["per_class"]["block"]["agreement"] is None
