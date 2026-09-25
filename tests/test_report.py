import json

from stuntd.train.artifacts import SiteModel
from stuntd.train.metrics import ClassStats, ConfidentError, Operating
from stuntd.train.report import render, render_json, thin_curve


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


def long_curve(size=1161):
    return [
        Operating(1.0 - answered / size, answered / size, 1.0 - answered / (10 * size))
        for answered in range(1, size + 1)
    ]


def test_render_curve_lists_a_short_curve_whole():
    text = render(model(), curve=True)
    assert "threshold  coverage  agreement" in text and "0.62" in text and "0.90" in text


def test_render_curve_thins_a_long_curve():
    site = model()
    site.curve = long_curve()
    site.threshold = site.curve[720].threshold
    lines = render(site, curve=True).splitlines()
    rows = lines[lines.index("threshold  coverage  agreement") + 1 :]
    assert len(rows) == 21
    assert rows[0] == "     0.95      0.05      0.995"
    assert rows[12] == "     0.38      0.62      0.938"
    assert rows[-1] == "     0.00      1.00      0.900"


def test_thin_curve_keeps_every_coverage_step_and_the_operating_point():
    curve = long_curve()
    operating = curve[720]
    thinned = thin_curve(curve, operating.threshold)
    assert len(thinned) == 21 and operating in thinned
    assert thinned == sorted(set(thinned), key=curve.index)
    assert thinned[-1] == curve[-1]
    assert [point.coverage for point in thinned if point != operating] == [
        min(point.coverage for point in curve if point.coverage >= step / 20)
        for step in range(1, 21)
    ]


def test_thin_curve_does_not_repeat_the_operating_point_on_a_step():
    curve = long_curve(100)
    assert len(thin_curve(curve, curve[49].threshold)) == 20


def test_thin_curve_keeps_a_curve_shorter_than_the_steps():
    curve = [Operating(0.9, 0.5, 1.0), Operating(0.62, 0.8, 0.99), Operating(0.1, 1.0, 0.9)]
    assert thin_curve(curve, None) == curve


def test_thin_curve_of_nothing_is_empty():
    assert thin_curve([], None) == []


def test_render_json_is_a_list():
    data = json.loads(render_json([model()]))
    assert data[0]["site"] == "s1" and data[0]["per_class"]["block"]["agreement"] is None
