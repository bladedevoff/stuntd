import json

from stuntd.train.artifacts import SiteModel
from stuntd.train.metrics import ClassStats, ConfidentError, Interval, Operating
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
        confident_errors=[ConfidentError("user: " + "x" * 100, 1, 0, 0.97, 0.98)],
        curve=[Operating(0.9, 0.5, 1.0), Operating(0.62, 0.8, 0.99)],
        agreement_interval=Interval(0.87, 0.98),
        covered_interval=Interval(0.97, 1.0),
        n_covered=250,
        novelty_cutoff=0.31,
        familiar_share=0.95,
    )


def test_render_shows_the_headline_numbers():
    text = render(model())
    assert text.startswith("site s1  choice verdict  trained ")
    assert "examples: 240 train, 60 holdout" in text
    assert "holdout agreement 0.950 (95% interval 0.870-0.980, 60 rows)  ece 0.021" in text
    assert "temperature 1.50" in text
    assert "at threshold 0.62: coverage 0.80, agreement 0.990" in text
    assert "allow" in text and "block" in text and "-" in text
    assert "0.97  0.98  expected block, got allow: user: " in text
    assert "x" * 100 in text
    assert "threshold  coverage  agreement" not in text


def test_render_shows_the_intervals_and_the_rows_behind_them():
    site = model()
    site.n_holdout = 400
    text = render(site)
    assert "holdout agreement 0.950 (95% interval 0.870-0.980, 400 rows)" in text
    assert "agreement 0.990 (95% interval 0.970-1.000, 250 rows)" in text
    assert "warning" not in text


def test_render_shows_the_novelty_cutoff_and_the_holdout_it_lets_through():
    assert "novelty cut-off 0.310, lets 95% of the holdout through" in render(model())


def test_render_of_a_head_without_a_cutoff_leaves_the_novelty_gate_out():
    site = model()
    site.novelty_cutoff = site.familiar_share = None
    assert "novelty" not in render(site)


def test_render_warns_about_a_small_holdout():
    site = model()
    site.n_holdout = 199
    text = render(site)
    assert "warning: the holdout has 199 rows, fewer than 200" in text


def test_render_warns_about_a_thin_operating_point():
    site = model()
    site.n_holdout = 400
    site.n_covered = 99
    assert "warning: the threshold rests on 99 rows, fewer than 100" in render(site)


def test_render_marks_a_mistake_under_the_threshold():
    site = model()
    site.confident_errors = [
        ConfidentError("served", 1, 0, 0.97, 0.98),
        ConfidentError("held back", 1, 0, 0.5, 0.6),
    ]
    lines = render(site).splitlines()
    assert next(line for line in lines if "served" in line).endswith("served")
    assert next(line for line in lines if "held back" in line).endswith("(under the threshold)")


def test_render_of_a_model_without_statistics_leaves_them_out():
    site = model()
    site.n_holdout = 400
    site.agreement_interval = site.covered_interval = site.n_covered = None
    site.confident_errors = [ConfidentError("old", 1, 0, 0.97)]
    text = render(site)
    assert "interval" not in text and "warning" not in text
    assert "0.97  -  expected block, got allow: old" in text


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
    assert data[0]["agreement_interval"] == {"low": 0.87, "high": 0.98}
    assert data[0]["n_covered"] == 250 and data[0]["confident_errors"][0]["probability"] == 0.98
