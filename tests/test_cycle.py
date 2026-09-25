import json
from dataclasses import dataclass

import pytest
from cycle_helpers import COLLECTED, SITE, UPSTREAM, Provider, app_for, message, post

from stuntd.cli import main
from stuntd.serve.modes import MODE_LIVE, MODE_SHADOW, read_mode, write_mode
from stuntd.serve.monitor import Window, window_agreement
from stuntd.settings import config_path, load_settings, models_path
from stuntd.store.db import Decision
from stuntd.train.artifacts import SiteModel, load_model, save_model, site_dir
from stuntd.train.layout import Layout
from stuntd.train.metrics import ClassStats, confidence, operating_point, predict, softmax
from stuntd.train.run import TrainedHead

pytestmark = pytest.mark.anyio

SATURATED_LOGITS = [[30.0, 0.0], [0.0, 30.0]]
NEARLY_SATURATED_LOGITS = [0.0, 29.9]


@dataclass(frozen=True)
class FakeVerdict:
    label: int
    confidence: float
    latency_ms: int
    probabilities: tuple[float, ...] = ()


SURE_REFUND = FakeVerdict(1, 1.0, 5)
SURE_AND_WRONG = FakeVerdict(0, 1.0, 5)


class FakeDecider:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    def decide(self, model, head_path, text):
        self.calls.append(text)
        return self.verdict


def write_config(data_dir, check_share=0.0, min_window=5):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "stuntd.toml").write_text(
        f'upstream = "{UPSTREAM}"\n'
        "[training]\nmin_examples = 10\nholdout = 0.25\n"
        f"[serving]\ncheck_share = {check_share}\nwindow = 10\nmin_window = {min_window}\n",
        encoding="utf-8",
    )


def perfect_trainer_factory(settings):
    def trainer(dataset, head_path):
        head_path.write_bytes(b"head")
        return TrainedHead(
            [[3.0, 0.0] if item.label == 0 else [0.0, 3.0] for item in dataset.holdout],
            Layout(512, 192, spaced_labels=True),
        )

    return trainer


def saturated_threshold():
    point = operating_point(predict(SATURATED_LOGITS, [0, 1], 1.0), 0.99)
    assert point is not None
    return point.threshold


def nearly_saturated_confidence():
    return confidence(softmax(NEARLY_SATURATED_LOGITS))


def saturated_model():
    return SiteModel(
        site=SITE,
        kind="boolean",
        field="refund",
        labels=["false", "true"],
        base_model="laya",
        temperature=1.0,
        threshold=saturated_threshold(),
        target_agreement=0.99,
        trained_at=100.0,
        n_train=30,
        n_holdout=10,
        agreement=1.0,
        coverage=1.0,
        covered_agreement=1.0,
        ece=0.0,
        per_class={"false": ClassStats(5, 1.0), "true": ClassStats(5, 1.0)},
        confident_errors=[],
        curve=[],
    )


async def test_cycle_collects_trains_shadows_serves_and_demotes(data_dir, monkeypatch, capsys):
    write_config(data_dir)
    provider = Provider()
    models = models_path(load_settings(config_path()))

    collecting = app_for(provider)
    for index in range(COLLECTED):
        relayed = await post(collecting, message(index))
        assert relayed.headers["x-stuntd"] == f"collect; site={SITE}"
        assert json.loads(relayed.content)["id"] == "upstream"
    assert provider.calls == COLLECTED
    assert [(row["site"], row["count"]) for row in collecting.state.store.stats()] == [
        (SITE, COLLECTED)
    ]
    assert collecting.state.store.decisions(SITE, 10) == []
    await collecting.state.proxy.aclose()

    monkeypatch.setattr("stuntd.cli._make_trainer", perfect_trainer_factory)
    assert main(["train"]) == 0
    assert f"{SITE}  trained  holdout agreement 1.000" in capsys.readouterr().out
    model = load_model(models, SITE)
    assert (model.labels, model.n_train, model.n_holdout) == (["false", "true"], 30, 10)
    assert model.threshold is not None
    assert read_mode(site_dir(models, SITE))[0] == MODE_SHADOW

    shadowing = app_for(provider, FakeDecider(SURE_REFUND))
    for index in (41, 43, 45):
        compared = await post(shadowing, message(index))
        assert compared.headers["x-stuntd"] == f"shadow; site={SITE}"
        assert json.loads(compared.content)["id"] == "upstream"
    assert provider.calls == COLLECTED + 3
    rows = shadowing.state.store.decisions(SITE, 10)
    assert [(row.mode, row.answer, row.agree) for row in rows] == [("shadow", "true", True)] * 3
    await shadowing.state.proxy.aclose()

    monkeypatch.setattr("stuntd.cli._require_serving", lambda: None)
    assert main(["enable", SITE]) == 0
    assert capsys.readouterr().out.strip() == f"{SITE}: {MODE_LIVE}"
    assert read_mode(site_dir(models, SITE))[0] == MODE_LIVE

    serving = app_for(provider, FakeDecider(SURE_REFUND))
    answered = await post(serving, message(51))
    assert provider.calls == COLLECTED + 3
    assert answered.headers["x-stuntd"] == f"live; site={SITE}; confidence=1.00"
    body = json.loads(answered.content)
    assert body["object"] == "chat.completion" and body["model"] == "gpt-x"
    assert json.loads(body["choices"][0]["message"]["content"]) == {"refund": True}
    assert [
        (row.mode, row.answer, row.agree) for row in serving.state.store.decisions(SITE, 1)
    ] == [("live", "true", None)]
    await serving.state.proxy.aclose()

    write_config(data_dir, check_share=0.9999, min_window=2)
    checking = app_for(provider, FakeDecider(SURE_AND_WRONG))
    first = await post(checking, message(53))
    assert first.headers["x-stuntd"] == f"collect; site={SITE}; reason=check"
    assert read_mode(site_dir(models, SITE))[0] == MODE_LIVE
    second = await post(checking, message(53))
    assert second.headers["x-stuntd"] == f"collect; site={SITE}; reason=check"
    assert provider.calls == COLLECTED + 5
    assert [
        (row.mode, row.answer, row.agree) for row in checking.state.store.decisions(SITE, 2)
    ] == [("check", "false", False)] * 2
    assert read_mode(site_dir(models, SITE))[0] == MODE_SHADOW
    demoted = await post(checking, message(55))
    assert demoted.headers["x-stuntd"] == f"shadow; site={SITE}"
    assert json.loads(demoted.content)["id"] == "upstream"
    await checking.state.proxy.aclose()

    assert main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "learning": True,
        "sites": [
            {
                "site": SITE,
                "mode": MODE_SHADOW,
                "captures": COLLECTED + 6,
                "shadow": 6,
                "live": 1,
                "agreement": 0.0,
            }
        ],
    }


async def test_live_site_answers_a_request_as_sure_as_its_holdout(data_dir):
    write_config(data_dir)
    provider = Provider()
    folder = save_model(models_path(load_settings(config_path())), saturated_model())
    write_mode(folder, MODE_LIVE, now=0.0)
    verdict = FakeVerdict(1, nearly_saturated_confidence(), 5)
    serving = app_for(provider, FakeDecider(verdict))
    try:
        answered = await post(serving, message(51))
        assert provider.calls == 0
        assert answered.headers["x-stuntd"].startswith(f"live; site={SITE}; confidence=")
    finally:
        await serving.state.proxy.aclose()


def test_window_counts_a_comparison_as_sure_as_the_threshold():
    compared = Decision(SITE, "shadow", "true", nearly_saturated_confidence(), True, 5, 0.0)
    assert window_agreement([compared], saturated_threshold()) == Window(1, 1.0)
