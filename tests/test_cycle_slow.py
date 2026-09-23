import json
import statistics
import time
from dataclasses import dataclass

import pytest
from cycle_helpers import COLLECTED, SITE, UPSTREAM, Provider, app_for, message, post

from stuntd.cli import main
from stuntd.serve.modes import MODE_LIVE, MODE_SHADOW, read_mode
from stuntd.settings import config_path, load_settings, models_path
from stuntd.train.artifacts import load_model, site_dir

pytestmark = [pytest.mark.slow, pytest.mark.anyio]

SHADOWED = 4
LIVE_REQUESTS = 20


@dataclass(frozen=True)
class Answered:
    header: str
    body: dict
    refund: bool
    wall_ms: float


def write_config(data_dir, checkpoint):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "stuntd.toml").write_text(
        f'upstream = "{UPSTREAM}"\n'
        "[training]\nmin_examples = 10\nholdout = 0.25\n"
        f'base_model = "{checkpoint.replace(chr(92), "/")}"\ndevice = "cpu"\nepochs = 1\n'
        "[serving]\ncheck_share = 0.0\n",
        encoding="utf-8",
    )


async def test_cycle_serves_a_trained_head_from_the_real_checkpoint(
    data_dir, laya_checkpoint, capsys
):
    from stuntd.serve.decider import Decider

    write_config(data_dir, laya_checkpoint)
    provider = Provider()
    models = models_path(load_settings(config_path()))

    collecting = app_for(provider)
    for index in range(COLLECTED):
        relayed = await post(collecting, message(index))
        assert relayed.headers["x-stuntd"] == f"collect; site={SITE}"
    assert provider.calls == COLLECTED
    assert [(row["site"], row["count"]) for row in collecting.state.store.stats()] == [
        (SITE, COLLECTED)
    ]
    await collecting.state.proxy.aclose()

    assert main(["train"]) == 0
    assert f"{SITE}  trained" in capsys.readouterr().out
    model = load_model(models, SITE)
    assert model.labels == ["false", "true"] and model.threshold is not None
    assert read_mode(site_dir(models, SITE))[0] == MODE_SHADOW

    decider = Decider(laya_checkpoint, device="cpu")
    shadowing = app_for(provider, decider)
    for index in range(100, 100 + SHADOWED):
        compared = await post(shadowing, message(index))
        assert compared.headers["x-stuntd"] == f"shadow; site={SITE}"
        assert json.loads(compared.content)["id"] == "upstream"
    assert provider.calls == COLLECTED + SHADOWED
    shadowed = shadowing.state.store.decisions(SITE, SHADOWED)
    assert [row.mode for row in shadowed] == ["shadow"] * SHADOWED
    assert all(0.0 <= row.confidence <= 1.0 for row in shadowed)
    assert all(row.agree for row in shadowed)
    await shadowing.state.proxy.aclose()

    assert main(["enable", SITE]) == 0
    assert capsys.readouterr().out.strip() == f"{SITE}: {MODE_LIVE}"
    assert read_mode(site_dir(models, SITE))[0] == MODE_LIVE

    serving = app_for(provider, decider)
    answers = []
    for index in range(200, 200 + LIVE_REQUESTS):
        started = time.perf_counter()
        answered = await post(serving, message(index))
        answers.append(
            Answered(
                answered.headers["x-stuntd"],
                json.loads(answered.content),
                index % 2 == 1,
                (time.perf_counter() - started) * 1000,
            )
        )
    served = [item for item in answers if item.header.startswith(f"live; site={SITE}; confidence=")]
    assert len(served) == LIVE_REQUESTS
    assert all(item.body["object"] == "chat.completion" for item in served)
    assert all(
        json.loads(item.body["choices"][0]["message"]["content"]) == {"refund": item.refund}
        for item in served
    )
    assert provider.calls == COLLECTED + SHADOWED + LIVE_REQUESTS - len(served)
    live = serving.state.store.decisions(SITE, len(served))
    assert [row.mode for row in live] == ["live"] * len(served)
    assert all(0.0 <= row.confidence <= 1.0 for row in live)
    print(
        f"cpu decision p50 {statistics.median(row.latency_ms for row in live):.0f} ms;"
        f" wall p50 {statistics.median(item.wall_ms for item in served):.0f} ms;"
        f" {len(served)} of {LIVE_REQUESTS} answered live"
    )
    await serving.state.proxy.aclose()
