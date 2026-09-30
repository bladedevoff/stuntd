import threading
import time

import pytest

torch = pytest.importorskip("torch")
Decider = pytest.importorskip("stuntd.serve.decider").Decider


class FakeAgent:
    loads = 0
    fail = False

    def __init__(self, checkpoint, device=None):
        time.sleep(0.05)
        type(self).loads += 1
        if type(self).fail:
            raise RuntimeError("no checkpoint")
        self.model = torch.nn.Linear(1, 1)

    def system_one(self, state, questions):
        return {"answers": {}, "usage": {}}


@pytest.fixture
def agent(monkeypatch):
    FakeAgent.loads = 0
    FakeAgent.fail = False
    monkeypatch.setattr("stuntd.serve.decider.laya.Agent", FakeAgent)
    return FakeAgent


def test_decider_loads_at_construction_by_default(agent):
    Decider("base")
    assert agent.loads == 1


def test_lazy_decider_loads_nothing_at_construction(agent):
    Decider("base", lazy=True)
    assert agent.loads == 0


def test_lazy_decider_loads_once_across_requests(agent):
    decider = Decider("base", lazy=True)
    decider.answer("state", {})
    decider.answer("state", {})
    assert agent.loads == 1


def test_lazy_decider_loads_once_under_concurrent_first_requests(agent):
    decider = Decider("base", lazy=True)
    threads = [threading.Thread(target=decider.answer, args=("state", {})) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert agent.loads == 1


def test_lazy_decider_retries_after_a_failed_load(agent):
    decider = Decider("base", lazy=True)
    agent.fail = True
    with pytest.raises(RuntimeError, match="base model failed to load: no checkpoint"):
        decider.answer("state", {})
    agent.fail = False
    assert decider.answer("state", {}) == {"answers": {}, "usage": {}}
    assert agent.loads == 2
