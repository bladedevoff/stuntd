import os
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from stuntd.store.db import Example
from stuntd.train.artifacts import HEAD_FILE, SiteModel
from stuntd.train.dataset import build_dataset

pytestmark = pytest.mark.slow

BOOL = '{"properties":{"refund":{"type":"boolean"}},"type":"object"}'

THREAD_ROUNDS = 3

CHOICE_QUESTION = {
    "type": "choice",
    "instructions": "Choose refund",
    "criteria": {"true": None, "false": None},
}

NOUL_QUESTION = {"type": "noul", "instructions": "The customer asks for money back"}


def synthetic():
    yes = "user: I was charged twice for invoice {n}, please send the money back."
    no = "user: Where do I find the invoice {n} for last month?"
    return [
        Example((yes if i % 2 else no).format(n=1000 + i), "true" if i % 2 else "false", float(i))
        for i in range(40)
    ]


def opposite_head(source, target):
    from safetensors.torch import load_file, save_file

    head = load_file(str(source))
    for name in ("scorer.3.weight", "scorer.3.bias"):
        head[name] = -head[name]
    save_file(head, str(target))


def site_model(base_model, dataset, layout, site=None):
    return SiteModel(
        site=site or dataset.site,
        kind=dataset.kind,
        field=dataset.field,
        labels=list(dataset.labels),
        base_model=base_model,
        temperature=1.0,
        threshold=None,
        target_agreement=0.95,
        trained_at=0.0,
        n_train=len(dataset.train),
        n_holdout=len(dataset.holdout),
        agreement=1.0,
        coverage=None,
        covered_agreement=None,
        ece=0.0,
        per_class={},
        confident_errors=[],
        curve=[],
        max_len=layout.max_len,
        head_max_len=layout.head_max_len,
        spaced_labels=layout.spaced_labels,
    )


def head_overhead(decider, model, head, text):
    started = time.perf_counter()
    verdict = decider.decide(model, head, text)
    return time.perf_counter() - started - verdict.latency_ms / 1000.0


@pytest.fixture(scope="module")
def trained(laya_checkpoint, tmp_path_factory):
    from stuntd.serve.decider import Decider
    from stuntd.train.trainer import LayaTrainer

    folder = tmp_path_factory.mktemp("sites")
    head = folder / "refund" / HEAD_FILE
    other = folder / "reversed" / HEAD_FILE
    for path in (head, other):
        path.parent.mkdir()
    dataset = build_dataset("refund", "boolean", BOOL, synthetic(), 10, 0.25)
    trained = LayaTrainer(laya_checkpoint, device="cpu", epochs=1, batch_size=8)(dataset, head)
    opposite_head(head, other)
    return SimpleNamespace(
        decider=Decider(laya_checkpoint, device="cpu"),
        model=site_model(laya_checkpoint, dataset, trained.layout),
        other_model=site_model(laya_checkpoint, dataset, trained.layout, "reversed"),
        head=head,
        other=other,
        dataset=dataset,
        logits=trained.logits,
    )


def test_decider_agrees_with_the_trainer(trained):
    verdicts = [
        trained.decider.decide(trained.model, trained.head, item.text)
        for item in trained.dataset.holdout
    ]
    expected = [max(range(len(row)), key=row.__getitem__) for row in trained.logits]
    assert [verdict.label for verdict in verdicts] == expected
    assert all(0.0 <= verdict.confidence <= 1.0 for verdict in verdicts)
    assert all(verdict.latency_ms >= 0 for verdict in verdicts)


def test_decider_reuses_a_cached_head(trained, monkeypatch):
    text = trained.dataset.holdout[0].text
    os.utime(trained.head)
    cold = head_overhead(trained.decider, trained.model, trained.head, text)
    warm = head_overhead(trained.decider, trained.model, trained.head, text)
    assert warm < cold

    inner = trained.decider._agent.model
    original = inner.load_state_dict
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(inner, "load_state_dict", counted)
    trained.decider.decide(trained.other_model, trained.other, text)
    trained.decider.decide(trained.model, trained.head, text)
    trained.decider.decide(trained.model, trained.head, text)
    assert len(calls) == 2


def test_decider_keeps_two_sites_apart_across_threads(trained):
    sites = [(trained.model, trained.head), (trained.other_model, trained.other)]
    texts = [item.text for item in trained.dataset.holdout][:4]
    alone = {
        (index, text): trained.decider.decide(model, head, text).label
        for index, (model, head) in enumerate(sites)
        for text in texts
    }
    assert any(alone[(0, text)] != alone[(1, text)] for text in texts)

    jobs = [(index, text) for _ in range(THREAD_ROUNDS) for index in (0, 1) for text in texts]

    def run(job):
        model, head = sites[job[0]]
        return job, trained.decider.decide(model, head, job[1]).label

    with ThreadPoolExecutor(max_workers=4) as pool:
        together = list(pool.map(run, jobs))
    assert [label for _, label in together] == [alone[job] for job, _ in together]


def test_decider_rejects_weights_that_are_not_a_head(trained, tmp_path):
    import torch
    from safetensors.torch import load_file, save_file

    text = trained.dataset.holdout[0].text
    before = trained.decider.decide(trained.model, trained.head, text)
    weights = trained.decider._agent.model.state_dict()
    encoder = min(
        (name for name in weights if name.startswith("encoder.")),
        key=lambda name: weights[name].numel(),
    )
    head = load_file(str(trained.head))
    head[encoder] = torch.zeros_like(weights[encoder], device="cpu")
    foreign = tmp_path / HEAD_FILE
    save_file(head, str(foreign))
    with pytest.raises(RuntimeError, match="head does not match the base model"):
        trained.decider.decide(trained.model, foreign, text)
    after = trained.decider.decide(trained.model, trained.head, text)
    assert (after.label, after.confidence) == (before.label, before.confidence)


def test_decider_rejects_a_head_from_another_model(trained, tmp_path):
    import torch
    from safetensors.torch import save_file

    foreign = tmp_path / HEAD_FILE
    save_file({"head.nonesuch.weight": torch.zeros(2, 2)}, str(foreign))
    with pytest.raises(RuntimeError, match="head does not match the base model"):
        trained.decider.decide(trained.model, foreign, trained.dataset.holdout[0].text)


def test_decider_reports_a_missing_head(trained, tmp_path):
    with pytest.raises(RuntimeError, match="decision failed"):
        trained.decider.decide(trained.model, tmp_path / HEAD_FILE, "user: anything")


def test_decider_keeps_one_head_per_path(trained, tmp_path):
    head = tmp_path / HEAD_FILE
    head.write_bytes(trained.head.read_bytes())
    text = trained.dataset.holdout[0].text
    trained.decider.decide(trained.model, head, text)
    os.utime(head, (0, 0))
    trained.decider.decide(trained.model, head, text)
    assert len([key for key in trained.decider._heads if key[0] == head]) == 1


def test_decider_caches_no_more_heads_than_the_bound(trained, tmp_path):
    from stuntd.serve.decider import HEAD_CACHE_SIZE

    text = trained.dataset.holdout[0].text
    heads = []
    for index in range(HEAD_CACHE_SIZE + 1):
        head = tmp_path / f"site{index}" / HEAD_FILE
        head.parent.mkdir()
        head.write_bytes(trained.head.read_bytes())
        heads.append(head)
        trained.decider.decide(trained.model, head, text)
    assert [key[0] for key in trained.decider._heads] == heads[1:]


def test_answer_gives_a_jev_shaped_answer(trained):
    result = trained.decider.answer(trained.dataset.holdout[0].text, {"refund": CHOICE_QUESTION})
    assert set(result) == {"answers", "usage"}
    answer = result["answers"]["refund"]
    assert answer["type"] == "choice"
    assert answer["choice"] in trained.model.labels
    assert set(answer["probabilities"]) == set(trained.model.labels)
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= answer["confidence"] <= 1.0
    assert "action" not in answer
    assert result["usage"]["input_tokens"] > 0


def test_warm_base_leaves_the_decider_answering(trained):
    trained.decider.warm_base()
    result = trained.decider.answer(trained.dataset.holdout[0].text, {"refund": CHOICE_QUESTION})
    assert result["answers"]["refund"]["choice"] in trained.model.labels


def test_answer_takes_questions_of_two_types(trained):
    result = trained.decider.answer(
        trained.dataset.holdout[0].text,
        {"refund": CHOICE_QUESTION, "money_back": NOUL_QUESTION},
    )
    assert set(result["answers"]) == {"refund", "money_back"}
    assert result["answers"]["refund"]["choice"] in trained.model.labels
    assert result["answers"]["money_back"]["type"] == "noul"
    assert 0.0 <= result["answers"]["money_back"]["noul"] <= 1.0
    assert all("action" not in answer for answer in result["answers"].values())


def test_answer_leaves_the_trained_head_reinstalled(trained):
    text = trained.dataset.holdout[0].text
    before = trained.decider.decide(trained.model, trained.head, text)
    cached = list(trained.decider._heads)
    trained.decider.answer(text, {"refund": CHOICE_QUESTION})
    after = trained.decider.decide(trained.model, trained.head, text)
    assert (after.label, after.confidence) == (before.label, before.confidence)
    assert list(trained.decider._heads) == cached


def test_answer_reports_a_question_that_does_not_fit(trained):
    crowded = {
        "type": "choice",
        "instructions": "Choose refund",
        "criteria": dict.fromkeys(f"label{index}" for index in range(400)),
    }
    with pytest.raises(RuntimeError, match="decision failed"):
        trained.decider.answer("user: anything", {"refund": crowded})
