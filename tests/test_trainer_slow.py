import pytest

from stuntd.store.db import Example
from stuntd.train.dataset import build_dataset

pytestmark = pytest.mark.slow

BOOL = '{"properties":{"refund":{"type":"boolean"}},"type":"object"}'


def synthetic():
    yes = "user: I was charged twice for invoice {n}, please send the money back."
    no = "user: Where do I find the invoice {n} for last month?"
    return [
        Example((yes if i % 2 else no).format(n=1000 + i), "true" if i % 2 else "false", float(i))
        for i in range(40)
    ]


@pytest.fixture(scope="module")
def trainer(laya_checkpoint):
    from stuntd.train.trainer import LayaTrainer

    return LayaTrainer(laya_checkpoint, device="cpu", epochs=1, batch_size=8)


def test_head_trains_on_a_local_checkpoint(tmp_path, trainer):
    from safetensors import safe_open

    from stuntd.train.artifacts import HEAD_FILE

    data = build_dataset("s", "boolean", BOOL, synthetic(), 10, 0.25)
    logits = trainer(data, tmp_path / HEAD_FILE)
    assert len(logits) == len(data.holdout) and all(len(row) == 2 for row in logits)
    with safe_open(str(tmp_path / HEAD_FILE), framework="pt") as handle:
        keys = list(handle.keys())
    assert keys and all(k.split(".")[0] in {"head", "scorer", "type_emb"} for k in keys)
    assert not any(k.startswith("encoder.") for k in keys)
    assert (tmp_path / HEAD_FILE).stat().st_size > 0


def test_head_resets_between_sites(tmp_path, trainer):
    from stuntd.train.artifacts import HEAD_FILE

    data = build_dataset("s", "boolean", BOOL, synthetic(), 10, 0.25)
    first = trainer(data, tmp_path / HEAD_FILE)
    second = trainer(data, tmp_path / HEAD_FILE)
    flat = [value for row in second for value in row]
    assert flat == pytest.approx([value for row in first for value in row], abs=1e-6)


def agreement(logits, holdout):
    pairs = zip(logits, holdout, strict=True)
    hits = sum(row.index(max(row)) == item.label for row, item in pairs)
    return hits / len(holdout)


def test_the_encoder_cache_agrees_with_the_uncached_path(tmp_path, laya_checkpoint):
    from safetensors import safe_open

    from stuntd.train.artifacts import HEAD_FILE
    from stuntd.train.trainer import LayaTrainer

    data = build_dataset("s", "boolean", BOOL, synthetic(), 10, 0.25)
    scores, keys = [], []
    for cached in (True, False):
        head = tmp_path / f"{int(cached)}-{HEAD_FILE}"
        trainer = LayaTrainer(
            laya_checkpoint, device="cpu", epochs=3, batch_size=8, cache_encoder=cached
        )
        scores.append(agreement(trainer(data, head), data.holdout))
        with safe_open(str(head), framework="pt") as handle:
            keys.append(set(handle.keys()))
    assert abs(scores[0] - scores[1]) <= 0.05
    assert keys[0] == keys[1]
