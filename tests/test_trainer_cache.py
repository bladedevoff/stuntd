import types

import pytest

HIDDEN = 1024
TOKENS = 64
TRAIN, HOLDOUT = 6, 2
NEEDED = (TRAIN + HOLDOUT) * TOKENS * HIDDEN * 2
CPU_ALLOCATOR = (
    "[enforce fail at alloc_cpu.cpp:117] . DefaultCPUAllocator: not enough memory: "
    "you tried to allocate 4194304 bytes."
)


@pytest.fixture(scope="module")
def trainer_class():
    pytest.importorskip("torch")
    from stuntd.train.trainer import LayaTrainer

    return LayaTrainer


def rows(count):
    return [{"ids": [0] * TOKENS} for _ in range(count)]


def encoded(count):
    return [f"encoded {index}" for index in range(count)]


def stub(trainer_class, cache_encoder=True, cache_max_bytes=NEEDED, encode=None):
    trainer = trainer_class.__new__(trainer_class)
    trainer._cache_encoder = cache_encoder
    trainer._cache_max_bytes = cache_max_bytes
    trainer._agent = types.SimpleNamespace(
        model=types.SimpleNamespace(
            encoder=types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=HIDDEN))
        )
    )
    trainer._encode = encode or (lambda rows: encoded(len(rows)))
    return trainer


def test_both_splits_are_cached_within_the_budget(trainer_class):
    trainer = stub(trainer_class)
    assert trainer._cache(rows(TRAIN), rows(HOLDOUT)) == (encoded(TRAIN), encoded(HOLDOUT))


@pytest.mark.parametrize(
    ("cache_encoder", "cache_max_bytes"),
    [(False, NEEDED), (True, 0), (True, NEEDED - 1)],
    ids=["turned off", "budget of zero", "budget one byte short"],
)
def test_the_uncached_path_is_taken_when_the_cache_is_refused(
    trainer_class, cache_encoder, cache_max_bytes
):
    trainer = stub(trainer_class, cache_encoder, cache_max_bytes)
    assert trainer._cache(rows(TRAIN), rows(HOLDOUT)) == (None, None)


def refuse(exc):
    def encode(rows):
        raise exc

    return encode


@pytest.mark.parametrize(
    "kind",
    ["host", "cpu allocator", "cuda"],
    ids=["MemoryError", "the torch CPU allocator", "torch out of memory"],
)
def test_the_uncached_path_is_taken_when_an_allocation_fails(trainer_class, kind):
    import torch

    failure = {
        "host": MemoryError(),
        "cpu allocator": RuntimeError(CPU_ALLOCATOR),
        "cuda": torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"),
    }[kind]
    trainer = stub(trainer_class, encode=refuse(failure))
    assert trainer._cache(rows(TRAIN), rows(HOLDOUT)) == (None, None)


def test_an_unrelated_failure_is_not_swallowed(trainer_class):
    trainer = stub(trainer_class, encode=refuse(RuntimeError("shapes do not match")))
    with pytest.raises(RuntimeError, match="shapes do not match"):
        trainer._cache(rows(TRAIN), rows(HOLDOUT))


def test_the_refusal_is_logged_where_a_train_run_sees_it(trainer_class, caplog):
    trainer = stub(trainer_class, cache_max_bytes=0)
    with caplog.at_level("WARNING", logger="stuntd.train.trainer"):
        trainer._cache(rows(TRAIN), rows(HOLDOUT))
    assert [record.levelname for record in caplog.records] == ["WARNING"]
    assert "re-encoding the encoder every epoch" in caplog.text
