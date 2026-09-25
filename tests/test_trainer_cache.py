import os
import sys
import types

import pytest

SITE = "banking77"
HIDDEN = 1024
TOKENS = 64
TRAIN, HOLDOUT = 6, 2
NEEDED = (TRAIN + HOLDOUT) * TOKENS * HIDDEN * 2
CPU_ALLOCATOR = (
    "[enforce fail at alloc_cpu.cpp:117] . DefaultCPUAllocator: not enough memory: "
    "you tried to allocate 4194304 bytes."
)


@pytest.fixture(scope="module")
def trainer_module():
    pytest.importorskip("torch")
    from stuntd.train import trainer

    return trainer


@pytest.fixture(scope="module")
def trainer_class(trainer_module):
    return trainer_module.LayaTrainer


def rows(count, tokens=TOKENS):
    return [{"ids": [0] * tokens} for _ in range(count)]


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
    assert trainer._cache(SITE, rows(TRAIN), rows(HOLDOUT)) == (encoded(TRAIN), encoded(HOLDOUT))


@pytest.mark.parametrize(
    ("cache_encoder", "cache_max_bytes"),
    [(False, NEEDED), (True, 1), (True, NEEDED - 1)],
    ids=["turned off", "budget of one byte", "budget one byte short"],
)
def test_the_uncached_path_is_taken_when_the_cache_is_refused(
    trainer_class, cache_encoder, cache_max_bytes
):
    trainer = stub(trainer_class, cache_encoder, cache_max_bytes)
    assert trainer._cache(SITE, rows(TRAIN), rows(HOLDOUT)) == (None, None)


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
    assert trainer._cache(SITE, rows(TRAIN), rows(HOLDOUT)) == (None, None)


def test_an_unrelated_failure_is_not_swallowed(trainer_class):
    trainer = stub(trainer_class, encode=refuse(RuntimeError("shapes do not match")))
    with pytest.raises(RuntimeError, match="shapes do not match"):
        trainer._cache(SITE, rows(TRAIN), rows(HOLDOUT))


@pytest.mark.parametrize(
    ("budget", "encode", "line"),
    [
        (
            2**30,
            None,
            "stuntd: banking77 needs 3.0 GiB for the encoder cache, over the 1.0 GiB budget,"
            " so every epoch re-encodes; raise training.cache_max_mb\n",
        ),
        (
            4 * 2**30,
            refuse(MemoryError()),
            "stuntd: banking77 needs 3.0 GiB for the encoder cache and memory ran out,"
            " so every epoch re-encodes; free memory: MemoryError()\n",
        ),
    ],
    ids=["over the budget", "allocation failed"],
)
def test_cache_fallback_prints_one_line_on_stderr(trainer_class, capsys, budget, encode, line):
    from stuntd.cli import _warnings_on_stderr

    trainer = stub(trainer_class, cache_max_bytes=budget, encode=encode)
    with _warnings_on_stderr():
        assert trainer._cache(SITE, rows(1, 3 * 2**19), []) == (None, None)
    assert capsys.readouterr().err == line


@pytest.mark.parametrize(
    ("requested", "memory", "expected"),
    [(0, 32 * 2**30, 16 * 2**30), (512 * 2**20, 32 * 2**30, 512 * 2**20), (0, 0, 4 * 2**30)],
    ids=["zero is half of memory", "positive is taken as given", "unknown memory is 4 GiB"],
)
def test_cache_budget_resolves_the_setting(
    trainer_module, monkeypatch, requested, memory, expected
):
    monkeypatch.setattr(trainer_module, "_physical_memory", lambda: memory)
    assert trainer_module._cache_budget(requested) == expected


def test_physical_memory_reads_this_machine(trainer_module):
    assert trainer_module._physical_memory() > 0


def unknown_name(name):
    raise ValueError(name)


def os_error(name):
    raise OSError(name)


def indeterminate(name):
    return -1


def pages_indeterminate(name):
    return -1 if name == "SC_PHYS_PAGES" else 4096


@pytest.mark.skipif(sys.platform == "win32", reason="the sysconf probe runs off Windows")
@pytest.mark.parametrize(
    "sysconf",
    [unknown_name, os_error, indeterminate, pages_indeterminate],
    ids=["unknown name", "os error", "both indeterminate", "pages indeterminate"],
)
def test_physical_memory_is_unknown_when_sysconf_fails(trainer_module, monkeypatch, sysconf):
    monkeypatch.setattr(os, "sysconf", sysconf)
    assert trainer_module._physical_memory() == 0


@pytest.mark.skipif(
    sys.platform != "win32", reason="the GlobalMemoryStatusEx probe runs on Windows"
)
def test_physical_memory_is_unknown_when_windows_refuses(trainer_module, monkeypatch):
    import ctypes

    monkeypatch.setattr(ctypes.windll.kernel32, "GlobalMemoryStatusEx", lambda status: 0)
    assert trainer_module._physical_memory() == 0
