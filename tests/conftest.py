import os

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def laya_checkpoint():
    pytest.importorskip("torch")
    base = os.environ.get("STUNTD_LAYA_MODEL")
    if not base:
        pytest.skip("STUNTD_LAYA_MODEL is not set")
    return base


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("STUNTD_DATA_DIR", str(tmp_path / "stuntd-data"))
    return tmp_path / "stuntd-data"
