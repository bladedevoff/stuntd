import stat
import sys

from stuntd import paths


def test_data_dir_honours_env_and_is_private(data_dir):
    created = paths.data_dir()
    assert created == data_dir
    assert created.is_dir()
    if sys.platform != "win32":
        assert stat.S_IMODE(created.stat().st_mode) == 0o700


def test_private_file_is_owner_only(data_dir):
    target = paths.private_file(paths.data_dir() / "captures.sqlite")
    assert target.exists()
    if sys.platform != "win32":
        assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_private_dir_is_owner_only(data_dir):
    target = paths.private_dir(paths.data_dir() / "models" / "s1")
    assert target.is_dir()
    if sys.platform != "win32":
        assert stat.S_IMODE(target.stat().st_mode) == 0o700
        assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700


def test_default_location_per_platform(monkeypatch, tmp_path):
    monkeypatch.delenv("STUNTD_DATA_DIR", raising=False)
    if sys.platform == "win32":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert paths.default_data_dir() == tmp_path / "stuntd"
    else:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        assert paths.default_data_dir() == tmp_path / "stuntd"
