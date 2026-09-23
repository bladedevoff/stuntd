from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

__all__ = ["data_dir", "default_data_dir", "private_dir", "private_file", "write_private"]


def default_data_dir() -> Path:
    """The platform's standard data directory for stuntd, ignoring any override."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "stuntd"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "stuntd"


def data_dir() -> Path:
    """The data directory, created if missing and readable only by its owner."""
    override = os.environ.get("STUNTD_DATA_DIR")
    target = Path(override) if override else default_data_dir()
    # Captures hold prompt text, so the directory is never world-readable, not even between
    # its creation and the chmod: mkdir applies the mode only when it creates the directory,
    # and the chmod narrows one that already existed.
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    if sys.platform != "win32":
        os.chmod(target, 0o700)
    return target


def private_file(path: Path) -> Path:
    """Creates path if it is missing and narrows it to its owner either way."""
    if not path.exists():
        path.touch(mode=0o600)
    # touch leaves the mode alone when the file already exists.
    if sys.platform != "win32":
        os.chmod(path, 0o600)
    return path


def write_private(path: Path, text: str) -> None:
    """Replaces path with text in one step, so a reader never sees a half-written file."""
    # A daemon reading these files while a command rewrites them would otherwise be handed a
    # truncated one, and os.replace is atomic on Windows as well as POSIX.
    handle, made = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    os.close(handle)
    temp = Path(made)
    private_file(temp)
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    except OSError:
        temp.unlink(missing_ok=True)
        raise


def private_dir(path: Path) -> Path:
    """Creates path and every missing ancestor, narrowing the directory itself to its owner."""
    # mkdir(parents=True) would leave the ancestors it creates at the default mode, and these
    # directories hold captured prompt text, so every missing step is created narrow.
    for step in (*reversed(path.parents), path):
        step.mkdir(exist_ok=True, mode=0o700)
    if sys.platform != "win32":
        os.chmod(path, 0o700)
    return path
