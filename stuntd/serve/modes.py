from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stuntd.modes import MODE_CHECK, MODE_COLLECT, MODE_LIVE, MODE_SHADOW
from stuntd.paths import private_dir, write_private
from stuntd.train.artifacts import SiteModel, list_models, load_model, site_dir

__all__ = [
    "MODE_CHECK",
    "MODE_COLLECT",
    "MODE_FILE",
    "MODE_LIVE",
    "MODE_SHADOW",
    "SiteState",
    "read_mode",
    "site_state",
    "site_states",
    "write_mode",
]

MODE_FILE = "mode.json"
"""File inside a site's model folder holding the mode it serves in."""

_MODES = (MODE_COLLECT, MODE_SHADOW, MODE_LIVE)


@dataclass(frozen=True)
class SiteState:
    """One decision site: the mode it serves in, its model and when the mode last changed."""

    site: str
    mode: str
    model: SiteModel | None
    changed_at: float | None


def read_mode(folder: Path) -> tuple[str, float | None]:
    """The mode recorded in folder; a model published without one yet serves in shadow."""
    path = folder / MODE_FILE
    if not path.is_file():
        return MODE_SHADOW, None
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
        mode = raw["mode"]
        changed_at = raw["changed_at"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if mode not in _MODES:
        raise ValueError(f"{path}: unknown mode {mode!r}")
    # bool is an int subclass, so `"changed_at": true` would otherwise pass as a number.
    if changed_at is not None and (
        isinstance(changed_at, bool) or not isinstance(changed_at, (int, float))
    ):
        raise ValueError(f"{path}: changed_at must be a number or null, got {changed_at!r}")
    return mode, changed_at


def write_mode(folder: Path, mode: str, now: float) -> None:
    """Records the mode a site serves in, creating its folder when the site has none yet."""
    if mode not in _MODES:
        raise ValueError(f"unknown mode {mode!r}")
    text = json.dumps({"mode": mode, "changed_at": now}, allow_nan=False)
    private_dir(folder)
    write_private(folder / MODE_FILE, text)


def _state(models: Path, model: SiteModel) -> SiteState:
    mode, changed_at = read_mode(site_dir(models, model.site))
    return SiteState(model.site, mode, model, changed_at)


def site_states(models: Path) -> list[SiteState]:
    """Every trained site, ordered by name, with the mode it is serving in."""
    return [_state(models, model) for model in list_models(models)]


def site_state(models: Path, site: str) -> SiteState:
    """One site's state; a site without a model is still collecting captures."""
    try:
        model = load_model(models, site)
    except FileNotFoundError:
        return SiteState(site, MODE_COLLECT, None, None)
    return _state(models, model)
