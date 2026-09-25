from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stuntd.paths import private_dir, write_private
from stuntd.train.metrics import ClassStats, ConfidentError, Operating

__all__ = [
    "HEAD_FILE",
    "META_FILE",
    "SiteModel",
    "list_models",
    "load_model",
    "save_model",
    "site_dir",
    "write_meta",
]

HEAD_FILE = "head.safetensors"
META_FILE = "meta.json"

_SITE_NAME = re.compile(r"[A-Za-z0-9_.-]{1,64}")


@dataclass
class SiteModel:
    """One trained site: how its head was fitted, and how well it did on the holdout."""

    site: str
    kind: str
    field: str
    labels: list[str]
    base_model: str
    temperature: float
    threshold: float | None
    target_agreement: float
    trained_at: float
    n_train: int
    n_holdout: int
    agreement: float
    coverage: float | None
    covered_agreement: float | None
    ece: float
    per_class: dict[str, ClassStats]
    confident_errors: list[ConfidentError]
    curve: list[Operating]
    # None is the checkpoint's own length, which every head written by 0.1.0 was trained with.
    max_len: int | None = None
    head_max_len: int | None = None
    spaced_labels: bool = False


def site_dir(models: Path, site: str) -> Path:
    """The folder holding one site's model, for a site name that cannot escape models."""
    # Site names arrive from a request header, so only that header's charset is let through:
    # anything else, a drive-relative "c:evil" included, would land outside models. A name of
    # nothing but dots passes the charset and still walks up, so it is refused separately.
    if not _SITE_NAME.fullmatch(site) or set(site) == {"."}:
        raise ValueError(f"invalid site name {site!r}")
    return models / site


def write_meta(folder: Path, model: SiteModel) -> None:
    """Writes one model's metadata into folder, which is created readable only by its owner."""
    # Serialised before anything is created, so a model carrying NaN leaves no truncated file.
    text = json.dumps(asdict(model), ensure_ascii=False, allow_nan=False, indent=2)
    private_dir(folder)
    write_private(folder / META_FILE, text)


def save_model(models: Path, model: SiteModel) -> Path:
    """Writes a model's metadata under its own folder in models and returns that folder."""
    folder = site_dir(models, model.site)
    write_meta(folder, model)
    return folder


def _from_dict(raw: dict[str, Any]) -> SiteModel:
    # Any because json.loads returns untyped values; the fields below give them their types back.
    return SiteModel(
        site=raw["site"],
        kind=raw["kind"],
        field=raw["field"],
        labels=list(raw["labels"]),
        base_model=raw["base_model"],
        temperature=raw["temperature"],
        threshold=raw["threshold"],
        target_agreement=raw["target_agreement"],
        trained_at=raw["trained_at"],
        n_train=raw["n_train"],
        n_holdout=raw["n_holdout"],
        agreement=raw["agreement"],
        coverage=raw["coverage"],
        covered_agreement=raw["covered_agreement"],
        ece=raw["ece"],
        per_class={name: ClassStats(**stats) for name, stats in raw["per_class"].items()},
        confident_errors=[ConfidentError(**error) for error in raw["confident_errors"]],
        curve=[Operating(**point) for point in raw["curve"]],
        max_len=raw.get("max_len"),
        head_max_len=raw.get("head_max_len"),
        spaced_labels=raw.get("spaced_labels", False),
    )


def load_model(models: Path, site: str) -> SiteModel:
    """Reads one site's model metadata, raising FileNotFoundError when it has none."""
    path = site_dir(models, site) / META_FILE
    text = path.read_text(encoding="utf-8")
    try:
        return _from_dict(json.loads(text))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def list_models(models: Path) -> list[SiteModel]:
    """Every model under models, ordered by site name, empty when nothing has been trained."""
    if not models.is_dir():
        return []
    return [
        load_model(models, folder.name)
        for folder in sorted(models.iterdir())
        if (folder / META_FILE).is_file()
    ]
