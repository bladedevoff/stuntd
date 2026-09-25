from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from stuntd.paths import private_dir
from stuntd.serve.modes import MODE_SHADOW, write_mode
from stuntd.settings import Settings, models_path
from stuntd.store.db import SiteInfo, Store
from stuntd.train.artifacts import HEAD_FILE, SiteModel, site_dir, write_meta
from stuntd.train.dataset import NotTrainable, SiteDataset, build_dataset
from stuntd.train.layout import Layout
from stuntd.train.metrics import (
    confident_errors,
    ece,
    fit_temperature,
    operating_curve,
    operating_point,
    per_class,
    predict,
)

__all__ = ["NO_CAPTURES", "TrainResult", "TrainedHead", "Trainer", "train_sites"]

NO_CAPTURES = "no captures for this site"
"""Why a named site gets no model: nothing was ever recorded for it."""

_WORK_SUFFIX = ".work"
_OLD_SUFFIX = ".old"

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainedHead:
    """What a trainer hands back for one site: the holdout logits and the layout it trained with."""

    logits: list[list[float]]
    layout: Layout


Trainer = Callable[[SiteDataset, Path], TrainedHead]
"""Fits a head on one dataset, writes it to the given path and says what it trained."""


@dataclass(frozen=True)
class TrainResult:
    """What one site got out of a training run: a model, or the reason it did not get one."""

    site: str
    model: SiteModel | None
    reason: str | None


def _remove(folder: Path) -> None:
    if folder.is_dir():
        shutil.rmtree(folder)


def _fresh(folder: Path) -> None:
    _remove(folder)
    private_dir(folder)


def _work_root(models: Path) -> Path:
    # A sibling of the models directory, so a work folder is never a model folder that list_models
    # would read back, and so nothing is appended to a site name that already fills the whitelist.
    return models.with_name(models.name + _WORK_SUFFIX)


def _restore(previous: Path, folder: Path) -> None:
    try:
        previous.rename(folder)
    except OSError:
        _log.error("%s could not be restored, the previous model is left in %s", folder, previous)


def _publish(models: Path, site: str) -> None:
    # Whole folders are renamed rather than files moved into place because a serving process may
    # be reading the previous model: it keeps a complete folder until the swap, never a half-new one.
    folder = site_dir(models, site)
    work = _work_root(models)
    trained = site_dir(work, site)
    previous = work / (site + _OLD_SUFFIX)
    private_dir(models)
    _remove(previous)
    if folder.is_dir():
        folder.rename(previous)
    published = False
    try:
        trained.rename(folder)
        published = True
    finally:
        # Anything from a failed rename to Ctrl-C between the two of them leaves the site without
        # a model, so the backup goes back before the interruption is allowed through.
        if not published and previous.is_dir():
            _restore(previous, folder)
    _remove(previous)


def _evaluate(
    dataset: SiteDataset, trained: TrainedHead, settings: Settings, trained_at: float
) -> SiteModel:
    labels = [item.label for item in dataset.holdout]
    temperature = fit_temperature(trained.logits, labels)
    predictions = predict(trained.logits, labels, temperature)
    point = operating_point(predictions, settings.target_agreement)
    correct = sum(one.label == one.predicted for one in predictions)
    return SiteModel(
        site=dataset.site,
        kind=dataset.kind,
        field=dataset.field,
        labels=list(dataset.labels),
        base_model=settings.base_model,
        temperature=temperature,
        threshold=None if point is None else point.threshold,
        target_agreement=settings.target_agreement,
        trained_at=trained_at,
        n_train=len(dataset.train),
        n_holdout=len(dataset.holdout),
        agreement=correct / len(predictions),
        coverage=None if point is None else point.coverage,
        covered_agreement=None if point is None else point.agreement,
        ece=ece(predictions),
        per_class=per_class(predictions, dataset.labels),
        confident_errors=confident_errors(predictions, [item.text for item in dataset.holdout]),
        curve=operating_curve(predictions),
        max_len=trained.layout.max_len,
        head_max_len=trained.layout.head_max_len,
        spaced_labels=trained.layout.spaced_labels,
    )


def _train_one(
    store: Store, settings: Settings, trainer: Trainer, info: SiteInfo, now: Callable[[], float]
) -> TrainResult:
    models = models_path(settings)
    try:
        site_dir(models, info.site)
    except ValueError as exc:
        return TrainResult(info.site, None, str(exc))
    try:
        dataset = build_dataset(
            info.site,
            info.kind,
            info.schema_canonical,
            store.examples(info.site),
            settings.min_examples,
            settings.holdout,
        )
    except NotTrainable as exc:
        return TrainResult(info.site, None, str(exc))
    working = site_dir(_work_root(models), info.site)
    try:
        _fresh(working)
        model = _evaluate(dataset, trainer(dataset, working / HEAD_FILE), settings, now())
        write_meta(working, model)
        write_mode(working, MODE_SHADOW, model.trained_at)
        _publish(models, info.site)
    except NotTrainable as exc:
        # The trainer refuses a dataset build_dataset accepted when its labels outgrow the head.
        return TrainResult(info.site, None, str(exc))
    finally:
        _remove(working)
    return TrainResult(info.site, model, None)


def train_sites(
    store: Store,
    settings: Settings,
    trainer: Trainer,
    sites: Sequence[str] = (),
    now: Callable[[], float] = time.time,
) -> list[TrainResult]:
    """Trains, measures and publishes a model for every named site, or for all of them."""
    known = {info.site: info for info in store.sites()}
    wanted = list(sites) if sites else list(known)
    results = []
    for site in wanted:
        info = known.get(site)
        if info is None:
            results.append(TrainResult(site, None, NO_CAPTURES))
            continue
        results.append(_train_one(store, settings, trainer, info, now))
    return results
