"""Trains every encoder on the same MASSIVE and Banking77 rows and prints how each one did."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from stuntd.serve.heads import Verdict
from stuntd.serve.runtime import is_novel
from stuntd.settings import LAYA_ENCODER, Settings, models_path
from stuntd.store.db import Capture
from stuntd.train.artifacts import SiteModel
from stuntd.train.metrics import quantile

if TYPE_CHECKING:
    from stuntd.store.db import Store

__all__ = [
    "Example",
    "Measured",
    "Result",
    "captures",
    "format_table",
    "latency",
    "massive_config",
    "measure",
    "stratified_sample",
]

LANGUAGES = ("ru", "de", "es", "zh", "ja", "en")
ENCODERS = (
    LAYA_ENCODER,
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-small",
    "sentence-transformers/all-MiniLM-L6-v2",
)
TRAIN_ROWS = 3000
SEED = 0
LATENCY_TEXTS = 200
CACHE_MAX_MB = 2048
LAYA_EPOCHS = 24
FIELD = "intent"

_MASSIVE = "mteb/amazon_massive_intent"
_MASSIVE_REVISION = "refs/convert/parquet"
_BANKING = "mteb/banking77"
_MASSIVE_CONFIGS = {"zh": "zh-CN"}


@dataclass(frozen=True)
class Example:
    """One labelled text of a benchmark dataset."""

    text: str
    label: str


@dataclass(frozen=True)
class Measured:
    """How a trained head did on a test split; coverage is None when the head has no threshold,
    and the accuracy at it also when no answer reached it."""

    accuracy: float
    coverage: float | None
    covered_accuracy: float | None
    local_share: float


@dataclass(frozen=True)
class Result:
    """One encoder on one dataset: its test-split numbers, training time and CPU latency."""

    dataset: str
    encoder: str
    measured: Measured
    train_seconds: float
    p50_ms: float
    p95_ms: float


def massive_config(language: str) -> str:
    """The MASSIVE configuration that holds a language."""
    return _MASSIVE_CONFIGS.get(language, language)


def stratified_sample(examples: Sequence[Example], size: int, seed: int) -> list[Example]:
    """Size examples in random order, each label in proportion to its share of examples."""
    rng = random.Random(seed)
    groups: dict[str, list[Example]] = {}
    for example in examples:
        groups.setdefault(example.label, []).append(example)
    if size >= len(examples):
        sample = list(examples)
    else:
        exact = {label: size * len(group) / len(examples) for label, group in groups.items()}
        quota = {label: int(share) for label, share in exact.items()}
        largest_remainder = sorted(groups, key=lambda label: (quota[label] - exact[label], label))
        for label in largest_remainder[: size - sum(quota.values())]:
            quota[label] += 1
        sample = [
            example
            for label in sorted(groups)
            for example in rng.sample(groups[label], quota[label])
        ]
    rng.shuffle(sample)
    return sample


def captures(site: str, labels: Sequence[str], examples: Sequence[Example]) -> list[Capture]:
    """The decisions a provider would have made for examples, as one choice site."""
    schema = {"type": "object", "properties": {FIELD: {"type": "string", "enum": list(labels)}}}
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return [
        Capture(site, canonical, "choice", example.text, example.label, "benchmark", 0, None, None)
        for example in examples
    ]


def measure(
    model: SiteModel, settings: Settings, verdicts: Sequence[Verdict], expected: Sequence[str]
) -> Measured:
    """Scores head verdicts against the true labels the way serving would act on them."""
    correct = [
        model.labels[verdict.label] == label
        for verdict, label in zip(verdicts, expected, strict=True)
    ]
    accuracy = sum(correct) / len(correct)
    if model.threshold is None:
        return Measured(accuracy, None, None, 0.0)
    sure = [verdict.confidence >= model.threshold for verdict in verdicts]
    local = sum(
        one and not is_novel(settings, model, verdict)
        for one, verdict in zip(sure, verdicts, strict=True)
    )
    right = sum(one and hit for one, hit in zip(sure, correct, strict=True))
    return Measured(
        accuracy,
        sum(sure) / len(sure),
        right / sum(sure) if any(sure) else None,
        local / len(verdicts),
    )


def latency(
    decide: Callable[[str], object],
    texts: Sequence[str],
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[float, float]:
    """Median and 95th percentile of the milliseconds one call of decide takes on each text."""
    durations = []
    for text in texts:
        started = clock()
        decide(text)
        durations.append((clock() - started) * 1000)
    return quantile(durations, 0.5), quantile(durations, 0.95)


def _percent(share: float | None) -> str:
    return "-" if share is None else f"{share:.1%}"


def format_table(results: Sequence[Result]) -> str:
    """The results as a Markdown table."""
    lines = [
        "| Dataset | Encoder | Accuracy | Coverage | Accuracy at coverage | Local"
        " | Train s | p50 ms | p95 ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        measured = result.measured
        lines.append(
            f"| {result.dataset} | {result.encoder} | {_percent(measured.accuracy)}"
            f" | {_percent(measured.coverage)} | {_percent(measured.covered_accuracy)}"
            f" | {_percent(measured.local_share)} | {result.train_seconds:.1f}"
            f" | {result.p50_ms:.1f} | {result.p95_ms:.1f} |"
        )
    return "\n".join(lines)


def _download(repo: str, filename: str, revision: str | None, data_dir: Path) -> list[Example]:
    from huggingface_hub import hf_hub_download
    from pyarrow import parquet

    path = hf_hub_download(
        repo,
        filename,
        repo_type="dataset",
        revision=revision,
        local_dir=data_dir / "downloads" / repo,
    )
    rows = parquet.read_table(path, columns=["text", "label_text"]).to_pylist()
    return [Example(row["text"], row["label_text"]) for row in rows]


def _run(store: Store, dataset: str, test: Sequence[Example], encoder: str, models: Path) -> Result:
    from stuntd.cli import _gold_deciders, _load_trainer
    from stuntd.train.artifacts import HEAD_FILE, site_dir
    from stuntd.train.run import train_sites

    settings = Settings(
        encoder=encoder, epochs=LAYA_EPOCHS, cache_max_mb=CACHE_MAX_MB, models_dir=models
    )
    trainer = _load_trainer(settings)
    started = time.perf_counter()
    (trained,) = train_sites(store, settings, trainer, [dataset])
    train_seconds = time.perf_counter() - started
    model = trained.model
    if model is None:
        raise RuntimeError(f"{dataset} with {encoder}: {trained.reason}")
    head = site_dir(models_path(settings), dataset) / HEAD_FILE
    decider = _gold_deciders(settings, model, False)[0]
    verdicts = [decider.decide(model, head, example.text) for example in test]
    on_cpu = _gold_deciders(replace(settings, device="cpu"), model, False)[0]
    on_cpu.warm(model, head)
    p50, p95 = latency(
        lambda text: on_cpu.decide(model, head, text),
        [example.text for example in test[:LATENCY_TEXTS]],
    )
    measured = measure(model, settings, verdicts, [example.label for example in test])
    return Result(dataset, encoder, measured, train_seconds, p50, p95)


def _benchmark(
    dataset: str,
    train: Sequence[Example],
    test: Sequence[Example],
    encoders: Sequence[str],
    data_dir: Path,
) -> list[Result]:
    from stuntd.store.db import Store
    from stuntd.store.redact import Redactor

    folder = data_dir / "runs" / dataset
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    store = Store(folder / "captures.sqlite", Redactor(builtin=False))
    labels = sorted({example.label for example in train})
    for capture in captures(dataset, labels, stratified_sample(train, TRAIN_ROWS, SEED)):
        store.record(capture)
    results = []
    for encoder in encoders:
        models = folder / "models" / encoder.replace("/", "--")
        results.append(_run(store, dataset, test, encoder, models))
        print(format_table(results[-1:]).splitlines()[-1], file=sys.stderr, flush=True)
    store.close()
    return results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="downloads and runs go here")
    parser.add_argument(
        "--languages", nargs="*", default=list(LANGUAGES), help="MASSIVE languages to run"
    )
    parser.add_argument("--encoders", nargs="+", default=list(ENCODERS), help="encoders to run")
    args = parser.parse_args(argv)
    results: list[Result] = []
    for language in args.languages:
        config = massive_config(language)
        train, test = (
            _download(_MASSIVE, f"{config}/{split}/0000.parquet", _MASSIVE_REVISION, args.data_dir)
            for split in ("train", "test")
        )
        results += _benchmark(f"massive-{language}", train, test, args.encoders, args.data_dir)
    train, test = (
        _download(_BANKING, f"data/{split}-00000-of-00001.parquet", None, args.data_dir)
        for split in ("train", "test")
    )
    results += _benchmark("banking77", train, test, args.encoders, args.data_dir)
    print(format_table(results))


if __name__ == "__main__":
    main()
