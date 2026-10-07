"""Trains the support heads on all the templates and with some left out, and prints how they
generalize: the share answered locally, how often that is right and what each novelty gate stops."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from generate import (
    CATEGORIES,
    CHANNELS,
    DEFAULT_ROWS,
    DEFAULT_SEED,
    PLANS,
    SITES,
    TEMPLATES,
    URGENCY_LEVELS,
    Ticket,
    state_text,
    tickets,
    write_rows,
)

from stuntd.cli import main as stuntd
from stuntd.jev.state import serialize_state
from stuntd.settings import Settings, config_path, database_path, load_settings, models_path
from stuntd.store.db import Store
from stuntd.store.redact import Redactor
from stuntd.train.artifacts import HEAD_FILE, load_model, site_dir
from stuntd.train.dataset import build_dataset
from stuntd.train.metrics import quantile

if TYPE_CHECKING:
    from generate import _Template

CHECK_SEED = DEFAULT_SEED + 10_000
CHECKED = 1000
FRESH_MARGIN = 2
QUANTILES = (0.95, 0.98, 0.99)
EPOCHS = 24
CACHE_MAX_MB = 2048
DATA_DIR_VARIABLE = "STUNTD_DATA_DIR"
PROBES = ("what's the weather in Paris", "asdf qwer zxcv", {})

_IMPORT_FLAGS = {
    "category": ("--labels", ",".join(CATEGORIES)),
    "urgency": ("--labels", ",".join(str(level) for level in range(URGENCY_LEVELS))),
    "needs_human": ("--kind", "boolean"),
}
_NO_GATE = "no gate"

Cutoffs = dict[str, float | None]


@dataclass(frozen=True)
class Reading:
    """What one head made of one request: its answer, whether it is sure and how unfamiliar the
    request is."""

    answer: str
    confident: bool
    novelty: float


@dataclass(frozen=True)
class Tally:
    """Tickets checked, how many all three heads answered locally and how many of those were
    right on every question."""

    total: int
    local: int
    right: int


def held_out() -> tuple[_Template, ...]:
    """The first template of each category."""
    return tuple(
        next(template for template in TEMPLATES if template.category == category)
        for category in CATEGORIES
    )


def unseen(trained: set[str], templates: tuple[_Template, ...], count: int) -> list[Ticket]:
    """Count tickets from the templates whose states the heads were not trained on."""
    drawn = [
        ticket
        for ticket in tickets(CHECK_SEED, count * FRESH_MARGIN, templates)
        if state_text(ticket) not in trained
    ]
    if len(drawn) < count:
        raise RuntimeError(f"only {len(drawn)} unseen tickets at seed {CHECK_SEED}")
    return drawn[:count]


def gates(novelties: dict[str, list[float]]) -> dict[str, Cutoffs]:
    """The cut-off of each site under no gate and under each quantile of its holdout novelty."""
    return {_NO_GATE: dict.fromkeys(SITES)} | {
        f"{level} quantile": {site: quantile(novelties[site], level) for site in SITES}
        for level in QUANTILES
    }


def _sure(reading: Reading, cutoff: float | None) -> bool:
    return reading.confident and (cutoff is None or reading.novelty <= cutoff)


def tally(drawn: list[Ticket], readings: dict[str, list[Reading]], cutoffs: Cutoffs) -> Tally:
    """Counts the tickets all three heads would answer locally, and the ones they get right."""
    local = right = 0
    for index, ticket in enumerate(drawn):
        if not all(_sure(readings[site][index], cutoffs[site]) for site in SITES):
            continue
        local += 1
        right += all(
            readings[site][index].answer == teacher(ticket) for site, teacher in SITES.items()
        )
    return Tally(len(drawn), local, right)


def stopped(readings: dict[str, list[Reading]], index: int, cutoffs: Cutoffs) -> bool:
    """Whether no head answers the request at that position."""
    return not any(_sure(readings[site][index], cutoffs[site]) for site in SITES)


def swapped(ticket: Ticket) -> Ticket:
    """The same ticket arriving by the next channel from a customer on the next plan."""
    return replace(
        ticket,
        channel=CHANNELS[(CHANNELS.index(ticket.channel) + 1) % len(CHANNELS)],
        plan=PLANS[(PLANS.index(ticket.plan) + 1) % len(PLANS)],
    )


def changes(before: dict[str, list[Reading]], after: dict[str, list[Reading]]) -> dict[str, int]:
    """How many answers of each site differ between two readings of the same tickets."""
    return {
        site: sum(
            one.answer != other.answer for one, other in zip(before[site], after[site], strict=True)
        )
        for site in SITES
    }


def _share(hits: int, total: int) -> str:
    return "-" if total == 0 else f"{100 * hits / total:.1f}%"


def render_tallies(title: str, tallies: dict[str, Tally]) -> str:
    """One table of the share answered locally and the share right, a row per gate."""
    lines = [title, f"  {'gate':<15}{'answered locally':>18}{'all three right when local':>28}"]
    lines += [
        f"  {name:<15}{_share(one.local, one.total):>18}{_share(one.right, one.local):>28}"
        for name, one in tallies.items()
    ]
    return "\n".join(lines)


def render_probes(stops: dict[str, dict[str, bool]]) -> str:
    """Whether each probe is stopped under each gate."""
    names = list(next(iter(stops.values())))
    width = max(len(state) for state in stops)
    lines = [
        "junk and out-of-scope states, stopped if no head answers them",
        f"  {'state':<{width}}  " + "  ".join(names),
    ]
    lines += [
        f"  {state:<{width}}  "
        + "  ".join(f"{'yes' if row[name] else 'no':<{len(name)}}" for name in names).rstrip()
        for state, row in stops.items()
    ]
    return "\n".join(lines)


def render_changes(counts: dict[str, int], total: int) -> str:
    """The share of tickets whose answer changes when only the channel and the plan do."""
    shares = ", ".join(f"{site} on {_share(count, total)}" for site, count in counts.items())
    return f"changing only channel and plan changes the answer of {shares}"


class _Heads:
    """The three trained heads of one data directory, ready to be asked."""

    def __init__(self, directory: Path) -> None:
        # Torch is reached here and nowhere else, so everything above imports without it.
        from stuntd.serve.decider import Decider

        os.environ[DATA_DIR_VARIABLE] = str(directory)
        settings = load_settings(config_path())
        self._models_dir = models_path(settings)
        self._redactor = Redactor(settings.redaction_patterns, builtin=settings.redact)
        self._models = {site: load_model(self._models_dir, site) for site in SITES}
        self._decider = Decider(settings.base_model, settings.device)
        self.gates = gates(self._holdout_novelties(settings))

    def __repr__(self) -> str:
        return f"_Heads(models={str(self._models_dir)!r})"

    def _holdout_novelties(self, settings: Settings) -> dict[str, list[float]]:
        store = Store(database_path(settings), self._redactor)
        try:
            infos = {info.site: info for info in store.sites()}
            holdouts = {
                site: build_dataset(
                    site,
                    infos[site].kind,
                    infos[site].schema_canonical,
                    store.examples(site),
                    settings.min_examples,
                    settings.holdout,
                ).holdout
                for site in SITES
            }
        finally:
            store.close()
        return {
            site: [reading.novelty for reading in self.read(site, [i.text for i in holdout])]
            for site, holdout in holdouts.items()
        }

    def read(self, site: str, texts: list[str]) -> list[Reading]:
        """What the site's head answers to each text."""
        model = self._models[site]
        head = site_dir(self._models_dir, site) / HEAD_FILE
        readings = []
        for text in texts:
            verdict = self._decider.decide(model, head, self._redactor.apply(text))
            if verdict.novelty is None:
                raise RuntimeError(f"the head of {site} kept no embeddings to measure novelty by")
            confident = model.threshold is not None and verdict.confidence >= model.threshold
            readings.append(Reading(model.labels[verdict.label], confident, verdict.novelty))
        return readings

    def read_all(self, texts: list[str]) -> dict[str, list[Reading]]:
        """What every head answers to each text."""
        return {site: self.read(site, texts) for site in SITES}

    def table(self, title: str, drawn: list[Ticket]) -> str:
        """The tally of the tickets under every gate."""
        readings = self.read_all([state_text(ticket) for ticket in drawn])
        return render_tallies(
            title, {name: tally(drawn, readings, cutoffs) for name, cutoffs in self.gates.items()}
        )


def _stuntd(*argv: str) -> None:
    code = stuntd(list(argv))
    if code:
        raise RuntimeError(f"stuntd {' '.join(argv)} exited with {code}")


def _train(directory: Path, base_model: str, drawn: list[Ticket]) -> set[str]:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stuntd.toml").write_text(
        f"[training]\nbase_model = {json.dumps(base_model)}\nepochs = {EPOCHS}\n"
        f"cache_max_mb = {CACHE_MAX_MB}\n",
        encoding="utf-8",
    )
    os.environ[DATA_DIR_VARIABLE] = str(directory)
    rows = directory / "rows"
    write_rows(drawn, rows)
    for site, flags in _IMPORT_FLAGS.items():
        _stuntd("import", site, str(rows / f"{site}.jsonl"), *flags)
    _stuntd("train")
    return {state_text(ticket) for ticket in drawn}


def _all_templates(root: Path, base_model: str) -> None:
    drawn = tickets(DEFAULT_SEED, DEFAULT_ROWS)
    trained = _train(root / "all", base_model, drawn)
    heads = _Heads(root / "all")
    checked = unseen(trained, TEMPLATES, CHECKED)
    readings = heads.read_all([state_text(ticket) for ticket in checked])
    print(heads.table(f"heads trained on all {len(TEMPLATES)} templates", checked))
    swaps = heads.read_all([state_text(swapped(ticket)) for ticket in checked])
    print(render_changes(changes(readings, swaps), len(checked)))
    probes = heads.read_all([serialize_state(probe) for probe in PROBES])
    print(
        render_probes(
            {
                serialize_state(probe): {
                    name: stopped(probes, index, cutoffs) for name, cutoffs in heads.gates.items()
                }
                for index, probe in enumerate(PROBES)
            }
        )
    )


def _left_out_templates(root: Path, base_model: str) -> None:
    left_out = held_out()
    kept = tuple(template for template in TEMPLATES if template not in left_out)
    trained = _train(root / "held-out", base_model, tickets(DEFAULT_SEED, DEFAULT_ROWS, kept))
    heads = _Heads(root / "held-out")
    print(heads.table(f"heads trained on {len(kept)} templates", unseen(trained, kept, CHECKED)))
    print(heads.table("tickets from the left-out templates", unseen(trained, left_out, CHECKED)))


def main(argv: list[str] | None = None) -> int:
    """Trains the heads twice and prints what they do with tickets they were not trained on."""
    args = _parser().parse_args(argv)
    root = Path(args.data_dir)
    _all_templates(root, args.base_model)
    _left_out_templates(root, args.base_model)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="check-data",
        metavar="DIR",
        help="where the two stuntd data directories are made; start from an empty one",
    )
    parser.add_argument(
        "--base-model",
        default=Settings().base_model,
        metavar="M",
        help="the Laya checkpoint, a local directory or a model name",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
