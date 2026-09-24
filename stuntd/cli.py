from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

from stuntd.decisions.schema import DecisionSchema, detect_schema
from stuntd.paths import private_file
from stuntd.serve.modes import (
    MODE_CHECK,
    MODE_COLLECT,
    MODE_LIVE,
    MODE_SHADOW,
    SiteState,
    site_states,
    write_mode,
)
from stuntd.serve.monitor import window_agreement, window_start
from stuntd.settings import (
    CONFIG_TEMPLATE,
    Settings,
    config_path,
    database_path,
    load_settings,
    models_path,
)
from stuntd.train.artifacts import HEAD_FILE, SiteModel, list_models, load_model, site_dir
from stuntd.train.report import render, render_json
from stuntd.train.run import NO_CAPTURES, Trainer, TrainResult, train_sites

if TYPE_CHECKING:
    from stuntd.serve.runtime import DeciderLike
    from stuntd.store.db import Capture, Store

__all__ = ["main"]

_COLUMN_WIDTHS = (18, 8, 9, 8, 6)
_CONFIG_HELP = "settings file to read instead of the one in the data directory"
_IMPORT_MODEL = "import"
_COVERAGE_DIGITS = 2
# Below this share of the holdout a head answers so little that the operator should hear about it.
_LOW_COVERAGE = 0.10
_KINDS = ("choice", "boolean", "number")
_BOOLEAN_ANSWERS = ("true", "false")


@dataclass(frozen=True)
class _SiteStatus:
    """One row of the status table: how a site serves and what it has answered lately."""

    site: str
    mode: str
    captures: int
    shadow: int
    live: int
    agreement: float | None


class _LearningOff(Exception):
    """A command that would write captures, models or decisions while learning is off."""


def _require_learning(config: str | None, settings: Settings) -> None:
    if not settings.learn:
        raise _LearningOff(f"learning is off in {_config_file(config)}")


def _make_decider(settings: Settings) -> DeciderLike:
    # The only place serving reaches torch and laya, so every other command runs without the extra.
    from stuntd.serve.decider import Decider

    return Decider(settings.base_model, settings.device)


def _require_serving() -> None:
    """Refuses the way serve would when the train extra is missing, without loading a model."""
    try:
        import stuntd.serve.decider  # noqa: F401
    except ImportError as exc:
        raise ValueError('serving needs the train extra: pip install "stuntd[train]"') from exc


def _serve_decider(settings: Settings, serving: int) -> DeciderLike | None:
    # A local Jev answers its questions zero-shot from the base checkpoint, trained head or not,
    # so it needs the decider even where no site serves one of its own.
    local_jev = settings.jev_upstream == ""
    if serving == 0 and not local_jev:
        return None
    try:
        return _make_decider(settings)
    except ImportError:
        # The proxy still records captures, so a missing extra costs the answers, not the run.
        print('stuntd: serving disabled: pip install "stuntd[train]"', file=sys.stderr)
        return None


def _serving_sites(models: Path) -> list[tuple[str, SiteModel]]:
    return [
        (state.site, state.model)
        for state in site_states(models)
        if state.mode != MODE_COLLECT and state.model is not None
    ]


def _serve(args: argparse.Namespace) -> int:
    overrides: dict[str, object] = {"upstream": args.upstream, "port": args.port}
    if args.lazy:
        overrides["lazy_load"] = True
    settings = load_settings(_config_file(args.config), overrides)
    import uvicorn

    from stuntd.proxy.app import build_app

    models = models_path(settings)
    # With learning off no head answers, so the sites on disk serve nothing.
    serving = _serving_sites(models) if settings.learn else []
    lazy = getattr(settings, "lazy_load", False)
    decider = None
    if not lazy:
        # The base model loads before anything is printed: it takes seconds, and a checkpoint that
        # will not load must fail the command rather than leave a line promising a proxy that is up.
        decider = _serve_decider(settings, len(serving))
        if decider is not None and serving:
            # The first pass through a freshly loaded model is far slower than the rest, so one site
            # pays for it here rather than the first caller of whichever site asks first.
            site, model = serving[0]
            decider.warm(model, site_dir(models, site) / HEAD_FILE)
        elif decider is not None:
            decider.warm_base()
    # uvicorn.run never returns while the daemon is up, so a piped stdout needs the lines now.
    listening = f"stuntd listening on http://{settings.host}:{settings.port}"
    if settings.upstream:
        print(f"{listening} -> {settings.upstream}", flush=True)
    else:
        print(listening, flush=True)
        print("no upstream: only the Jev routes are served", flush=True)
    if decider is not None or lazy:
        served = f"{len(serving)} site(s)" if serving else "jev locally"
        lazy_suffix = " (lazy)" if lazy else ""
        print(f"serving {served} with {settings.base_model}{lazy_suffix}", flush=True)
    if not settings.learn:
        print("learning off", flush=True)
    # The relay is byte-exact, so uvicorn must not put its own Date and Server headers next to
    # the ones the provider sent.
    uvicorn.run(
        build_app(settings, decider=decider),
        host=settings.host,
        port=settings.port,
        log_level="warning",
        server_header=False,
        date_header=False,
    )
    return 0


def _enable(args: argparse.Namespace) -> int:
    settings = _load(args.config, None, None)
    _require_learning(args.config, settings)
    models = models_path(settings)
    site = _site_name(args.site)
    model = _one_model(models, site)
    if model.threshold is None:
        print(
            f"stuntd: {site} never reached the target agreement on its holdout;"
            " retrain with more examples",
            file=sys.stderr,
        )
        return 1
    # The curve always answers at least one holdout row, so a useless operating point shows up
    # as the coverage the report rounds to, not as a bare zero.
    coverage = None if model.coverage is None else round(model.coverage, _COVERAGE_DIGITS)
    if coverage == 0:
        print(
            f"stuntd: {site} covers 0% of the holdout at target"
            f" {model.target_agreement:.2f}: nothing will be answered locally;"
            " lower training.target_agreement or retrain",
            file=sys.stderr,
        )
        return 1
    if coverage is not None and coverage < _LOW_COVERAGE:
        print(
            f"stuntd: {site} covers {coverage:.0%} of the holdout at target"
            f" {model.target_agreement:.2f}; most answers will still go to the provider",
            file=sys.stderr,
        )
    _require_serving()
    # The proxy re-reads the file once it changes, so the running daemon needs no restart.
    write_mode(site_dir(models, site), MODE_LIVE, time.time())
    print(f"{site}: {MODE_LIVE}")
    return 0


def _disable(args: argparse.Namespace) -> int:
    settings = _load(args.config, None, None)
    _require_learning(args.config, settings)
    models = models_path(settings)
    site = _site_name(args.site)
    _one_model(models, site)
    write_mode(site_dir(models, site), MODE_SHADOW, time.time())
    print(f"{site}: {MODE_SHADOW}")
    return 0


def _status(args: argparse.Namespace) -> int:
    settings = _load(args.config, None, None)
    rows = _status_rows(settings) if settings.learn else []
    if args.json:
        print(json.dumps({"learning": settings.learn, "sites": [asdict(row) for row in rows]}))
        return 0
    if not settings.learn:
        print("learning off")
        return 0
    if not rows:
        print("no captures yet")
        return 0
    print(_row("site", "mode", "captures", "shadow", "live", "agreement"))
    for row in rows:
        print(_status_line(row))
    return 0


def _config_init(args: argparse.Namespace) -> int:
    path = Path(args.config) if args.config else config_path()
    if path.exists() and not args.force:
        print(f"{path} exists; use --force to overwrite")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    private_file(path).write_text(CONFIG_TEMPLATE, encoding="utf-8")
    print(f"wrote {path}")
    return 0


def _config_show(args: argparse.Namespace) -> int:
    settings = _load(args.config, args.upstream, args.port)
    flagged = {name for name in ("upstream", "port") if getattr(args, name) is not None}
    defaults = Settings()
    for item in fields(Settings):
        value = getattr(settings, item.name)
        if item.name in flagged:
            source = "flag"
        elif value != getattr(defaults, item.name):
            source = "file"
        else:
            source = "default"
        if item.name == "redaction_patterns":
            shown = ", ".join(settings.redaction_patterns)
        else:
            shown = "" if value is None else str(value)
        print(f"{item.name} = {shown}  ({source})")
    return 0


def _make_trainer(settings: Settings) -> Trainer:
    # The only place torch and laya are reached, so every other command runs without the extra.
    from stuntd.train.trainer import LayaTrainer

    return LayaTrainer(
        settings.base_model,
        settings.device,
        settings.epochs,
        cache_encoder=settings.cache_encoder,
        cache_max_bytes=settings.cache_max_mb * 2**20,
    )


def _load_trainer(settings: Settings) -> Trainer:
    try:
        return _make_trainer(settings)
    except ImportError as exc:
        raise ValueError('training needs the train extra: pip install "stuntd[train]"') from exc


def _base_model_line(settings: Settings) -> str:
    # A checkpoint already on disk is read from there, so only a model name is fetched.
    if Path(settings.base_model).is_dir():
        return f"base model: {settings.base_model}"
    return f"base model: {settings.base_model} (downloaded on first use)"


def _result_line(result: TrainResult) -> str:
    model = result.model
    if model is None:
        return f"{result.site}  skipped  {result.reason}"
    headline = f"{result.site}  trained  holdout agreement {model.agreement:.3f}"
    if model.coverage is None or model.covered_agreement is None:
        return f"{headline}  coverage -  ece {model.ece:.3f}"
    return (
        f"{headline}  coverage {model.coverage:.2f}"
        f" at agreement {model.covered_agreement:.3f}  ece {model.ece:.3f}"
    )


def _train(args: argparse.Namespace) -> int:
    from stuntd.store.db import Store
    from stuntd.store.redact import Redactor

    settings = _load(args.config, None, None)
    _require_learning(args.config, settings)
    database = database_path(settings)
    if not database.exists():
        print("no captures yet")
        return 0
    store = Store(database, Redactor())
    try:
        known = {info.site for info in store.sites()}
        sites = [_site_name(site) for site in args.sites]
        if not known and not sites:
            print("no captures yet")
            return 0
        trainable = [site for site in sites if site in known] if sites else list(known)
        if not trainable:
            results = [TrainResult(site, None, NO_CAPTURES) for site in sites]
        else:
            # Building the trainer fetches the checkpoint, which is a long silent wait without
            # the line, so it waits until a site actually needs training.
            print(_base_model_line(settings), flush=True)
            results = train_sites(store, settings, _load_trainer(settings), sites)
    finally:
        store.close()
    for result in results:
        print(_result_line(result))
    return 0


class _InvalidLine(Exception):
    """A row of an import file that cannot be recorded, already named by its line number."""


def _decision_schema(schema: object) -> DecisionSchema:
    # Read back the way a request is, so an imported site carries the canonical schema and the
    # kind the same decision would have got through the proxy.
    found = detect_schema(
        {"response_format": {"type": "json_schema", "json_schema": {"schema": schema}}}
    )
    if found is None:
        raise ValueError("schema must describe an object with one typed field")
    return found


def _built_schema(site: str, kind: str, labels: str | None) -> dict[str, object]:
    if kind != "choice":
        spec: dict[str, object] = {"type": "boolean" if kind == "boolean" else "integer"}
    else:
        # A label repeated in the list would become a class no example can carry.
        options = dict.fromkeys(
            label.strip() for label in (labels or "").split(",") if label.strip()
        )
        if not options:
            raise ValueError("a choice site needs --labels a,b,c or --schema")
        spec = {"type": "string", "enum": list(options)}
    # The field is what the head is asked for at serve time, so a site built here reads the same
    # way a Jev question of that name does rather than asking the model to "choose answer".
    return {"type": "object", "properties": {site: spec}}


def _import_schema(
    site: str, kind: str | None, raw: str | None, labels: str | None
) -> DecisionSchema:
    if raw is None:
        return _decision_schema(_built_schema(site, kind or "choice", labels))
    try:
        given = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--schema is not valid JSON: {exc.msg}") from exc
    schema = _decision_schema(given)
    if kind is not None and kind != schema.kind:
        raise ValueError(f"--schema describes a {schema.kind} field, not {kind}")
    return schema


def _check_answer(schema: DecisionSchema, answer: str) -> None:
    if schema.kind == "choice":
        if answer not in schema.options:
            raise ValueError(f"answer {answer!r} not in labels")
    elif schema.kind == "boolean":
        if answer not in _BOOLEAN_ANSWERS:
            raise ValueError(f"answer {answer!r} is not true or false")
    else:
        try:
            value = float(answer)
        except ValueError as exc:
            raise ValueError(f"answer {answer!r} is not a number") from exc
        # An infinity or a NaN parses and is then dropped when the dataset is built, which would
        # cost rows the import reported as recorded.
        if not math.isfinite(value):
            raise ValueError(f"answer {answer!r} is not a finite number")


def _labelled_row(line: str, schema: DecisionSchema) -> tuple[str, str]:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(row, dict):
        raise ValueError("expected an object with text and answer")
    text, answer = row.get("text"), row.get("answer")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if not isinstance(answer, str):
        raise ValueError("answer must be a string")
    _check_answer(schema, answer)
    return text, answer


def _import_captures(path: Path, site: str, schema: DecisionSchema) -> list[Capture]:
    from stuntd.store.db import Capture

    captures = []
    # utf-8-sig so a file written by a Windows editor is not read with its BOM glued to line 1.
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            text, answer = _labelled_row(line, schema)
        except ValueError as exc:
            raise _InvalidLine(f"line {number}: {exc}") from exc
        captures.append(
            Capture(site, schema.canonical, schema.kind, text, answer, _IMPORT_MODEL, 0, None, None)
        )
    return captures


def _import(args: argparse.Namespace) -> int:
    from stuntd.store.db import Store
    from stuntd.store.redact import Redactor

    settings = _load(args.config, None, None)
    _require_learning(args.config, settings)
    site = _site_name(args.site)
    # Nothing is trained here; the name only has to pass the whitelist it would be stored under.
    site_dir(models_path(settings), site)
    schema = _import_schema(site, args.kind, args.schema, args.labels)
    try:
        # The whole file is read before the store is opened, so a rejected row leaves no database.
        captures = _import_captures(Path(args.file), site, schema)
    except _InvalidLine as exc:
        print(exc, file=sys.stderr)
        return 1
    store = Store(
        database_path(settings),
        Redactor(settings.redaction_patterns, builtin=settings.redact),
        max_rows=settings.max_rows,
        max_age_days=settings.max_age_days,
    )
    # Importing the same file twice records its rows twice; training keeps the latest row per text.
    try:
        for capture in captures:
            store.record(capture)
    finally:
        store.close()
    print(f"imported {len(captures)} rows into {site}")
    return 0


def _site_name(site: str) -> str:
    # A Jev question namespaced with a colon lands in a folder named with a dot, so either
    # spelling of the name reaches the same site whichever command is given it.
    return site.replace(":", ".")


def _one_model(models: Path, site: str) -> SiteModel:
    try:
        return load_model(models, site)
    except FileNotFoundError as exc:
        raise ValueError(f"no model for {site}") from exc


def _report(args: argparse.Namespace) -> int:
    settings = _load(args.config, None, None)
    _require_learning(args.config, settings)
    models = models_path(settings)
    site = None if args.site is None else _site_name(args.site)
    found = list_models(models) if site is None else [_one_model(models, site)]
    if args.json:
        print(render_json(found))
        return 0
    if not found:
        print("no trained sites yet")
        return 0
    print("\n\n".join(render(model, args.curve) for model in found))
    return 0


def _row(
    site: object, mode: object, captures: object, shadow: object, live: object, agreement: object
) -> str:
    # The space past each pad keeps the columns apart when a value fills its width.
    columns = zip((site, mode, captures, shadow, live), _COLUMN_WIDTHS, strict=True)
    return "".join(f"{value:<{width - 1}} " for value, width in columns) + str(agreement)


def _status_line(row: _SiteStatus) -> str:
    # A site with no model has nothing to compare or to serve, which is not the same as none yet.
    served = row.mode != MODE_COLLECT
    return _row(
        row.site,
        row.mode,
        row.captures,
        row.shadow if served else "-",
        row.live if served else "-",
        "-" if row.agreement is None else f"{row.agreement:.3f}",
    )


def _compared(counts: dict[str, int]) -> int:
    # A checked live request is answered by both sides, so it counts as a comparison too.
    return counts.get(MODE_SHADOW, 0) + counts.get(MODE_CHECK, 0)


def _agreement(store: Store, site: str, state: SiteState | None, window: int) -> float | None:
    if state is None or state.model is None or state.model.threshold is None:
        return None
    rows = store.comparisons(site, window, window_start(state))
    return window_agreement(rows, state.model.threshold).agreement


def _row_without_captures(state: SiteState) -> _SiteStatus:
    return _SiteStatus(state.site, state.mode, 0, 0, 0, None)


def _status_rows(settings: Settings) -> list[_SiteStatus]:
    from stuntd.store.db import Store
    from stuntd.store.redact import Redactor

    states = {state.site: state for state in site_states(models_path(settings))}
    database = database_path(settings)
    if not database.exists():
        # Opening a store would create the database that status is only here to read.
        return [_row_without_captures(state) for state in states.values()]
    store = Store(database, Redactor())
    try:
        captures = {info.site: info.count for info in store.sites()}
        counts = store.decision_counts()
        rows = []
        for site in sorted(set(captures) | set(states)):
            state = states.get(site)
            decisions = counts.get(site, {})
            rows.append(
                _SiteStatus(
                    site=site,
                    mode=MODE_COLLECT if state is None else state.mode,
                    captures=captures.get(site, 0),
                    shadow=_compared(decisions),
                    live=decisions.get(MODE_LIVE, 0),
                    agreement=_agreement(store, site, state, settings.window),
                )
            )
        return rows
    finally:
        store.close()


def _config_file(explicit: str | None) -> Path:
    if explicit is None:
        return config_path()
    path = Path(explicit)
    if path.is_dir():
        raise ValueError(f"{path} is not a file")
    if not path.is_file():
        raise ValueError(f"{path} does not exist")
    return path


def _load(config: str | None, upstream: str | None, port: int | None) -> Settings:
    return load_settings(_config_file(config), {"upstream": upstream, "port": port})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stuntd", description="Local proxy that records typed LLM decisions."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run the proxy until it is interrupted")
    serve.add_argument("--config", metavar="PATH", help=_CONFIG_HELP)
    serve.add_argument("--upstream", metavar="URL", help="provider base URL to forward to")
    serve.add_argument("--port", type=int, metavar="N", help="port to listen on")
    serve.add_argument("--lazy", action="store_true", help="load the base checkpoint on first use")
    serve.set_defaults(handler=_serve)

    status = commands.add_parser("status", help="summarise the captures recorded so far")
    status.add_argument("--config", metavar="PATH", help=_CONFIG_HELP)
    status.add_argument("--json", action="store_true", help="print the rows as JSON")
    status.set_defaults(handler=_status)

    train = commands.add_parser("train", help="train a model for each site with enough captures")
    train.add_argument(
        "sites", nargs="*", metavar="SITE", help="sites to train, every known site by default"
    )
    train.add_argument("--config", metavar="PATH", help=_CONFIG_HELP)
    train.set_defaults(handler=_train)

    report = commands.add_parser("report", help="print how the trained models did")
    report.add_argument(
        "site", nargs="?", metavar="SITE", help="site to report on, every trained site by default"
    )
    report.add_argument("--config", metavar="PATH", help=_CONFIG_HELP)
    report.add_argument("--json", action="store_true", help="print the models as JSON")
    report.add_argument("--curve", action="store_true", help="add the whole threshold curve")
    report.set_defaults(handler=_report)

    enable = commands.add_parser("enable", help="let a trained site answer from its own model")
    enable.add_argument("site", metavar="SITE", help="site to serve locally")
    enable.add_argument("--config", metavar="PATH", help=_CONFIG_HELP)
    enable.set_defaults(handler=_enable)

    disable = commands.add_parser("disable", help="send a site back to the provider")
    disable.add_argument("site", metavar="SITE", help="site to stop serving locally")
    disable.add_argument("--config", metavar="PATH", help=_CONFIG_HELP)
    disable.set_defaults(handler=_disable)

    importer = commands.add_parser(
        "import",
        help="record labelled examples for a site",
        description=(
            "Record labelled examples for a site from a JSONL file. Each text is stored and"
            " served exactly as written, so it has to be spelled the way the site sees its"
            " input: for a chat site the 'user: ...' lines the store holds, for a Jev site the"
            " serialised state."
        ),
    )
    importer.add_argument("site", metavar="SITE", help="site to record the examples under")
    importer.add_argument(
        "file",
        metavar="FILE",
        help='JSONL file of {"text": ..., "answer": ...} rows, each text written the way the'
        " site sees its input: 'user: ...' lines for a chat site, the serialised state for Jev",
    )
    importer.add_argument("--kind", choices=_KINDS, help="what the answers are, choice by default")
    importer.add_argument("--schema", metavar="JSON", help="JSON schema of the answered field")
    importer.add_argument(
        "--labels", metavar="A,B,C", help="answers a choice site may carry, instead of --schema"
    )
    importer.add_argument("--config", metavar="PATH", help=_CONFIG_HELP)
    importer.set_defaults(handler=_import)

    config = commands.add_parser("config", help="create or inspect the settings file")
    config_commands = config.add_subparsers(dest="config_command", required=True)

    init = config_commands.add_parser("init", help="write a commented settings file")
    init.add_argument("--config", metavar="PATH", help=_CONFIG_HELP)
    init.add_argument("--force", action="store_true", help="overwrite an existing file")
    init.set_defaults(handler=_config_init)

    show = config_commands.add_parser("show", help="print every setting with its source")
    show.add_argument("--config", metavar="PATH", help=_CONFIG_HELP)
    show.add_argument("--upstream", metavar="URL", help="provider base URL to forward to")
    show.add_argument("--port", type=int, metavar="N", help="port to listen on")
    show.set_defaults(handler=_config_show)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parses one command line and runs the command it names, returning the exit code."""
    args = _build_parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except _LearningOff as exc:
        print(f"stuntd: {exc}", file=sys.stderr)
        return 1
    # A rejected setting, an unwritable path, a database that will not open and a failed training
    # run are all things the user has to fix, so they are reported rather than raised as a traceback.
    except (ValueError, OSError, RuntimeError, sqlite3.DatabaseError) as exc:
        print(f"stuntd: {exc}", file=sys.stderr)
        return 2