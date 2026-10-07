from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import signal
import sqlite3
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from stuntd.decisions.schema import DecisionSchema, detect_schema
from stuntd.jev.answer import answer_label
from stuntd.jev.schema import laya_question, typed_field
from stuntd.paths import data_dir, private_file, write_private
from stuntd.serve.modes import (
    MODE_CHECK,
    MODE_COLLECT,
    MODE_LIVE,
    MODE_SHADOW,
    SiteState,
    site_state,
    site_states,
    write_mode,
)
from stuntd.serve.monitor import window_agreement, window_start
from stuntd.serve.retrain import last_result
from stuntd.serve.runtime import is_novel
from stuntd.settings import (
    CONFIG_TEMPLATE,
    Settings,
    config_path,
    database_path,
    load_settings,
    models_path,
    setting_key,
)
from stuntd.train.artifacts import (
    HEAD_FILE,
    META_FILE,
    SiteModel,
    list_models,
    load_model,
    rename_model,
    site_dir,
)
from stuntd.train.gold import GoldRow, render_gold, render_gold_json, score_gold
from stuntd.train.report import render, render_json
from stuntd.train.run import NO_CAPTURES, Trainer, TrainResult, train_sites

if TYPE_CHECKING:
    from stuntd.serve.runtime import DeciderLike
    from stuntd.store.db import Capture, Store
    from stuntd.store.redact import Redactor

__all__ = ["main"]

_COLUMN_WIDTHS = (18, 8, 9, 8, 6, 10)
_CONFIG_HELP = "settings file to read instead of the one in the data directory"
_IMPORT_MODEL = "import"
_PID_FILE = "stuntd.pid"
_COVERAGE_DIGITS = 2
# Below this share of the holdout a head answers so little that the operator should hear about it.
_LOW_COVERAGE = 0.10
# A local Jev answers whatever the head does not with the base checkpoint, so past this share it
# is the checkpoint that mostly answers.
_LOW_LOCAL_COVERAGE = 0.50
_KINDS = ("choice", "boolean", "number")
_BOOLEAN_ANSWERS = ("true", "false")


@dataclass(frozen=True)
class _SiteStatus:
    """One row of the status table: how a site serves and what it has answered lately."""

    site: str
    name: str | None
    mode: str
    captures: int
    shadow: int
    live: int
    agreement: float | None
    retrain: str | None


class _LearningOff(Exception):
    """A command that would write captures, models or decisions while learning is off."""


def _require_learning(config: str | None, settings: Settings) -> None:
    if not settings.learn:
        raise _LearningOff(f"learning is off in {_config_file(config)}")


def _make_decider(settings: Settings) -> DeciderLike:
    # The only place serving reaches torch and laya, so every other command runs without the extra.
    from stuntd.serve.decider import Decider

    return Decider(settings.base_model, settings.device, settings.lazy_load)


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
    settings = _load(args.config, args.upstream, args.port)
    settings.lazy_load = settings.lazy_load or args.lazy
    import uvicorn

    from stuntd.proxy.app import build_app

    models = models_path(settings)
    # With learning off no head answers, so the sites on disk serve nothing.
    serving = _serving_sites(models) if settings.learn else []
    # The base model loads before anything is printed: it takes seconds, and a checkpoint that
    # will not load must fail the command rather than leave a line promising a proxy that is up.
    decider = _serve_decider(settings, len(serving))
    if decider is not None and not settings.lazy_load:
        if serving:
            # The first pass through a freshly loaded model is far slower than the rest, so one
            # site pays for it here rather than the first caller of whichever site asks first.
            site, model = serving[0]
            decider.warm(model, site_dir(models, site) / HEAD_FILE)
        else:
            decider.warm_base()
    # uvicorn.run never returns while the daemon is up, so a piped stdout needs the lines now.
    listening = f"stuntd listening on http://{settings.host}:{settings.port}"
    print(f"{listening} -> {settings.upstream}" if settings.upstream else listening, flush=True)
    jev = "jev proxy" if settings.jev_upstream else "jev local"
    endpoints = [*(["openai", "anthropic"] if settings.upstream else []), jev]
    print(f"endpoints: {', '.join(endpoints)}", flush=True)
    if decider is not None:
        served = f"{len(serving)} site(s)" if serving else "jev locally"
        loading = ", loading on first use" if settings.lazy_load else ""
        print(f"serving {served} with {settings.base_model}{loading}", flush=True)
    if not settings.learn:
        print("learning off", flush=True)
    pid_file = data_dir() / _PID_FILE
    write_private(pid_file, str(os.getpid()))
    try:
        # The relay is byte-exact, so uvicorn must not put its own Date and Server headers next
        # to the ones the provider sent.
        uvicorn.run(
            build_app(settings, decider=decider, config=_config_file(args.config)),
            host=settings.host,
            port=settings.port,
            log_level="warning",
            server_header=False,
            date_header=False,
        )
    finally:
        pid_file.unlink(missing_ok=True)
    return 0


def _stop(args: argparse.Namespace) -> int:
    pid_file = data_dir() / _PID_FILE
    if not pid_file.is_file():
        print("stuntd is not running", file=sys.stderr)
        return 1
    pid = int(pid_file.read_text(encoding="utf-8"))
    try:
        os.kill(pid, signal.SIGTERM)
    except PermissionError:
        raise
    except OSError:
        # Windows reports a dead pid as a plain OSError, not ProcessLookupError.
        pid_file.unlink(missing_ok=True)
        print("stuntd is not running", file=sys.stderr)
        return 1
    pid_file.unlink(missing_ok=True)
    print(f"stopped stuntd (pid {pid})")
    return 0


def _enable(args: argparse.Namespace) -> int:
    settings = _load(args.config, None, None)
    _require_learning(args.config, settings)
    models = models_path(settings)
    return max(_enable_site(settings, models, model) for model in _served_models(models, args.site))


def _enable_site(settings: Settings, models: Path, model: SiteModel) -> int:
    site = model.site
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
    local = not settings.upstream and not settings.jev_upstream
    if local:
        if (
            coverage is not None
            and coverage < _LOW_LOCAL_COVERAGE
            and settings.local_fallback == "zeroshot"
        ):
            print(
                f"stuntd: {site} covers {coverage:.0%} of the holdout at target"
                f" {model.target_agreement:.2f}; the rest is answered zero-shot by the base"
                " checkpoint",
                file=sys.stderr,
            )
    elif coverage is not None and coverage < _LOW_COVERAGE:
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
    for model in _served_models(models, args.site):
        write_mode(site_dir(models, model.site), MODE_SHADOW, time.time())
        print(f"{model.site}: {MODE_SHADOW}")
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
    print(_row("site", "mode", "captures", "shadow", "live", "agreement", "retrain"))
    for line in _status_lines(rows):
        print(line)
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
        print(f"{setting_key(item.name)} = {shown}  ({source})")
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
        max_option_tokens=settings.max_option_tokens,
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


@contextmanager
def _log_on_stderr() -> Iterator[None]:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("stuntd: %(message)s"))
    logger = logging.getLogger("stuntd")
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.setLevel(level)
        logger.removeHandler(handler)


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
            with _log_on_stderr():
                results = train_sites(store, settings, _load_trainer(settings), sites)
    finally:
        store.close()
    for result in results:
        print(_result_line(result))
    return 0


class _InvalidLine(Exception):
    """A row of a labelled JSONL file that cannot be used, already named by its line number."""


def _decision_schema(schema: object) -> DecisionSchema:
    # Read back the way a request is, so an imported site carries the canonical schema and the
    # kind the same decision would have got through the proxy.
    found = detect_schema(
        {"response_format": {"type": "json_schema", "json_schema": {"schema": schema}}}
    )
    if not isinstance(found, DecisionSchema):
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


def _labelled_row(line: str, schema: DecisionSchema | None) -> tuple[str, str]:
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
    if schema is not None:
        _check_answer(schema, answer)
    return text, answer


def _labelled_rows(path: Path, schema: DecisionSchema | None = None) -> list[tuple[str, str]]:
    rows = []
    # utf-8-sig so a file written by a Windows editor is not read with its BOM glued to line 1.
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(_labelled_row(line, schema))
        except ValueError as exc:
            raise _InvalidLine(f"line {number}: {exc}") from exc
    return rows


def _import_captures(path: Path, site: str, schema: DecisionSchema) -> list[Capture]:
    from stuntd.store.db import Capture

    return [
        Capture(site, schema.canonical, schema.kind, text, answer, _IMPORT_MODEL, 0, None, None)
        for text, answer in _labelled_rows(path, schema)
    ]


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
    redactor = Redactor(settings.redaction_patterns, builtin=settings.redact)
    database = database_path(settings)
    # A dry run reads the store only if there is one, so it never creates the database.
    store = (
        None
        if args.dry_run and not database.exists()
        else Store(
            database, redactor, max_rows=settings.max_rows, max_age_days=settings.max_age_days
        )
    )
    try:
        recorded = (
            set()
            if store is None
            else {(example.input_text, example.answer) for example in store.examples(site)}
        )
        new = []
        for capture in captures:
            key = (redactor.apply(capture.input_text), capture.answer)
            if key not in recorded:
                recorded.add(key)
                new.append(capture)
        if store is not None and not args.dry_run:
            for capture in new:
                store.record(capture)
    finally:
        if store is not None:
            store.close()
    skipped = len(captures) - len(new)
    if args.dry_run:
        print(f"would import {len(new)} rows into {site}, skip {skipped} already recorded")
    else:
        print(f"imported {len(new)} rows into {site}, skipped {skipped} already recorded")
    return 0


def _confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _family(store: Store, models: Path, site: str) -> list[str]:
    """The site and its field sites, as far as captures or a model folder know them."""
    known = {info.site for info in store.sites()}
    if models.is_dir():
        known.update(folder.name for folder in models.iterdir() if folder.is_dir())
    return sorted(name for name in known if name == site or name.startswith(f"{site}."))


def _is_live(models: Path, site: str) -> bool:
    try:
        return site_state(models, site).mode == MODE_LIVE
    except ValueError:
        return False


def _site_rm(args: argparse.Namespace) -> int:
    from stuntd.store.db import Store
    from stuntd.store.redact import Redactor

    settings = _load(args.config, None, None)
    _require_learning(args.config, settings)
    models = models_path(settings)
    site = _site_name(args.site)
    store = Store(database_path(settings), Redactor())
    try:
        family = _family(store, models, site)
        if not family:
            raise ValueError(f"no site {site}")
        served = next((name for name in family if _is_live(models, name)), None)
        if served is not None and not args.force:
            print(f"stuntd: {served} is served live; disable it or use --force", file=sys.stderr)
            return 1
        names = ", ".join(family)
        if not args.yes and not _confirm(f"remove {names} with its captures, decisions and model?"):
            print("stuntd: nothing removed", file=sys.stderr)
            return 1
        store.forget_site(site)
    finally:
        store.close()
    for name in family:
        if site_dir(models, name).is_dir():
            shutil.rmtree(site_dir(models, name))
    print(f"removed {names}")
    return 0


def _site_rename(args: argparse.Namespace) -> int:
    from stuntd.store.db import Store
    from stuntd.store.redact import Redactor

    settings = _load(args.config, None, None)
    _require_learning(args.config, settings)
    models = models_path(settings)
    old, new = _site_name(args.old), _site_name(args.new)
    site_dir(models, new)
    store = Store(database_path(settings), Redactor())
    try:
        family = _family(store, models, old)
        if not family:
            raise ValueError(f"no site {old}")
        if _family(store, models, new):
            raise ValueError(f"site {new} already exists")
        moved = [name for name in family if site_dir(models, name).is_dir()]
        for name in moved:
            rename_model(models, name, new + name.removeprefix(old))
        try:
            store.rename_site(old, new)
        except Exception:
            for name in moved:
                rename_model(models, new + name.removeprefix(old), name)
            raise
    finally:
        store.close()
    print(f"renamed {old} to {new}")
    return 0


def _site_name(site: str) -> str:
    # A Jev question namespaced with a colon lands in a folder named with a dot, so either
    # spelling of the name reaches the same site whichever command is given it.
    return site.replace(":", ".")


def _served_models(models: Path, name: str) -> list[SiteModel]:
    """The model of a site, or, when the name is only the parent of field sites, theirs."""
    site = _site_name(name)
    if (site_dir(models, site) / META_FILE).is_file():
        return [load_model(models, site)]
    fields_of_parent = [model for model in list_models(models) if model.site.startswith(f"{site}.")]
    if not fields_of_parent:
        raise ValueError(f"no model for {site}")
    return fields_of_parent


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
    if args.gold is not None:
        if site is None:
            raise ValueError("report --gold needs the SITE to score")
        if args.curve:
            raise ValueError("--curve and --gold do not combine")
        return _gold_report(settings, site, Path(args.gold), args.json)
    found = list_models(models) if site is None else [_one_model(models, site)]
    if args.json:
        print(render_json(found))
        return 0
    if not found:
        print("no trained sites yet")
        return 0
    print("\n\n".join(render(model, args.curve) for model in found))
    return 0


def _stored(settings: Settings, site: str, redactor: Redactor) -> tuple[dict[str, str], str | None]:
    from stuntd.store.db import Store

    database = database_path(settings)
    if not database.exists():
        return {}, None
    store = Store(database, redactor)
    try:
        # Examples come oldest first, so the latest capture of a text is the one left standing.
        teacher = {example.input_text: example.answer for example in store.examples(site)}
        schema = next((info.schema_canonical for info in store.sites() if info.site == site), None)
        return teacher, schema
    finally:
        store.close()


def _gold_rows(
    settings: Settings, model: SiteModel, usable: list[tuple[str, str]]
) -> tuple[list[GoldRow], str | None]:
    from stuntd.store.redact import Redactor

    try:
        decider = _make_decider(settings)
    except ImportError as exc:
        raise ValueError(
            'report --gold needs the train extra: pip install "stuntd[train]"'
        ) from exc
    # Captures are stored redacted, so a gold text is matched and asked in its redacted form.
    redactor = Redactor(settings.redaction_patterns, builtin=settings.redact)
    teacher, schema = _stored(settings, model.site, redactor)
    local_schema = None
    if not settings.jev_upstream and schema is not None and typed_field(model.site, schema):
        local_schema = schema
    head_path = site_dir(models_path(settings), model.site) / HEAD_FILE
    rows = []
    for text, answer in usable:
        redacted = redactor.apply(text)
        verdict = decider.decide(model, head_path, redacted)
        zero_shot = None
        if local_schema is not None and settings.local_fallback == "zeroshot":
            # laya answers in its own untyped shape, which the decider passes through as object.
            try:
                reply: Any = decider.answer(redacted, {model.site: laya_question(local_schema)})
            except RuntimeError as exc:
                print(f"stuntd: zero-shot failed: {exc}", file=sys.stderr)
            else:
                zero_shot = answer_label(reply["answers"][model.site])
        rows.append(
            GoldRow(
                answer,
                model.labels[verdict.label],
                verdict.confidence,
                teacher.get(redacted),
                is_novel(settings, model, verdict),
                zero_shot,
            )
        )
    return rows, settings.local_fallback if local_schema is not None else None


def _gold_report(settings: Settings, site: str, path: Path, as_json: bool) -> int:
    model = _one_model(models_path(settings), site)
    try:
        gold = _labelled_rows(path)
    except _InvalidLine as exc:
        print(exc, file=sys.stderr)
        return 1
    usable = [(text, answer) for text, answer in gold if answer in model.labels]
    rows, fallback = _gold_rows(settings, model, usable) if usable else ([], None)
    score = score_gold(rows, model.threshold, len(gold) - len(usable), fallback)
    print(render_gold_json(model, score) if as_json else render_gold(model, score))
    return 0


def _row(
    site: object,
    mode: object,
    captures: object,
    shadow: object,
    live: object,
    agreement: object,
    retrain: object,
) -> str:
    # The space past each pad keeps the columns apart when a value fills its width.
    columns = zip((site, mode, captures, shadow, live, agreement), _COLUMN_WIDTHS, strict=True)
    return "".join(f"{value:<{width - 1}} " for value, width in columns) + str(retrain)


def _label(site: str, name: str | None) -> str:
    return f"{site} ({name})" if name else site


def _status_line(row: _SiteStatus, label: str) -> str:
    # A site with no model has nothing to compare or to serve, which is not the same as none yet.
    served = row.mode != MODE_COLLECT
    return _row(
        label,
        row.mode,
        row.captures,
        row.shadow if served else "-",
        row.live if served else "-",
        "-" if row.agreement is None else f"{row.agreement:.3f}",
        row.retrain or "-",
    )


def _status_lines(rows: list[_SiteStatus]) -> list[str]:
    groups: dict[str, list[_SiteStatus]] = {}
    for row in rows:
        groups.setdefault(row.site.partition(".")[0], []).append(row)
    lines = []
    for parent, members in groups.items():
        own = members[0] if members[0].site == parent else None
        fields_of_parent = members[1:] if own else members
        if own:
            lines.append(_status_line(own, _label(parent, own.name)))
        else:
            lines.append(_row(_label(parent, members[0].name), "", "", "", "", "", "").rstrip())
        lines.extend(
            _status_line(row, f"  {row.site.removeprefix(parent)}") for row in fields_of_parent
        )
    return lines


def _compared(counts: dict[str, int]) -> int:
    # A checked live request is answered by both sides, so it counts as a comparison too.
    return counts.get(MODE_SHADOW, 0) + counts.get(MODE_CHECK, 0)


def _agreement(store: Store, site: str, state: SiteState | None, window: int) -> float | None:
    if state is None or state.model is None or state.model.threshold is None:
        return None
    rows = store.comparisons(site, window, window_start(state))
    return window_agreement(rows, state.model.threshold).agreement


def _mode_label(state: SiteState | None) -> str:
    if state is None:
        return MODE_COLLECT
    return f"{state.mode} (retrained, was live)" if state.was_live else state.mode


def _row_without_captures(state: SiteState) -> _SiteStatus:
    return _SiteStatus(state.site, None, _mode_label(state), 0, 0, 0, None, last_result(state.site))


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
        infos = {info.site: info for info in store.sites()}
        counts = store.decision_counts()
        rows = []
        for site in sorted(set(infos) | set(states)):
            state = states.get(site)
            decisions = counts.get(site, {})
            info = infos.get(site)
            rows.append(
                _SiteStatus(
                    site=site,
                    name=None if info is None else info.name,
                    mode=_mode_label(state),
                    captures=0 if info is None else info.count,
                    shadow=_compared(decisions),
                    live=decisions.get(MODE_LIVE, 0),
                    agreement=_agreement(store, site, state, settings.window),
                    retrain=last_result(site),
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
    serve.add_argument(
        "--lazy", action="store_true", help="load the base checkpoint on the first request"
    )
    serve.set_defaults(handler=_serve)

    stop = commands.add_parser("stop", help="stop the daemon started by serve")
    stop.set_defaults(handler=_stop)

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
    report.add_argument(
        "--curve", action="store_true", help="add the threshold curve at every 5%% of coverage"
    )
    report.add_argument(
        "--gold",
        metavar="FILE",
        help='score the head and its teacher against a JSONL file of verified {"text": ...,'
        ' "answer": ...} rows; needs the train extra',
    )
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
    importer.add_argument(
        "--dry-run", action="store_true", help="count the rows to record without recording them"
    )
    importer.add_argument("--config", metavar="PATH", help=_CONFIG_HELP)
    importer.set_defaults(handler=_import)

    site = commands.add_parser("site", help="remove or rename a decision site")
    site_commands = site.add_subparsers(dest="site_command", required=True)

    remove = site_commands.add_parser(
        "rm", help="delete a site's captures, decisions and model, and its field sites'"
    )
    remove.add_argument("site", metavar="SITE", help="site to remove")
    remove.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    remove.add_argument("--force", action="store_true", help="remove a site that is served live")
    remove.add_argument("--config", metavar="PATH", help=_CONFIG_HELP)
    remove.set_defaults(handler=_site_rm)

    rename = site_commands.add_parser(
        "rename",
        help="rename a site and its field sites; requests for the old name land on the new one",
    )
    rename.add_argument("old", metavar="OLD", help="site to rename")
    rename.add_argument("new", metavar="NEW", help="name to give it")
    rename.add_argument("--config", metavar="PATH", help=_CONFIG_HELP)
    rename.set_defaults(handler=_site_rename)

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
