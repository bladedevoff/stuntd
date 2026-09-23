from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlsplit

from stuntd.paths import data_dir

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

__all__ = [
    "CONFIG_TEMPLATE",
    "Settings",
    "config_path",
    "database_path",
    "load_settings",
    "models_path",
]

CONFIG_TEMPLATE = """# stuntd settings. Every key is optional; a missing key keeps the default shown.

# Provider base URL that requests are forwarded to: the origin only, because the caller's own
# path, /v1/chat/completions and the rest, is appended to it unchanged. Left empty, only the
# Jev endpoint is served and the OpenAI path answers 502.
# upstream = "https://api.openai.com"
# host = "127.0.0.1"
# port = 8787

# Whether the daemon learns at all. false: never write captures, models or
# decisions; pure proxy / local Jev.
# learn = true

[redaction]
# Built-in rules replace emails and phone numbers before anything is stored.
# enabled = true
# Extra regular expressions, each replaced with [redacted]. Single quotes need no escaping.
# patterns = ['ACC-\\d{6}']

[storage]
# db_path = "captures.sqlite"    # relative paths resolve against the data directory
# max_rows = 100000
# max_age_days = 30
# models_dir = "models"          # relative paths resolve against the data directory

[training]
# Captures a decision site needs before it is worth training on.
# min_examples = 300
# Share of the examples held back to measure the trained model.
# holdout = 0.2
# Agreement with the provider the trained model must reach before it is used.
# target_agreement = 0.99
# base_model = "convaiinnovations/laya"
# epochs = 3
# device = "auto"                # auto, cpu, cuda or mps
# Whether the frozen encoder runs once per example instead of once per epoch; off re-encodes.
# cache_encoder = true
# Most memory the cached encoder output may take, in megabytes; a bigger site trains uncached.
# cache_max_mb = 4096

[serving]
# Share of the requests a serving site still sends to the provider to check its own answer.
# Ignored in local Jev mode, where there is no provider to check it against.
# check_share = 0.02
# Recent decisions a site is judged on.
# window = 100
# Decisions needed before that judgement means anything.
# min_window = 20
# Whether a shadow site that holds its target agreement goes live on its own.
# auto_promote = false
# auto_promote_after_hours = 24
# Answers kept in memory, in one cache shared by every site, dropping the least recently used.
# cache_size = 1000

[jev]
# Origin of the Jev provider answers are fetched from: the origin only, no path. Left empty, the
# base Laya answers locally instead.
# upstream = ""
# Whether a Jev request must carry a Bearer token; the token itself is not checked.
# require_key = false
# Name reported in the model field of a Jev answer.
# model_name = "stuntd"
"""
"""The commented stuntd.toml written for a fresh installation."""

_DEFAULT_DB_NAME = "captures.sqlite"
_DEFAULT_MODELS_NAME = "models"
_MAX_PORT = 65535
_DEVICES = ("auto", "cpu", "cuda", "mps")
_ORIGIN_SCHEMES = ("http", "https")
_MODEL_NAME = re.compile(r"[A-Za-z0-9_.-]{1,64}")

_TOP = {"upstream": str, "host": str, "port": int, "learn": bool}
_SECTIONS = {
    "redaction": {"enabled": bool, "patterns": list},
    "storage": {"db_path": str, "max_rows": int, "max_age_days": int, "models_dir": str},
    "training": {
        "min_examples": int,
        "holdout": float,
        "target_agreement": float,
        "base_model": str,
        "epochs": int,
        "device": str,
        "cache_encoder": bool,
        "cache_max_mb": int,
    },
    "serving": {
        "check_share": float,
        "window": int,
        "min_window": int,
        "auto_promote": bool,
        "auto_promote_after_hours": int,
        "cache_size": int,
    },
    "jev": {"upstream": str, "require_key": bool, "model_name": str},
}

_T = TypeVar("_T")


@dataclass
class Settings:
    """Everything the daemon needs: where to forward, where to listen, what to keep."""

    upstream: str = ""
    """Provider base URL that requests are forwarded to; empty serves only the Jev endpoint."""

    host: str = "127.0.0.1"
    """Address the daemon listens on."""

    port: int = 8787
    """Port the daemon listens on."""

    db_path: Path | None = None
    """Capture database; None means the default file inside the data directory."""

    redact: bool = True
    """Whether the built-in email and phone rules run before a capture is written."""

    redaction_patterns: list[str] = field(default_factory=list)
    """Extra regular expressions, each replaced with [redacted]."""

    max_rows: int = 100_000
    """Captures kept before the oldest are pruned."""

    max_age_days: int = 30
    """Age at which a capture is pruned."""

    models_dir: Path | None = None
    """Trained models; None means the default directory inside the data directory."""

    min_examples: int = 300
    """Captures a decision site needs before it is worth training on."""

    holdout: float = 0.2
    """Share of the examples held back to measure the trained model."""

    target_agreement: float = 0.99
    """Agreement with the provider the trained model must reach before it is used."""

    base_model: str = "convaiinnovations/laya"
    """Model the per-site models are trained from."""

    epochs: int = 3
    """Passes over the training examples."""

    device: str = "auto"
    """Device training runs on: auto, cpu, cuda or mps."""

    cache_encoder: bool = True
    """Whether the frozen encoder runs once per example rather than once per epoch."""

    cache_max_mb: int = 4096
    """Most memory the cached encoder output may take; a site that needs more trains uncached."""

    check_share: float = 0.02
    """Share of the requests a serving site still sends to the provider to check its own answer,
    ignored in local Jev mode, where there is no provider to check it against."""

    window: int = 100
    """Recent decisions a site is judged on."""

    min_window: int = 20
    """Decisions needed before that judgement means anything."""

    auto_promote: bool = False
    """Whether a shadow site that holds its target agreement goes live on its own."""

    auto_promote_after_hours: int = 24
    """Hours a site stays in shadow before it may be promoted."""

    cache_size: int = 1000
    """Answers kept in one cache shared by every site, dropping the least recently used."""

    learn: bool = True
    """Whether captures, models and decisions are written at all."""

    jev_upstream: str = ""
    """Origin of the Jev provider; empty means the base Laya answers locally."""

    jev_require_key: bool = False
    """Whether a Jev request must carry a Bearer token; the token itself is not checked."""

    jev_model_name: str = "stuntd"
    """Name reported in the model field of a Jev answer."""


def config_path() -> Path:
    """The settings file inside the data directory."""
    return data_dir() / "stuntd.toml"


def database_path(settings: Settings) -> Path:
    """The capture database: the configured file, or the default one in the data directory."""
    if settings.db_path is None:
        return data_dir() / _DEFAULT_DB_NAME
    return settings.db_path


def models_path(settings: Settings) -> Path:
    """The trained models: the configured directory, or the default one in the data directory."""
    if settings.models_dir is None:
        return data_dir() / _DEFAULT_MODELS_NAME
    return settings.models_dir


def _under_data_dir(raw: str) -> Path:
    # The config file is not tied to a working directory, so a relative path in it must name the
    # same place wherever the daemon is started from.
    configured = Path(raw)
    return configured if configured.is_absolute() else data_dir() / configured


def _check_jev_upstream(url: str) -> None:
    message = f"jev.upstream must be an origin like https://api.typesafe.ai, got {url!r}"
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        # urlsplit rejects a malformed IPv6 host itself, with a message that names no setting.
        raise ValueError(message) from exc
    # Userinfo is barred as well: config show prints this value, and a password does not belong
    # on stdout.
    if (
        parts.scheme not in _ORIGIN_SCHEMES
        or not parts.netloc
        or "@" in parts.netloc
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
    ):
        raise ValueError(message)


def _typed(where: str, value: object, expected: type[_T]) -> _T:
    # bool is an int subclass, so `port = true` would otherwise pass as an int.
    if isinstance(value, expected) and not (expected is int and isinstance(value, bool)):
        return value
    raise ValueError(f"{where} must be {expected.__name__}, got {type(value).__name__}")


def _read_file(path: Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        try:
            raw = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"{path}: {exc}") from exc
    values: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _TOP:
            values[key] = _typed(key, value, _TOP[key])
        elif key in _SECTIONS:
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a table")
            for sub, subvalue in value.items():
                if sub not in _SECTIONS[key]:
                    raise ValueError(f"unknown key {key}.{sub}")
                values[f"{key}.{sub}"] = _typed(f"{key}.{sub}", subvalue, _SECTIONS[key][sub])
        else:
            raise ValueError(f"unknown key {key}")
    return values


def load_settings(path: Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    """Reads the TOML file if it exists, then applies overrides whose value is not None."""
    values = _read_file(path) if path is not None and path.exists() else {}
    settings = Settings()
    settings.upstream = values.get("upstream", settings.upstream)
    settings.host = values.get("host", settings.host)
    settings.port = values.get("port", settings.port)
    settings.redact = values.get("redaction.enabled", settings.redact)
    patterns = values.get("redaction.patterns", [])
    if not all(isinstance(item, str) for item in patterns):
        raise ValueError("redaction.patterns must be a list of strings")
    settings.redaction_patterns = list(patterns)
    if "storage.db_path" in values:
        settings.db_path = _under_data_dir(values["storage.db_path"])
    settings.max_rows = values.get("storage.max_rows", settings.max_rows)
    settings.max_age_days = values.get("storage.max_age_days", settings.max_age_days)
    if "storage.models_dir" in values:
        settings.models_dir = _under_data_dir(values["storage.models_dir"])
    settings.min_examples = values.get("training.min_examples", settings.min_examples)
    settings.holdout = values.get("training.holdout", settings.holdout)
    settings.target_agreement = values.get("training.target_agreement", settings.target_agreement)
    settings.base_model = values.get("training.base_model", settings.base_model)
    settings.epochs = values.get("training.epochs", settings.epochs)
    settings.device = values.get("training.device", settings.device)
    settings.cache_encoder = values.get("training.cache_encoder", settings.cache_encoder)
    settings.cache_max_mb = values.get("training.cache_max_mb", settings.cache_max_mb)
    settings.check_share = values.get("serving.check_share", settings.check_share)
    settings.window = values.get("serving.window", settings.window)
    settings.min_window = values.get("serving.min_window", settings.min_window)
    settings.auto_promote = values.get("serving.auto_promote", settings.auto_promote)
    settings.auto_promote_after_hours = values.get(
        "serving.auto_promote_after_hours", settings.auto_promote_after_hours
    )
    settings.cache_size = values.get("serving.cache_size", settings.cache_size)
    settings.learn = values.get("learn", settings.learn)
    settings.jev_upstream = values.get("jev.upstream", settings.jev_upstream)
    settings.jev_require_key = values.get("jev.require_key", settings.jev_require_key)
    settings.jev_model_name = values.get("jev.model_name", settings.jev_model_name)
    if overrides:
        known = {item.name for item in fields(Settings)}
        for key, value in overrides.items():
            if key not in known:
                raise ValueError(f"unknown override {key}")
            if value is not None:
                setattr(settings, key, value)
    if not 1 <= settings.port <= _MAX_PORT:
        raise ValueError(f"port must be between 1 and {_MAX_PORT}, got {settings.port!r}")
    if settings.max_rows < 1:
        raise ValueError(f"max_rows must be at least 1, got {settings.max_rows!r}")
    if settings.max_age_days < 1:
        raise ValueError(f"max_age_days must be at least 1, got {settings.max_age_days!r}")
    if settings.min_examples < 2:
        raise ValueError(f"min_examples must be at least 2, got {settings.min_examples!r}")
    if not 0 < settings.holdout <= 0.5:
        raise ValueError(f"holdout must be above 0 and at most 0.5, got {settings.holdout!r}")
    if not 0 < settings.target_agreement <= 1:
        raise ValueError(
            f"target_agreement must be above 0 and at most 1, got {settings.target_agreement!r}"
        )
    if settings.epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {settings.epochs!r}")
    if settings.cache_max_mb < 0:
        raise ValueError(f"cache_max_mb cannot be negative, got {settings.cache_max_mb!r}")
    if settings.device not in _DEVICES:
        raise ValueError(f"device must be one of {', '.join(_DEVICES)}, got {settings.device!r}")
    if not 0 <= settings.check_share < 1:
        raise ValueError(
            f"check_share must be at least 0 and below 1, got {settings.check_share!r}"
        )
    if settings.window < 1:
        raise ValueError(f"window must be at least 1, got {settings.window!r}")
    if not 1 <= settings.min_window <= settings.window:
        raise ValueError(
            f"min_window must be between 1 and {settings.window}, got {settings.min_window!r}"
        )
    if settings.auto_promote_after_hours < 0:
        raise ValueError(
            f"auto_promote_after_hours must be at least 0, "
            f"got {settings.auto_promote_after_hours!r}"
        )
    if settings.cache_size < 0:
        raise ValueError(f"cache_size must be at least 0, got {settings.cache_size!r}")
    if settings.jev_upstream:
        _check_jev_upstream(settings.jev_upstream)
    if not _MODEL_NAME.fullmatch(settings.jev_model_name):
        raise ValueError(
            f"jev.model_name must be 1 to 64 letters, digits, dot, dash or underscore, "
            f"got {settings.jev_model_name!r}"
        )
    return settings
