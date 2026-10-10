from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

__all__ = ["ALLOWED_MODEL_TYPES", "EncoderSpec", "resolve_encoder"]

ALLOWED_MODEL_TYPES = ("bert", "distilbert", "roberta", "xlm-roberta", "mpnet", "modernbert")
"""Architectures a stock encoder may have; anything else is refused before its weights load."""

_DEFAULT_MAX_LENGTH = 512

_PRESET_PREFIXES = {
    "intfloat/multilingual-e5-base": "query: ",
    "intfloat/multilingual-e5-small": "query: ",
    "sentence-transformers/all-MiniLM-L6-v2": "",
}

_Pooling = Literal["mean", "cls"]


@dataclass(frozen=True)
class EncoderSpec:
    """How a stock sentence encoder is run: which model, how its tokens pool, how long an input
    it takes and what every input starts with."""

    name: str
    pooling: _Pooling
    max_length: int
    prefix: str


def _read(name: str, filename: str) -> str | None:
    folder = Path(name)
    if folder.is_dir():
        path = folder / filename
        return path.read_text(encoding="utf-8") if path.is_file() else None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, HFValidationError

    try:
        return Path(hf_hub_download(name, filename)).read_text(encoding="utf-8")
    except EntryNotFoundError:
        return None
    except (HfHubHTTPError, HFValidationError) as exc:
        raise ValueError(
            f"training.encoder {name!r} is neither a folder nor a Hugging Face model"
        ) from exc


def _json(name: str, filename: str) -> dict[str, Any] | None:
    # Any because json.loads returns untyped values; every field read below is checked.
    text = _read(name, filename)
    if text is None:
        return None
    try:
        content = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"training.encoder {name!r}: {filename} is not valid JSON: {exc}") from exc
    if not isinstance(content, dict):
        raise ValueError(f"training.encoder {name!r}: {filename} is not a JSON object")
    return content


def _pooling(name: str, config: dict[str, Any] | None) -> _Pooling:
    if config is None:
        return "mean"
    enabled = [key for key, value in config.items() if key.startswith("pooling_mode_") and value]
    if enabled == ["pooling_mode_mean_tokens"]:
        return "mean"
    if enabled == ["pooling_mode_cls_token"]:
        return "cls"
    raise ValueError(
        f"training.encoder {name!r} pools with {enabled}, only mean and cls pooling are supported"
    )


def _max_length(name: str, config: dict[str, Any] | None) -> int:
    if config is None or "max_seq_length" not in config:
        return _DEFAULT_MAX_LENGTH
    length = config["max_seq_length"]
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError(
            f"training.encoder {name!r}: max_seq_length must be a positive integer, got {length!r}"
        )
    return length


def _prefix(name: str, config: dict[str, Any] | None) -> str:
    if name in _PRESET_PREFIXES:
        return _PRESET_PREFIXES[name]
    prompts = None if config is None else config.get("prompts")
    query = prompts.get("query") if isinstance(prompts, dict) else None
    return query if isinstance(query, str) else ""


def resolve_encoder(name: str) -> EncoderSpec:
    """Reads a stock encoder's configuration, refusing one that would run remote code."""
    config = _json(name, "config.json")
    if config is None:
        raise ValueError(f"training.encoder {name!r} has no config.json")
    if "auto_map" in config:
        raise ValueError(f"training.encoder {name!r} needs remote code (auto_map in config.json)")
    if config.get("model_type") not in ALLOWED_MODEL_TYPES:
        raise ValueError(
            f"training.encoder {name!r} has model_type {config.get('model_type')!r},"
            f" expected one of {', '.join(ALLOWED_MODEL_TYPES)}"
        )
    return EncoderSpec(
        name,
        _pooling(name, _json(name, "1_Pooling/config.json")),
        _max_length(name, _json(name, "sentence_bert_config.json")),
        _prefix(name, _json(name, "config_sentence_transformers.json")),
    )
