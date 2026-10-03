from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import laya
import torch
from laya.common import collate_items
from safetensors.torch import load_file

from stuntd.train.artifacts import EMBEDDINGS_FILE, SiteModel
from stuntd.train.layout import Layout
from stuntd.train.metrics import confidence, softmax
from stuntd.train.novelty import novelty, pool_hidden
from stuntd.train.trainer import site_row

__all__ = ["HEAD_CACHE_SIZE", "Decider", "Verdict"]

HEAD_CACHE_SIZE = 4
"""Heads kept in memory at once, dropping the one left unused longest."""

_HEAD_PREFIXES = ("head.", "scorer.", "type_emb.")

_BASE_HEAD: Literal["base"] = "base"
"""Stands in for a head key while the checkpoint's own head is the one installed."""

_TORCH_ERRORS: tuple[type[Exception], ...] = (KeyError, OSError, RuntimeError, ValueError)
"""What a failing torch, laya or safetensors call raises; stuntd reports them all the same way."""

_SHOWN_KEYS = 4
"""How many names a rejected head lists per fault before the message stops growing."""

_WARM_TEXT = "user: hello"
"""Stand-in input the warm-up pass runs on, short enough to cost one short sequence."""

# laya ships no type information, so the agent and everything reached through it is Any, and
# the row below is the untyped dict its collate function consumes.
_Row = dict[str, Any]

_HeadKey = tuple[Path, int]


@dataclass(frozen=True)
class _Head:
    weights: dict[str, torch.Tensor]
    embeddings: torch.Tensor | None


@dataclass(frozen=True)
class Verdict:
    """One answer from a site's head: the label it chose, every label's probability, how sure it
    is, how long it took, and how far the request is from the rows the head trained on, when the
    head kept them."""

    label: int
    confidence: float
    latency_ms: int
    probabilities: tuple[float, ...]
    novelty: float | None = None


class Decider:
    """Runs the Laya encoder once and swaps in the trained head of whichever site asks."""

    def __init__(self, base_model: str, device: str = "auto", lazy: bool = False) -> None:
        self._base_model = base_model
        self._device = device
        self._agent: Any = None
        self._heads: OrderedDict[_HeadKey, _Head] = OrderedDict()
        self._embeddings: torch.Tensor | None = None
        self._resident: _HeadKey | Literal["base"] | None = None
        self._lock = threading.Lock()
        if not lazy:
            self._load()

    def __repr__(self) -> str:
        return f"Decider(base_model={self._base_model!r}, device={self._device!r})"

    def decide(self, model: SiteModel, head_path: Path, text: str) -> Verdict:
        """Answers one request with the site's own head."""
        # One model carries one site's head at a time, so a caller's thread pool would otherwise
        # have two sites reading each other's weights mid-pass.
        with self._lock:
            self._ensure_loaded()
            self._install(head_path)
            try:
                started = time.perf_counter()
                scores, pooled = self._scores(self._row(text, model))
                distance = None if self._embeddings is None else novelty(pooled, self._embeddings)
                latency_ms = round((time.perf_counter() - started) * 1000)
            except _TORCH_ERRORS as exc:
                raise RuntimeError(f"decision failed: {exc}") from exc
        probs = softmax(scores[: len(model.labels)], model.temperature)
        label = max(range(len(probs)), key=probs.__getitem__)
        return Verdict(
            label,
            confidence(probs),
            latency_ms,
            tuple(probs),
            None if distance is None else float(distance[0]),
        )

    def answer(self, state: object, questions: dict[str, dict[str, object]]) -> dict[str, object]:
        """Answers Jev questions zero-shot, with the checkpoint's own head rather than a site's.

        The reply drops laya's per-answer action estimate, which is not part of a Jev answer.
        """
        with self._lock:
            self._ensure_loaded()
            self._install_base()
            try:
                result = self._agent.system_one(state, questions)
            except _TORCH_ERRORS as exc:
                raise RuntimeError(f"decision failed: {exc}") from exc
        answers = {
            qid: {name: value for name, value in fields.items() if name != "action"}
            for qid, fields in result["answers"].items()
        }
        return {"answers": answers, "usage": result["usage"]}

    def warm(self, model: SiteModel, head_path: Path) -> None:
        """Installs one site's head and runs a pass through it, so no caller waits for the first."""
        with self._lock:
            self._ensure_loaded()
            self._install(head_path)
            try:
                self._scores(self._row(_WARM_TEXT, model))
            except _TORCH_ERRORS as exc:
                raise RuntimeError(f"warm-up failed: {exc}") from exc

    def warm_base(self) -> None:
        """Runs one zero-shot pass, so the first Jev caller of a fresh daemon does not pay for it."""
        self.answer(_WARM_TEXT, {"warm": {"type": "noul", "instructions": "", "criteria": None}})

    def _ensure_loaded(self) -> None:
        if self._agent is None:
            self._load()

    def _load(self) -> None:
        try:
            agent: Any = laya.Agent(
                self._base_model, device=None if self._device == "auto" else self._device
            )
            agent.model.eval()
            self._dtype: torch.dtype = next(agent.model.parameters()).dtype
            self._base_head: dict[str, torch.Tensor] = {
                name: tensor.detach().clone()
                for name, tensor in agent.model.state_dict().items()
                if name.startswith(_HEAD_PREFIXES)
            }
        except _TORCH_ERRORS as exc:
            raise RuntimeError(f"base model failed to load: {exc}") from exc
        self._agent = agent

    def _head(self, head_path: Path, key: _HeadKey) -> _Head:
        cached = self._heads.get(key)
        if cached is not None:
            self._heads.move_to_end(key)
            return cached
        try:
            saved = load_file(str(head_path))
        except _TORCH_ERRORS as exc:
            raise RuntimeError(f"decision failed: {exc}") from exc
        foreign = sorted(name for name in saved if not name.startswith(_HEAD_PREFIXES))
        if foreign:
            raise RuntimeError(
                f"head does not match the base model: {foreign[:_SHOWN_KEYS]} are not head weights"
            )
        # The head is stored in float16 on the CPU; casting and moving it once keeps the pass at
        # one precision and keeps its weights off the bus on every later decision.
        weights = {
            name: tensor.to(self._agent.device, self._dtype) for name, tensor in saved.items()
        }
        head = _Head(weights, self._load_embeddings(head_path))
        # Retraining a site leaves its earlier head behind, and nothing will ever ask for it
        # again, so one path keeps one head rather than a version per training run.
        for stale in [known for known in self._heads if known[0] == head_path]:
            del self._heads[stale]
        self._heads[key] = head
        # A head is tens of megabytes on the device, so a daemon serving many sites keeps only
        # the few it was asked for last.
        if len(self._heads) > HEAD_CACHE_SIZE:
            self._heads.popitem(last=False)
        return head

    def _load_embeddings(self, head_path: Path) -> torch.Tensor | None:
        path = head_path.with_name(EMBEDDINGS_FILE)
        if not path.is_file():
            return None
        try:
            return load_file(str(path))["embeddings"].to(self._agent.device, torch.float32)
        except _TORCH_ERRORS as exc:
            raise RuntimeError(f"decision failed: {exc}") from exc

    def _install(self, head_path: Path) -> None:
        try:
            # Retraining replaces a site's folder wholesale, so the file's own mtime is what
            # tells a cached head apart from the one now on disk.
            key: _HeadKey = (head_path, head_path.stat().st_mtime_ns)
        except OSError as exc:
            raise RuntimeError(f"decision failed: {exc}") from exc
        if key == self._resident:
            return
        head = self._head(head_path, key)
        self._resident = None
        try:
            incompatible = self._agent.model.load_state_dict(head.weights, strict=False)
        except _TORCH_ERRORS as exc:
            raise RuntimeError(f"decision failed: {exc}") from exc
        unexpected = sorted(incompatible.unexpected_keys)
        missing = sorted(
            name for name in incompatible.missing_keys if name.startswith(_HEAD_PREFIXES)
        )
        if unexpected or missing:
            raise RuntimeError(
                f"head does not match the base model: unexpected {unexpected[:_SHOWN_KEYS]},"
                f" missing {missing[:_SHOWN_KEYS]}"
            )
        self._embeddings = head.embeddings
        self._resident = key

    def _install_base(self) -> None:
        # A site's head answers that site's labels and nothing else, so a zero-shot question is
        # only meaningful once the head the checkpoint shipped is back in the model.
        if self._resident == _BASE_HEAD:
            return
        self._resident = None
        try:
            self._agent.model.load_state_dict(self._base_head, strict=False)
        except _TORCH_ERRORS as exc:
            raise RuntimeError(f"decision failed: {exc}") from exc
        self._resident = _BASE_HEAD

    def _row(self, text: str, model: SiteModel) -> _Row:
        cfg = self._agent.cfg
        layout = Layout(
            cfg["max_len"] if model.max_len is None else model.max_len,
            cfg["head_max_len"] if model.head_max_len is None else model.head_max_len,
            model.spaced_labels,
        )
        row = site_row(self._agent.tok, text, model.field, model.labels, layout)
        if len(row["markers"]) < len(model.labels):
            raise ValueError(f"{len(model.labels)} labels do not fit the head budget")
        return row

    def _scores(self, row: _Row) -> tuple[list[float], torch.Tensor]:
        # laya's collate_items types its return as optional, but a one-row batch never yields
        # None; the boundary is untyped, not the batch shape stuntd builds.
        batch = collate_items([[row]], self._agent.tok.pad_token_id)
        device = self._agent.device
        attention_mask = batch["attention_mask"].to(device)  # pyright: ignore[reportOptionalSubscript]
        hidden: list[torch.Tensor] = []
        # The model's forward runs the encoder itself, so its output is read off the call rather
        # than computed a second time.
        handle = self._agent.model.encoder.register_forward_hook(
            lambda _module, _inputs, output: hidden.append(output.last_hidden_state)
        )
        try:
            with torch.no_grad():
                logits, _ = self._agent.model(
                    input_ids=batch["input_ids"].to(device),  # pyright: ignore[reportOptionalSubscript]
                    attention_mask=attention_mask,
                    marker_pos=batch["marker_pos"].to(device),  # pyright: ignore[reportOptionalSubscript]
                    marker_mask=batch["marker_mask"].to(device),  # pyright: ignore[reportOptionalSubscript]
                    qtype=batch["qtype"].to(device),  # pyright: ignore[reportOptionalSubscript]
                )
        finally:
            handle.remove()
        scores: list[float] = logits[0].float().cpu().tolist()
        return scores, pool_hidden(hidden[0].float(), attention_mask)
