from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from stuntd.serve.heads import HEAD_CACHE_SIZE, Verdict
from stuntd.train.artifacts import EMBEDDINGS_FILE, SiteModel
from stuntd.train.encoders import EncoderSpec
from stuntd.train.metrics import confidence, softmax
from stuntd.train.novelty import novelty
from stuntd.train.pooled import HEAD_BIAS, HEAD_WEIGHT, TORCH_ERRORS, Encoder

__all__ = ["PooledDecider"]

_WARM_TEXT = "user: hello"
"""Stand-in input the warm-up pass runs on."""

_HeadKey = tuple[Path, int]


@dataclass(frozen=True)
class _Head:
    weight: torch.Tensor
    bias: torch.Tensor
    embeddings: torch.Tensor | None


class PooledDecider:
    """Runs a stock sentence encoder once per request and applies the linear head of the site."""

    def __init__(self, encoder: EncoderSpec, device: str = "auto", lazy: bool = False) -> None:
        self._spec = encoder
        self._device = device
        self._encoder: Encoder | None = None
        self._heads: OrderedDict[_HeadKey, _Head] = OrderedDict()
        self._lock = threading.Lock()
        if not lazy:
            self._ensure_loaded()

    def __repr__(self) -> str:
        return f"PooledDecider(encoder={self._spec.name!r}, device={self._device!r})"

    def decide(self, model: SiteModel, head_path: Path, text: str) -> Verdict:
        """Answers one request with the site's own head."""
        with self._lock:
            encoder = self._ensure_loaded()
            head = self._head(encoder, model, head_path)
            try:
                started = time.perf_counter()
                vector = encoder.embed([text])
                scores = (vector @ head.weight.T + head.bias)[0].tolist()
                distance = None if head.embeddings is None else novelty(vector, head.embeddings)
                latency_ms = round((time.perf_counter() - started) * 1000)
            except TORCH_ERRORS as exc:
                raise RuntimeError(f"decision failed: {exc}") from exc
        probs = softmax(scores, model.temperature)
        label = max(range(len(probs)), key=probs.__getitem__)
        return Verdict(
            label,
            confidence(probs),
            latency_ms,
            tuple(probs),
            None if distance is None else float(distance[0]),
        )

    def warm(self, model: SiteModel, head_path: Path) -> None:
        """Installs one site's head and runs a pass through the encoder."""
        with self._lock:
            encoder = self._ensure_loaded()
            self._head(encoder, model, head_path)
            try:
                encoder.embed([_WARM_TEXT])
            except TORCH_ERRORS as exc:
                raise RuntimeError(f"warm-up failed: {exc}") from exc

    def _ensure_loaded(self) -> Encoder:
        if self._encoder is None:
            self._encoder = Encoder(self._spec, self._device)
        return self._encoder

    def _head(self, encoder: Encoder, model: SiteModel, head_path: Path) -> _Head:
        try:
            # Retraining replaces a site's folder wholesale, so the file's own mtime is what
            # tells a cached head apart from the one now on disk.
            key: _HeadKey = (head_path, head_path.stat().st_mtime_ns)
        except OSError as exc:
            raise RuntimeError(f"decision failed: {exc}") from exc
        cached = self._heads.get(key)
        if cached is not None:
            self._heads.move_to_end(key)
            return cached
        head = self._load(encoder, model, head_path)
        for stale in [known for known in self._heads if known[0] == head_path]:
            del self._heads[stale]
        self._heads[key] = head
        if len(self._heads) > HEAD_CACHE_SIZE:
            self._heads.popitem(last=False)
        return head

    def _load(self, encoder: Encoder, model: SiteModel, head_path: Path) -> _Head:
        width = encoder.hidden_size
        embeddings_path = head_path.with_name(EMBEDDINGS_FILE)
        try:
            saved = load_file(str(head_path))
            embeddings = (
                load_file(str(embeddings_path))["embeddings"].float()
                if embeddings_path.is_file()
                else None
            )
        except TORCH_ERRORS as exc:
            raise RuntimeError(f"decision failed: {exc}") from exc
        shapes = {name: tuple(tensor.shape) for name, tensor in saved.items()}
        expected = {HEAD_WEIGHT: (len(model.labels), width), HEAD_BIAS: (len(model.labels),)}
        if shapes != expected or (embeddings is not None and embeddings.shape[1] != width):
            raise RuntimeError(
                f"head does not fit the encoder {self._spec.name!r}: expected tensors {expected},"
                f" found {shapes}"
            )
        return _Head(saved[HEAD_WEIGHT].float(), saved[HEAD_BIAS].float(), embeddings)
