from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import laya
import torch
from laya.common import QTYPES, build_sequence, collate_items
from safetensors.torch import save_file
from torch.nn.functional import cross_entropy

from stuntd.train.dataset import Item, NotTrainable, SiteDataset

__all__ = ["LayaTrainer", "question_for"]

# The cache holds one fp16 row per token, so its cost is known from the tokenised rows before any
# of it is built. 4 GiB leaves a 16 GB machine room for the model, the store and the rest of the
# run, and covers around 30 000 examples of the length the demos capture.
_CACHE_MAX_BYTES = 4 * 2**30
_BYTES_PER_VALUE = 2

_ALLOCATION_FAILED = ("not enough memory", "out of memory")
"""What torch puts in a RuntimeError when an allocation fails instead of raising MemoryError."""

_HEAD_PREFIXES = ("head.", "scorer.", "type_emb.")

_TORCH_ERRORS: tuple[type[Exception], ...] = (KeyError, OSError, RuntimeError, ValueError)
"""What a failing torch or laya call raises; stuntd reports every one of them the same way."""

# laya ships no type information, so the agent and everything reached through it is Any, the rows
# below are the untyped dicts its collate function consumes, and a batch is a list of either
# those rows or the cached encoder output that stands in for them.
_Row = dict[str, Any]
_Forward = Callable[[list[Any]], tuple[torch.Tensor, torch.Tensor]]
_Item = TypeVar("_Item")

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Encoded:
    hidden: torch.Tensor
    markers: Sequence[int]
    qtype: int
    label: int


def question_for(field: str, labels: Sequence[str]) -> dict[str, Any]:
    """The laya choice question for one site's field, asked the same way in training and serving."""
    return {"t": "choice", "ins": f"Choose {field}", "crit": dict.fromkeys(labels)}


def _chunks(items: Sequence[_Item], size: int) -> Iterator[list[_Item]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _allocation_failed(exc: Exception) -> bool:
    # Only a host allocation raises MemoryError; torch reports both its own out of memory and the
    # CPU allocator's as a RuntimeError, so the message is the only thing left to read.
    if isinstance(exc, MemoryError | torch.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and any(
        text in str(exc).lower() for text in _ALLOCATION_FAILED
    )


def _cache_bytes(rows: Sequence[_Row], hidden_size: int) -> int:
    return sum(len(row["ids"]) for row in rows) * hidden_size * _BYTES_PER_VALUE


def _head_forward(
    model: Any,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    marker_pos: torch.Tensor,
    marker_mask: torch.Tensor,
    qtype: torch.Tensor,
) -> torch.Tensor:
    # laya.common.DecisionModel.forward from the encoder on. Its act_head branch is left out:
    # act_head is frozen, and stuntd reads the label logits and nothing else.
    hidden = hidden + model.type_emb(qtype)[:, None, :]
    if model.head is not None:
        pad = ~attention_mask.bool()
        for layer in model.head.layers:
            hidden = layer(hidden, src_key_padding_mask=pad)
    index = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, hidden.size(-1))
    marked = torch.gather(hidden, 1, index)
    logits: torch.Tensor = model.scorer(marked).squeeze(-1).float()
    return logits.masked_fill(~marker_mask, -1e4)


def _head_tensors(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach()
        for name, tensor in model.state_dict().items()
        if name.startswith(_HEAD_PREFIXES)
    }


def _save_head(model: Any, head_path: Path) -> None:
    # float16 halves a file that is written once per site and only ever read back for inference.
    saved = {name: tensor.to("cpu", torch.float16) for name, tensor in _head_tensors(model).items()}
    save_file(saved, str(head_path))


class LayaTrainer:
    """Fine-tunes the Laya decision head on one site and writes it next to the metadata."""

    def __init__(
        self,
        base_model: str,
        device: str = "auto",
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        seed: int = 0,
        cache_encoder: bool = True,
        cache_max_bytes: int = _CACHE_MAX_BYTES,
    ) -> None:
        self._base_model = base_model
        self._epochs = epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._seed = seed
        self._cache_encoder = cache_encoder
        self._cache_max_bytes = cache_max_bytes
        try:
            self._agent: Any = laya.Agent(base_model, device=None if device == "auto" else device)
        except _TORCH_ERRORS as exc:
            raise RuntimeError(f"training failed: {exc}") from exc
        for parameter in self._agent.model.encoder.parameters():
            parameter.requires_grad = False
        # stuntd reads the label logits and nothing else, so act_head is frozen with the encoder
        # and stays out of both the optimizer and the saved head.
        for parameter in self._agent.model.act_head.parameters():
            parameter.requires_grad = False
        self._pristine_head = {
            name: tensor.clone() for name, tensor in _head_tensors(self._agent.model).items()
        }

    def __repr__(self) -> str:
        return f"LayaTrainer(base_model={self._base_model!r}, device={str(self._agent.device)!r})"

    def __call__(self, dataset: SiteDataset, head_path: Path) -> list[list[float]]:
        """Trains the head on one site, writes it to head_path and returns the holdout logits."""
        labels = len(dataset.labels)
        try:
            # One trainer fits every site, so each call starts from the checkpoint's own head
            # rather than from whatever the site before it left behind.
            self._agent.model.load_state_dict(self._pristine_head, strict=False)
            train, holdout = self._items(dataset)
            train_cache, holdout_cache = self._cache(train, holdout)
            self._fit(train, train_cache, labels)
            logits = self._logits(holdout, holdout_cache, labels)
            _save_head(self._agent.model, head_path)
        except _TORCH_ERRORS as exc:
            raise RuntimeError(f"training failed: {exc}") from exc
        return logits

    def _items(self, dataset: SiteDataset) -> tuple[list[_Row], list[_Row]]:
        cfg = self._agent.cfg
        question = question_for(dataset.field, dataset.labels)

        def row(item: Item) -> _Row:
            ids, markers = build_sequence(
                self._agent.tok, item.text, question, cfg["max_len"], cfg["head_max_len"]
            )
            if len(markers) < len(dataset.labels):
                raise NotTrainable("labels do not fit the head budget")
            return {"ids": ids, "markers": markers, "qtype": QTYPES["choice"], "label": item.label}

        return [row(item) for item in dataset.train], [row(item) for item in dataset.holdout]

    def _forward(self, rows: list[_Row]) -> tuple[torch.Tensor, torch.Tensor]:
        # laya's collate_items types its return as optional, but a non-empty batch never yields
        # None; the boundary is untyped, not the batch shape stuntd builds.
        batch = collate_items([rows], self._agent.tok.pad_token_id)
        device = self._agent.device
        logits, _ = self._agent.model(
            input_ids=batch["input_ids"].to(device),  # pyright: ignore[reportOptionalSubscript]
            attention_mask=batch["attention_mask"].to(device),  # pyright: ignore[reportOptionalSubscript]
            marker_pos=batch["marker_pos"].to(device),  # pyright: ignore[reportOptionalSubscript]
            marker_mask=batch["marker_mask"].to(device),  # pyright: ignore[reportOptionalSubscript]
            qtype=batch["qtype"].to(device),  # pyright: ignore[reportOptionalSubscript]
            detach_encoder=True,
        )
        scores: torch.Tensor = logits
        targets: torch.Tensor = batch["label"].to(device)  # pyright: ignore[reportOptionalSubscript]
        return scores, targets

    def _forward_cached(self, items: list[_Encoded]) -> tuple[torch.Tensor, torch.Tensor]:
        device = self._agent.device
        width = max(item.hidden.shape[0] for item in items)
        markers = max(len(item.markers) for item in items)
        hidden = torch.zeros((len(items), width, items[0].hidden.shape[1]), dtype=torch.float16)
        attention_mask = torch.zeros((len(items), width), dtype=torch.long)
        marker_pos = torch.zeros((len(items), markers), dtype=torch.long)
        marker_mask = torch.zeros((len(items), markers), dtype=torch.bool)
        for index, item in enumerate(items):
            length, kept = item.hidden.shape[0], len(item.markers)
            hidden[index, :length] = item.hidden
            attention_mask[index, :length] = 1
            marker_pos[index, :kept] = torch.tensor(item.markers)
            marker_mask[index, :kept] = True
        scores = _head_forward(
            self._agent.model,
            hidden.to(device).float(),
            attention_mask.to(device),
            marker_pos.to(device),
            marker_mask.to(device),
            torch.tensor([item.qtype for item in items]).to(device),
        )
        return scores, torch.tensor([item.label for item in items]).to(device)

    def _encode(self, rows: list[_Row]) -> list[_Encoded]:
        model = self._agent.model
        # Dropout must be off for this pass, or one random mask would be baked into every epoch.
        model.eval()
        device = self._agent.device
        encoded: list[_Encoded] = []
        with torch.no_grad():
            for chunk in _chunks(rows, self._batch_size):
                batch = collate_items([chunk], self._agent.tok.pad_token_id)
                with self._autocast():
                    hidden = model.encoder(
                        input_ids=batch["input_ids"].to(device),  # pyright: ignore[reportOptionalSubscript]
                        attention_mask=batch["attention_mask"].to(device),  # pyright: ignore[reportOptionalSubscript]
                    ).last_hidden_state
                # Padding is masked out of attention, so an example's own rows are the same
                # whatever else shared its batch, and only those rows are worth keeping.
                kept = hidden.to("cpu", torch.float16)
                for index, row in enumerate(chunk):
                    encoded.append(
                        _Encoded(
                            kept[index, : len(row["ids"])].clone(),
                            row["markers"],
                            row["qtype"],
                            row["label"],
                        )
                    )
        return encoded

    def _cache(
        self, train: list[_Row], holdout: list[_Row]
    ) -> tuple[list[_Encoded] | None, list[_Encoded] | None]:
        if not self._cache_encoder:
            return None, None
        hidden_size = self._agent.model.encoder.config.hidden_size
        wanted = _cache_bytes(train, hidden_size) + _cache_bytes(holdout, hidden_size)
        if wanted > self._cache_max_bytes:
            _log.warning(
                "re-encoding the encoder every epoch: its output needs %.1f GiB, over the %.1f "
                "GiB this run caches at most",
                wanted / 2**30,
                self._cache_max_bytes / 2**30,
            )
            return None, None
        try:
            return self._encode(train), self._encode(holdout)
        except (MemoryError, RuntimeError) as exc:
            if not _allocation_failed(exc):
                raise
            _log.warning("re-encoding the encoder every epoch: its output did not fit: %s", exc)
            return None, None

    def _autocast(self) -> AbstractContextManager[Any]:
        if self._agent.device.type != "cuda":
            return nullcontext()
        scope: AbstractContextManager[Any] = torch.autocast("cuda", dtype=self._agent.dtype)
        return scope

    def _fit(self, rows: list[_Row], cache: list[_Encoded] | None, labels: int) -> None:
        torch.manual_seed(self._seed)
        model = self._agent.model
        model.train()
        # The encoder is frozen and its output detached, so it stays a plain feature extractor;
        # only the head trains, dropout and all.
        model.encoder.eval()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self._learning_rate)
        items: list[Any] = rows if cache is None else cache
        forward: _Forward = self._forward if cache is None else self._forward_cached
        for _ in range(self._epochs):
            order = torch.randperm(len(items)).tolist()
            for chunk in _chunks([items[index] for index in order], self._batch_size):
                optimizer.zero_grad()
                with self._autocast():
                    scores, targets = forward(chunk)
                    loss = cross_entropy(scores[:, :labels], targets)
                torch.autograd.backward(loss)
                optimizer.step()

    def _logits(
        self, rows: list[_Row], cache: list[_Encoded] | None, labels: int
    ) -> list[list[float]]:
        self._agent.model.eval()
        items: list[Any] = rows if cache is None else cache
        forward: _Forward = self._forward if cache is None else self._forward_cached
        values: list[list[float]] = []
        with torch.no_grad():
            for chunk in _chunks(items, self._batch_size):
                scores, _ = forward(chunk)
                values.extend(scores[:, :labels].float().cpu().tolist())
        return values
