from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.nn.functional import cross_entropy, dropout, normalize
from transformers import AutoModel, AutoTokenizer

from stuntd.train.artifacts import EMBEDDINGS_FILE
from stuntd.train.dataset import SiteDataset
from stuntd.train.encoders import EncoderSpec
from stuntd.train.layout import Layout
from stuntd.train.novelty import novelty, pool_hidden, save_embeddings
from stuntd.train.run import TrainedHead

__all__ = ["HEAD_BIAS", "HEAD_WEIGHT", "TORCH_ERRORS", "Encoder", "PooledTrainer"]

HEAD_WEIGHT = "head.weight"
HEAD_BIAS = "head.bias"

TORCH_ERRORS: tuple[type[Exception], ...] = (KeyError, OSError, RuntimeError, ValueError)
"""What a failing torch, transformers or safetensors call raises; stuntd reports them all alike."""

_BATCH_SIZE = 64
_EPOCHS = 80
_LEARNING_RATE = 3e-2
_WEIGHT_DECAY = 1e-2
_DROPOUT = 0.1


class Encoder:
    """A frozen sentence encoder that turns texts into unit-length vectors."""

    def __init__(self, spec: EncoderSpec, device: str = "auto") -> None:
        self._spec = spec
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(spec.name, trust_remote_code=False)
            model = AutoModel.from_pretrained(spec.name, trust_remote_code=False)
        except TORCH_ERRORS as exc:
            raise RuntimeError(f"encoder {spec.name!r} failed to load: {exc}") from exc
        self._model = model.to(self._device).eval()

    def __repr__(self) -> str:
        return f"Encoder(name={self._spec.name!r}, device={str(self._device)!r})"

    @property
    def hidden_size(self) -> int:
        return int(self._model.config.hidden_size)

    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        """One unit-length vector per text, on the CPU."""
        vectors = []
        with torch.no_grad():
            for start in range(0, len(texts), _BATCH_SIZE):
                batch = self._tokenizer(
                    [self._spec.prefix + text for text in texts[start : start + _BATCH_SIZE]],
                    padding=True,
                    truncation=True,
                    max_length=self._spec.max_length,
                    return_tensors="pt",
                ).to(self._device)
                hidden = self._model(**batch).last_hidden_state.float()
                if self._spec.pooling == "mean":
                    pooled = pool_hidden(hidden, batch["attention_mask"])
                else:
                    pooled = normalize(hidden[:, 0], dim=1)
                vectors.append(pooled.cpu())
        return torch.cat(vectors)


class PooledTrainer:
    """Trains a linear head on the pooled vectors of a frozen stock sentence encoder."""

    def __init__(
        self,
        encoder: EncoderSpec,
        device: str = "auto",
        epochs: int = _EPOCHS,
        learning_rate: float = _LEARNING_RATE,
        weight_decay: float = _WEIGHT_DECAY,
        seed: int = 0,
    ) -> None:
        self._spec = encoder
        self._encoder = Encoder(encoder, device)
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._seed = seed

    def __repr__(self) -> str:
        return f"PooledTrainer(encoder={self._spec.name!r})"

    def __call__(self, dataset: SiteDataset, head_path: Path) -> TrainedHead:
        """Trains the head on one site, writes it to head_path and returns its holdout logits."""
        try:
            reference = self._encoder.embed([item.text for item in dataset.train])
            queries = self._encoder.embed([item.text for item in dataset.holdout])
            targets = torch.tensor([item.label for item in dataset.train])
            head = self._fit(reference, targets, len(dataset.labels))
            with torch.no_grad():
                logits = head(queries).tolist()
            save_file(
                {HEAD_WEIGHT: head.weight.detach().clone(), HEAD_BIAS: head.bias.detach().clone()},
                str(head_path),
            )
            save_embeddings(reference, head_path.with_name(EMBEDDINGS_FILE))
            distances = novelty(queries, reference).tolist()
        except TORCH_ERRORS as exc:
            raise RuntimeError(f"training failed: {exc}") from exc
        return TrainedHead(logits, Layout(self._spec.max_length, 0, False), distances)

    def _fit(self, vectors: torch.Tensor, targets: torch.Tensor, labels: int) -> torch.nn.Linear:
        torch.manual_seed(self._seed)
        head = torch.nn.Linear(vectors.shape[1], labels)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay
        )
        for _ in range(self._epochs):
            order = torch.randperm(len(vectors))
            for start in range(0, len(order), _BATCH_SIZE):
                batch = order[start : start + _BATCH_SIZE]
                optimizer.zero_grad()
                loss = cross_entropy(head(dropout(vectors[batch], _DROPOUT)), targets[batch])
                torch.autograd.backward(loss)
                optimizer.step()
        return head.eval()
