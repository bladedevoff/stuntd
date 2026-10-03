from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.nn.functional import normalize

__all__ = ["NEIGHBOURS", "novelty", "pool_hidden", "save_embeddings"]

NEIGHBOURS = 5
"""Training rows a request is compared with when its novelty is measured."""


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """The mean of each row's encoder output over its real tokens, scaled to unit length."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return normalize(mean, dim=1)


def novelty(queries: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """One minus the mean cosine similarity of each query to its nearest reference rows."""
    nearest = (queries @ reference.T).topk(min(NEIGHBOURS, reference.shape[0]), dim=1).values
    return 1.0 - nearest.mean(dim=1)


def save_embeddings(embeddings: torch.Tensor, path: Path) -> None:
    """Writes the training rows' pooled embeddings in half precision."""
    save_file({"embeddings": embeddings.to("cpu", torch.float16).contiguous()}, str(path))
