import math

import pytest

torch = pytest.importorskip("torch")

from safetensors.torch import load_file  # noqa: E402

from stuntd.train.artifacts import EMBEDDINGS_FILE  # noqa: E402
from stuntd.train.novelty import NEIGHBOURS, novelty, pool_hidden, save_embeddings  # noqa: E402


def test_pooled_embedding_is_the_masked_mean_normalised():
    hidden = torch.tensor([[[3.0, 0.0], [0.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = pool_hidden(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[0.6, 0.8]]))
    assert math.isclose(pooled.norm().item(), 1.0, rel_tol=1e-6)


def test_pooled_embedding_ignores_the_padding_around_a_row():
    row = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
    padded = torch.cat([row, torch.full((2, 2), 9.0)])
    alone = pool_hidden(row[None], torch.ones(1, 2))
    batched = pool_hidden(padded[None], torch.tensor([[1, 1, 0, 0]]))
    assert torch.allclose(alone, batched)


def test_novelty_is_one_minus_the_mean_of_the_nearest_similarities():
    near = torch.tensor([[1.0, 0.0]]).repeat(3, 1)
    far = torch.tensor([[0.0, 1.0]]).repeat(4, 1)
    reference = torch.cat([far, near])
    expected = 1 - 3 / NEIGHBOURS
    assert novelty(torch.tensor([[1.0, 0.0]]), reference).tolist() == pytest.approx([expected])


def test_novelty_of_a_seen_row_is_zero():
    reference = torch.eye(3).repeat(NEIGHBOURS, 1)
    assert novelty(torch.eye(3), reference).tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_novelty_of_an_unrelated_row_is_one():
    reference = torch.tensor([[1.0, 0.0]]).repeat(NEIGHBOURS, 1)
    assert novelty(torch.tensor([[0.0, 1.0]]), reference).tolist() == pytest.approx([1.0])


def test_novelty_with_fewer_rows_than_neighbours_uses_all_of_them():
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert novelty(torch.tensor([[1.0, 0.0]]), reference).tolist() == pytest.approx([0.5])


def test_saved_embeddings_are_one_half_precision_row_per_training_row(tmp_path):
    embeddings = torch.nn.functional.normalize(torch.randn(7, 16), dim=1)
    path = tmp_path / EMBEDDINGS_FILE
    save_embeddings(embeddings, path)
    saved = load_file(str(path))["embeddings"]
    assert saved.dtype == torch.float16 and saved.shape == (7, 16)
    assert torch.allclose(saved.float(), embeddings, atol=1e-3)
