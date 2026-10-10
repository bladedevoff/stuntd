import pytest

from stuntd.serve.runtime import is_novel
from stuntd.settings import Settings
from stuntd.store.db import Capture, Store
from stuntd.store.redact import Redactor
from stuntd.train.artifacts import HEAD_FILE, load_model, site_dir
from stuntd.train.run import train_sites

pytestmark = pytest.mark.slow

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
SCHEMA = '{"properties":{"refund":{"type":"boolean"}},"type":"object"}'
YES = "user: I was charged twice for invoice {n}, please send the money back."
NO = "user: Where do I find the invoice {n} for last month?"


@pytest.fixture(scope="module")
def cached_encoder():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    hub = pytest.importorskip("huggingface_hub")
    if not isinstance(hub.try_to_load_from_cache(ENCODER, "model.safetensors"), str):
        pytest.skip(f"{ENCODER} is not in the Hugging Face cache")
    return ENCODER


def test_a_stock_encoder_trains_and_serves_a_toy_site(cached_encoder, tmp_path):
    from stuntd.serve.pooled import PooledDecider
    from stuntd.train.encoders import resolve_encoder
    from stuntd.train.pooled import PooledTrainer

    store = Store(tmp_path / "captures.sqlite", Redactor())
    for index in range(60):
        text, answer = (YES, "true") if index % 2 else (NO, "false")
        store.record(
            Capture("refund", SCHEMA, "boolean", text.format(n=1000 + index), answer, "m", 1, 1, 1)
        )
    models = tmp_path / "models"
    settings = Settings(
        min_examples=10, holdout=0.25, encoder=ENCODER, models_dir=models, device="cpu"
    )
    spec = resolve_encoder(ENCODER)
    results = train_sites(store, settings, PooledTrainer(spec, "cpu"))
    store.close()
    assert [result.reason for result in results] == [None]

    model = load_model(models, "refund")
    assert model.encoder == ENCODER and model.novelty_cutoff is not None
    head = site_dir(models, "refund") / HEAD_FILE
    decider = PooledDecider(spec, "cpu")
    familiar = decider.decide(model, head, YES.format(n=4242))
    assert model.labels[familiar.label] == "true"
    unrelated = decider.decide(model, head, "user: Quantum chromodynamics of heavy-ion collisions")
    assert familiar.novelty < unrelated.novelty
    assert unrelated.novelty > model.novelty_cutoff
    assert is_novel(settings, model, unrelated)
