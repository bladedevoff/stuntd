import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from stuntd.serve.heads import HEAD_CACHE_SIZE
from stuntd.store.db import Example
from stuntd.train.artifacts import EMBEDDINGS_FILE, HEAD_FILE, SiteModel
from stuntd.train.dataset import build_dataset
from stuntd.train.encoders import EncoderSpec, resolve_encoder

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tensors = pytest.importorskip("safetensors.torch")
pooled = pytest.importorskip("stuntd.train.pooled")
PooledDecider = pytest.importorskip("stuntd.serve.pooled").PooledDecider

BOOL = '{"properties":{"refund":{"type":"boolean"}},"type":"object"}'
YES = "user: I was charged twice for invoice {n}, please send the money back."
NO = "user: Where do I find the invoice {n} for last month?"
HIDDEN = 16
TEXT = "user: hello"


def synthetic():
    return [
        Example((YES if i % 2 else NO).format(n=1000 + i), "true" if i % 2 else "false", float(i))
        for i in range(60)
    ]


@pytest.fixture(scope="module")
def encoder_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("encoder")
    words = {word for text in (YES, NO) for word in text.lower().replace("?", " ?").split()}
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *sorted(words), ":", ",", ".", "?"]
    (folder / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    transformers.BertTokenizerFast(str(folder / "vocab.txt")).save_pretrained(folder)
    torch.manual_seed(0)
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    transformers.BertModel(config).save_pretrained(folder)
    return folder


@pytest.fixture(scope="module")
def spec(encoder_folder):
    return resolve_encoder(str(encoder_folder))


def site_model(dataset, spec):
    return SiteModel(
        site=dataset.site,
        kind=dataset.kind,
        field=dataset.field,
        labels=list(dataset.labels),
        base_model="laya",
        encoder=spec.name,
        temperature=1.0,
        threshold=None,
        target_agreement=0.95,
        trained_at=0.0,
        n_train=len(dataset.train),
        n_holdout=len(dataset.holdout),
        agreement=1.0,
        coverage=None,
        covered_agreement=None,
        ece=0.0,
        per_class={},
        confident_errors=[],
        curve=[],
    )


@pytest.fixture(scope="module")
def trained(spec, tmp_path_factory):
    folder = tmp_path_factory.mktemp("site")
    dataset = build_dataset("refund", "boolean", BOOL, synthetic(), 10, 0.25)
    result = pooled.PooledTrainer(spec, device="cpu")(dataset, folder / HEAD_FILE)
    return SimpleNamespace(
        dataset=dataset,
        result=result,
        head=folder / HEAD_FILE,
        model=site_model(dataset, spec),
        decider=PooledDecider(spec, device="cpu"),
    )


def test_trainer_writes_a_linear_head_that_fits_the_encoder(trained):
    head = tensors.load_file(str(trained.head))
    assert set(head) == {pooled.HEAD_WEIGHT, pooled.HEAD_BIAS}
    assert head[pooled.HEAD_WEIGHT].shape == (2, HIDDEN)
    assert head[pooled.HEAD_BIAS].shape == (2,)


def test_trainer_keeps_one_unit_length_vector_per_training_row(trained):
    path = str(trained.head.with_name(EMBEDDINGS_FILE))
    embeddings = tensors.load_file(path)["embeddings"]
    assert embeddings.shape == (len(trained.dataset.train), HIDDEN)
    assert embeddings.float().norm(dim=1) == pytest.approx(1.0, abs=1e-2)


def test_trainer_returns_holdout_logits_and_novelty(trained):
    holdout = len(trained.dataset.holdout)
    assert len(trained.result.logits) == holdout
    assert all(len(row) == 2 for row in trained.result.logits)
    assert trained.result.novelty is not None and len(trained.result.novelty) == holdout
    assert all(-1e-5 <= value <= 2.0 for value in trained.result.novelty)


def test_trainer_learns_a_separable_site(trained):
    pairs = zip(trained.result.logits, trained.dataset.holdout, strict=True)
    assert all(row.index(max(row)) == item.label for row, item in pairs)


def test_trainer_records_the_encoder_length_in_the_layout(trained, spec):
    assert trained.result.layout.max_len == spec.max_length


def test_trainer_is_repeatable(spec, trained, tmp_path):
    again = pooled.PooledTrainer(spec, device="cpu")(trained.dataset, tmp_path / HEAD_FILE)
    assert again.logits == trained.result.logits


def test_decider_agrees_with_the_trainer(trained):
    verdicts = [
        trained.decider.decide(trained.model, trained.head, item.text)
        for item in trained.dataset.holdout
    ]
    expected = [max(range(2), key=row.__getitem__) for row in trained.result.logits]
    assert [verdict.label for verdict in verdicts] == expected
    assert all(0.0 <= verdict.confidence <= 1.0 for verdict in verdicts)
    assert all(sum(verdict.probabilities) == pytest.approx(1.0) for verdict in verdicts)


def test_decider_novelty_matches_the_trainers_for_the_same_text(trained):
    verdict = trained.decider.decide(trained.model, trained.head, trained.dataset.holdout[0].text)
    assert verdict.novelty == pytest.approx(trained.result.novelty[0], abs=1e-3)


def test_decider_finds_an_unrelated_text_further_from_the_training_rows(trained):
    seen = trained.decider.decide(trained.model, trained.head, trained.dataset.train[0].text)
    unseen = trained.decider.decide(trained.model, trained.head, "zzz qqq xxx")
    assert seen.novelty < unseen.novelty


def test_decider_reports_no_novelty_for_a_head_without_embeddings(trained, tmp_path):
    head = tmp_path / HEAD_FILE
    head.write_bytes(trained.head.read_bytes())
    assert trained.decider.decide(trained.model, head, TEXT).novelty is None


@pytest.mark.parametrize(
    "contents",
    [
        {"head.weight": (2, HIDDEN)},
        {"head.weight": (2, HIDDEN), "head.bias": (2,), "extra": (1,)},
        {"scorer.weight": (2, HIDDEN), "scorer.bias": (2,)},
    ],
    ids=["missing-bias", "extra-tensor", "laya-names"],
)
def test_decider_refuses_a_head_with_other_tensor_names(trained, tmp_path, contents):
    head = tmp_path / HEAD_FILE
    tensors.save_file({name: torch.zeros(shape) for name, shape in contents.items()}, str(head))
    with pytest.raises(RuntimeError, match="does not fit the encoder"):
        trained.decider.decide(trained.model, head, TEXT)


@pytest.mark.parametrize(
    ("weight", "bias"),
    [((2, HIDDEN + 1), (2,)), ((3, HIDDEN), (3,)), ((2, HIDDEN), (3,))],
    ids=["wider-encoder", "more-labels", "bias-length"],
)
def test_decider_refuses_a_head_of_another_shape(trained, tmp_path, weight, bias):
    head = tmp_path / HEAD_FILE
    contents = {pooled.HEAD_WEIGHT: torch.zeros(weight), pooled.HEAD_BIAS: torch.zeros(bias)}
    tensors.save_file(contents, str(head))
    with pytest.raises(RuntimeError, match="does not fit the encoder"):
        trained.decider.decide(trained.model, head, TEXT)


def test_decider_refuses_embeddings_of_another_width(trained, tmp_path):
    head = tmp_path / HEAD_FILE
    head.write_bytes(trained.head.read_bytes())
    tensors.save_file({"embeddings": torch.zeros(3, HIDDEN + 1)}, str(tmp_path / EMBEDDINGS_FILE))
    with pytest.raises(RuntimeError, match="does not fit the encoder"):
        trained.decider.decide(trained.model, head, TEXT)


def test_decider_keeps_one_head_per_path(trained, tmp_path):
    head = tmp_path / HEAD_FILE
    head.write_bytes(trained.head.read_bytes())
    trained.decider.decide(trained.model, head, TEXT)
    os.utime(head, (0, 0))
    trained.decider.decide(trained.model, head, TEXT)
    assert len([key for key in trained.decider._heads if key[0] == head]) == 1


def test_decider_caches_no_more_heads_than_the_bound(trained, tmp_path):
    heads = []
    for index in range(HEAD_CACHE_SIZE + 1):
        head = tmp_path / f"site{index}" / HEAD_FILE
        head.parent.mkdir()
        head.write_bytes(trained.head.read_bytes())
        heads.append(head)
        trained.decider.decide(trained.model, head, TEXT)
    assert [key[0] for key in trained.decider._heads] == heads[1:]


def test_warm_installs_the_head_before_the_first_decision(spec, trained):
    decider = PooledDecider(spec, device="cpu")
    decider.warm(trained.model, trained.head)
    assert [key[0] for key in decider._heads] == [trained.head]


def test_lazy_decider_loads_the_encoder_on_first_use(spec, trained, monkeypatch):
    loads = []

    def load(*args):
        loads.append(args)
        return pooled.Encoder(*args)

    monkeypatch.setattr("stuntd.serve.pooled.Encoder", load)
    decider = PooledDecider(spec, device="cpu", lazy=True)
    assert loads == []
    decider.decide(trained.model, trained.head, TEXT)
    decider.decide(trained.model, trained.head, TEXT)
    assert len(loads) == 1


def test_decider_reports_a_missing_head_file(trained, tmp_path):
    with pytest.raises(RuntimeError, match="decision failed"):
        trained.decider.decide(trained.model, tmp_path / HEAD_FILE, TEXT)


def test_encoder_never_runs_remote_code(spec, monkeypatch):
    seen = []
    real = transformers.AutoModel.from_pretrained

    def spy(name, **kwargs):
        seen.append(kwargs)
        return real(name, **kwargs)

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", spy)
    pooled.Encoder(spec, "cpu")
    assert seen == [{"trust_remote_code": False}]


def test_encoder_prepends_the_prefix_and_pools_by_the_spec(spec):
    plain = pooled.Encoder(spec, "cpu").embed([TEXT])
    prefixed = pooled.Encoder(EncoderSpec(spec.name, "mean", 512, "query: "), "cpu").embed([TEXT])
    cls = pooled.Encoder(EncoderSpec(spec.name, "cls", 512, ""), "cpu").embed([TEXT])
    assert not torch.allclose(plain, prefixed)
    assert not torch.allclose(plain, cls)
    assert plain.norm(dim=1) == pytest.approx(1.0, abs=1e-5)
    assert cls.norm(dim=1) == pytest.approx(1.0, abs=1e-5)


def test_encoder_vectors_do_not_depend_on_the_batch(spec):
    encoder = pooled.Encoder(spec, "cpu")
    texts = [TEXT, YES.format(n=1), NO.format(n=2)]
    alone = torch.cat([encoder.embed([text]) for text in texts])
    assert torch.allclose(encoder.embed(texts), alone, atol=1e-5)


def test_the_pooled_decider_never_imports_laya():
    code = "import sys, stuntd.serve.pooled, stuntd.train.pooled; assert 'laya' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
