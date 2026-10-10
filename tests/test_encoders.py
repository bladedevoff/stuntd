import json

import pytest

from stuntd.train.encoders import ALLOWED_MODEL_TYPES, EncoderSpec, resolve_encoder

BERT = {"model_type": "bert", "hidden_size": 8}


def encoder_folder(tmp_path, config=BERT, **files):
    folder = tmp_path / "encoder"
    (folder / "1_Pooling").mkdir(parents=True)
    names = {
        "pooling": "1_Pooling/config.json",
        "sentence_bert": "sentence_bert_config.json",
        "sentence_transformers": "config_sentence_transformers.json",
    }
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for key, content in files.items():
        (folder / names[key]).write_text(json.dumps(content), encoding="utf-8")
    return folder


class FakeHub:
    def __init__(self, files):
        self.files = files

    def __call__(self, name, filename):
        content = self.files.get(filename)
        return None if content is None else json.dumps(content)


@pytest.mark.parametrize("model_type", ALLOWED_MODEL_TYPES)
def test_every_allowed_architecture_resolves(tmp_path, model_type):
    folder = encoder_folder(tmp_path, {"model_type": model_type})
    assert resolve_encoder(str(folder)) == EncoderSpec(str(folder), "mean", 512, "")


def test_pooling_is_mean_when_the_folder_has_no_pooling_file(tmp_path):
    assert resolve_encoder(str(encoder_folder(tmp_path))).pooling == "mean"


@pytest.mark.parametrize(
    ("mode", "pooling"),
    [("pooling_mode_mean_tokens", "mean"), ("pooling_mode_cls_token", "cls")],
    ids=["mean", "cls"],
)
def test_pooling_follows_the_sentence_transformers_file(tmp_path, mode, pooling):
    folder = encoder_folder(tmp_path, pooling={mode: True, "pooling_mode_max_tokens": False})
    assert resolve_encoder(str(folder)).pooling == pooling


@pytest.mark.parametrize(
    "enabled",
    [
        ["pooling_mode_max_tokens"],
        ["pooling_mode_lasttoken"],
        ["pooling_mode_mean_tokens", "pooling_mode_cls_token"],
        ["pooling_mode_mean_tokens", "pooling_mode_weightedmean_tokens"],
        [],
    ],
    ids=["max", "last-token", "mean-and-cls", "mean-and-weighted", "none"],
)
def test_pooling_other_than_mean_or_cls_is_refused(tmp_path, enabled):
    folder = encoder_folder(tmp_path, pooling=dict.fromkeys(enabled, True))
    with pytest.raises(ValueError, match="only mean and cls pooling"):
        resolve_encoder(str(folder))


def test_max_length_comes_from_the_sentence_bert_config(tmp_path):
    folder = encoder_folder(tmp_path, sentence_bert={"max_seq_length": 256})
    assert resolve_encoder(str(folder)).max_length == 256


@pytest.mark.parametrize("value", [0, -1, "256", True, None], ids=repr)
def test_max_length_that_is_not_a_positive_integer_is_refused(tmp_path, value):
    folder = encoder_folder(tmp_path, sentence_bert={"max_seq_length": value})
    with pytest.raises(ValueError, match="max_seq_length"):
        resolve_encoder(str(folder))


def test_prefix_comes_from_the_query_prompt_of_the_sentence_transformers_config(tmp_path):
    folder = encoder_folder(
        tmp_path, sentence_transformers={"prompts": {"query": "search: ", "document": "doc: "}}
    )
    assert resolve_encoder(str(folder)).prefix == "search: "


@pytest.mark.parametrize(
    "content", [{"prompts": {"document": "doc: "}}, {"prompts": {"query": ""}}, {}], ids=repr
)
def test_prefix_is_empty_without_a_query_prompt(tmp_path, content):
    folder = encoder_folder(tmp_path, sentence_transformers=content)
    assert resolve_encoder(str(folder)).prefix == ""


@pytest.mark.parametrize(
    ("name", "prefix"),
    [
        ("intfloat/multilingual-e5-base", "query: "),
        ("intfloat/multilingual-e5-small", "query: "),
        ("sentence-transformers/all-MiniLM-L6-v2", ""),
    ],
)
def test_presets_carry_their_known_prefix(monkeypatch, name, prefix):
    monkeypatch.setattr("stuntd.train.encoders._read", FakeHub({"config.json": BERT}))
    assert resolve_encoder(name) == EncoderSpec(name, "mean", 512, prefix)


def test_a_preset_prefix_wins_over_the_sentence_transformers_config(monkeypatch):
    hub = FakeHub(
        {"config.json": BERT, "config_sentence_transformers.json": {"prompts": {"query": "x"}}}
    )
    monkeypatch.setattr("stuntd.train.encoders._read", hub)
    assert resolve_encoder("intfloat/multilingual-e5-small").prefix == "query: "


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "bert", "auto_map": {"AutoModel": "modeling.Custom"}},
        {"model_type": "bert", "auto_map": {}},
    ],
    ids=["custom-code", "empty-auto-map"],
)
def test_a_config_with_remote_code_is_refused(tmp_path, config):
    with pytest.raises(ValueError, match="auto_map"):
        resolve_encoder(str(encoder_folder(tmp_path, config)))


@pytest.mark.parametrize("config", [{"model_type": "llama"}, {}], ids=["llama", "no-type"])
def test_an_architecture_outside_the_allowlist_is_refused(tmp_path, config):
    with pytest.raises(ValueError, match="model_type"):
        resolve_encoder(str(encoder_folder(tmp_path, config)))


def test_a_folder_without_a_config_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no config.json"):
        resolve_encoder(str(tmp_path))


def test_a_file_that_is_not_json_is_named_in_the_error(tmp_path):
    folder = encoder_folder(tmp_path)
    (folder / "sentence_bert_config.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="sentence_bert_config.json is not valid JSON"):
        resolve_encoder(str(folder))


def test_a_path_that_does_not_exist_is_refused(tmp_path):
    pytest.importorskip("huggingface_hub")
    missing = str(tmp_path / "nowhere")
    with pytest.raises(ValueError, match="neither a folder nor a Hugging Face model"):
        resolve_encoder(missing)
