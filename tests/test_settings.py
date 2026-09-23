import re
from pathlib import Path

import pytest

from stuntd.settings import (
    CONFIG_TEMPLATE,
    Settings,
    config_path,
    database_path,
    load_settings,
    models_path,
)


def test_defaults_without_a_file(data_dir):
    settings = load_settings(None, {"upstream": "http://up"})
    assert settings == Settings(upstream="http://up")
    assert config_path() == data_dir / "stuntd.toml"


def test_file_values_are_read(tmp_path, data_dir):
    cfg = tmp_path / "stuntd.toml"
    cfg.write_text(
        "upstream = \"http://file\"\nport = 9000\n\n[redaction]\nenabled = false\npatterns = ['ACC-\\d{6}']\n\n"
        '[storage]\nmax_rows = 10\nmax_age_days = 1\ndb_path = "captures.db"\n',
        encoding="utf-8",
    )
    settings = load_settings(cfg)
    assert settings.upstream == "http://file" and settings.port == 9000
    assert settings.redact is False and settings.redaction_patterns == ["ACC-\\d{6}"]
    assert settings.max_rows == 10 and settings.max_age_days == 1
    assert settings.db_path == data_dir / "captures.db"


def test_absolute_db_path_is_kept(tmp_path, data_dir):
    target = tmp_path / "elsewhere" / "captures.db"
    cfg = tmp_path / "stuntd.toml"
    cfg.write_text(f"upstream = \"http://x\"\n[storage]\ndb_path = '{target}'\n", encoding="utf-8")
    assert load_settings(cfg).db_path == target


def test_unknown_override_is_rejected():
    with pytest.raises(ValueError, match="unknown override prot"):
        load_settings(None, {"upstream": "http://up", "prot": 1})


def test_overrides_beat_the_file(tmp_path):
    cfg = tmp_path / "stuntd.toml"
    cfg.write_text('upstream = "http://file"\nport = 9000\n', encoding="utf-8")
    settings = load_settings(cfg, {"upstream": "http://flag", "port": None})
    assert settings.upstream == "http://flag" and settings.port == 9000


def test_an_empty_upstream_is_accepted():
    assert load_settings(None, {}) == Settings()


def test_an_empty_upstream_stands_beside_a_jev_provider(tmp_path):
    cfg = tmp_path / "stuntd.toml"
    cfg.write_text('upstream = ""\n[jev]\nupstream = "https://api.typesafe.ai"\n', encoding="utf-8")
    settings = load_settings(cfg)
    assert settings.upstream == "" and settings.jev_upstream == "https://api.typesafe.ai"


def test_unknown_key_and_wrong_type_are_reported(tmp_path):
    bad_key = tmp_path / "a.toml"
    bad_key.write_text('upstream = "http://x"\nprot = 1\n', encoding="utf-8")
    with pytest.raises(ValueError, match="prot"):
        load_settings(bad_key)
    bad_type = tmp_path / "b.toml"
    bad_type.write_text('upstream = "http://x"\n[storage]\nmax_rows = "many"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="max_rows"):
        load_settings(bad_type)


def test_template_parses_to_defaults(tmp_path):
    cfg = tmp_path / "stuntd.toml"
    cfg.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    assert load_settings(cfg, {"upstream": "http://up"}) == Settings(upstream="http://up")


def test_malformed_toml_names_the_file(tmp_path):
    cfg = tmp_path / "broken.toml"
    cfg.write_text("upstream = ", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"{cfg}:")):
        load_settings(cfg)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"port": 0}, "port must be between 1 and 65535, got 0"),
        ({"port": 99999}, "port must be between 1 and 65535, got 99999"),
        ({"max_rows": 0}, "max_rows must be at least 1, got 0"),
        ({"max_age_days": -1}, "max_age_days must be at least 1, got -1"),
        ({"min_examples": 1}, "min_examples must be at least 2, got 1"),
        ({"holdout": 0.0}, "holdout must be above 0 and at most 0.5, got 0.0"),
        ({"holdout": 0.6}, "holdout must be above 0 and at most 0.5, got 0.6"),
        ({"target_agreement": 1.5}, "target_agreement must be above 0 and at most 1, got 1.5"),
        ({"epochs": 0}, "epochs must be at least 1, got 0"),
        ({"device": "tpu"}, "device must be one of auto, cpu, cuda, mps, got 'tpu'"),
        ({"check_share": 1.0}, "check_share must be at least 0 and below 1, got 1.0"),
        ({"window": 0}, "window must be at least 1, got 0"),
        ({"min_window": 0}, "min_window must be between 1 and 100, got 0"),
        ({"min_window": 200, "window": 100}, "min_window must be between 1 and 100, got 200"),
        ({"auto_promote_after_hours": -1}, "auto_promote_after_hours must be at least 0, got -1"),
        ({"cache_size": -1}, "cache_size must be at least 0, got -1"),
    ],
    ids=[
        "port-zero",
        "port-too-high",
        "no-rows-kept",
        "negative-age",
        "min-examples",
        "holdout-zero",
        "holdout-half",
        "target",
        "epochs",
        "device",
        "check-share",
        "window",
        "min-window-zero",
        "min-window-over-window",
        "promote-hours",
        "cache-size",
    ],
)
def test_out_of_range_values_are_rejected(override, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        load_settings(None, {"upstream": "http://up", **override})


def test_database_path_uses_the_data_directory_unless_configured(tmp_path, data_dir):
    assert database_path(Settings(upstream="http://up")) == data_dir / "captures.sqlite"
    configured = tmp_path / "elsewhere" / "captures.db"
    assert database_path(Settings(upstream="http://up", db_path=configured)) == configured


def test_training_section_is_read(data_dir):
    data_dir.mkdir(parents=True)
    path = data_dir / "stuntd.toml"
    path.write_text(
        'upstream = "http://up"\n'
        "[training]\nmin_examples = 50\nholdout = 0.25\ntarget_agreement = 0.95\n"
        'base_model = "local/laya"\nepochs = 1\ndevice = "cpu"\n[storage]\nmodels_dir = "m"\n',
        encoding="utf-8",
    )
    settings = load_settings(path)
    assert (settings.min_examples, settings.holdout, settings.target_agreement) == (50, 0.25, 0.95)
    assert (settings.base_model, settings.epochs, settings.device) == ("local/laya", 1, "cpu")
    assert settings.models_dir == data_dir / "m"


def test_models_path_defaults_inside_the_data_dir(data_dir):
    assert models_path(Settings()) == data_dir / "models"
    assert models_path(Settings(models_dir=Path("/x/models"))) == Path("/x/models")


def test_absolute_models_dir_is_kept(tmp_path, data_dir):
    target = tmp_path / "elsewhere" / "models"
    cfg = tmp_path / "stuntd.toml"
    cfg.write_text(
        f"upstream = \"http://x\"\n[storage]\nmodels_dir = '{target}'\n", encoding="utf-8"
    )
    assert load_settings(cfg).models_dir == target


@pytest.mark.parametrize("key", ["holdout", "target_agreement"], ids=["holdout", "agreement"])
def test_a_whole_number_share_is_rejected(tmp_path, key):
    cfg = tmp_path / "stuntd.toml"
    cfg.write_text(f'upstream = "http://x"\n[training]\n{key} = 1\n', encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"training.{key} must be float, got int")):
        load_settings(cfg)


def test_encoder_cache_is_on_unless_the_file_turns_it_off(tmp_path):
    cfg = tmp_path / "stuntd.toml"
    assert load_settings(cfg).cache_encoder is True
    cfg.write_text("[training]\ncache_encoder = false\n", encoding="utf-8")
    assert load_settings(cfg).cache_encoder is False


@pytest.mark.parametrize(
    ("line", "expected"),
    [("", 4096), ("cache_max_mb = 512\n", 512), ("cache_max_mb = 0\n", 0)],
    ids=["default", "custom", "zero-means-never-cached"],
)
def test_cache_budget_comes_from_the_file(tmp_path, line, expected):
    cfg = tmp_path / "stuntd.toml"
    cfg.write_text(f"[training]\n{line}", encoding="utf-8")
    assert load_settings(cfg).cache_max_mb == expected


def test_a_negative_cache_budget_is_refused(tmp_path):
    cfg = tmp_path / "stuntd.toml"
    cfg.write_text("[training]\ncache_max_mb = -1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cache_max_mb"):
        load_settings(cfg)


def test_template_documents_training():
    assert "[training]" in CONFIG_TEMPLATE and "# target_agreement = 0.99" in CONFIG_TEMPLATE
    assert "# cache_encoder = true" in CONFIG_TEMPLATE
    assert "# cache_max_mb = 4096" in CONFIG_TEMPLATE


def test_serving_section_is_read(tmp_path):
    cfg = tmp_path / "stuntd.toml"
    cfg.write_text(
        'upstream = "http://up"\n'
        "[serving]\ncheck_share = 0.1\nwindow = 50\nmin_window = 10\n"
        "auto_promote = true\nauto_promote_after_hours = 6\ncache_size = 0\n",
        encoding="utf-8",
    )
    settings = load_settings(cfg)
    assert (settings.check_share, settings.window, settings.min_window) == (0.1, 50, 10)
    assert (settings.auto_promote, settings.auto_promote_after_hours) == (True, 6)
    assert settings.cache_size == 0


def test_template_documents_serving():
    assert "[serving]" in CONFIG_TEMPLATE and "# check_share = 0.02" in CONFIG_TEMPLATE


def test_jev_section_is_read(tmp_path):
    cfg = tmp_path / "stuntd.toml"
    cfg.write_text(
        'upstream = "http://up"\nlearn = false\n'
        '[jev]\nupstream = "https://api.typesafe.ai"\nrequire_key = true\n'
        'model_name = "stuntd-1.0"\n',
        encoding="utf-8",
    )
    settings = load_settings(cfg)
    assert settings.learn is False
    assert settings.jev_upstream == "https://api.typesafe.ai"
    assert settings.jev_require_key is True
    assert settings.jev_model_name == "stuntd-1.0"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            '[jev]\nupstream = "api.typesafe.ai"\n',
            "jev.upstream must be an origin like https://api.typesafe.ai, got 'api.typesafe.ai'",
        ),
        (
            '[jev]\nupstream = "https://api.typesafe.ai/v1"\n',
            "jev.upstream must be an origin like https://api.typesafe.ai, "
            "got 'https://api.typesafe.ai/v1'",
        ),
        (
            '[jev]\nupstream = "https://api.typesafe.ai?key=1"\n',
            "jev.upstream must be an origin like https://api.typesafe.ai, "
            "got 'https://api.typesafe.ai?key=1'",
        ),
        (
            '[jev]\nupstream = "https://user:pw@api.typesafe.ai"\n',
            "jev.upstream must be an origin like https://api.typesafe.ai, "
            "got 'https://user:pw@api.typesafe.ai'",
        ),
        (
            '[jev]\nupstream = "https://[::1"\n',
            "jev.upstream must be an origin like https://api.typesafe.ai, got 'https://[::1'",
        ),
        (
            '[jev]\nmodel_name = "my model"\n',
            "jev.model_name must be 1 to 64 letters, digits, dot, dash or underscore, "
            "got 'my model'",
        ),
        ('learn = "yes"\n', "learn must be bool, got str"),
    ],
    ids=[
        "upstream-without-scheme",
        "upstream-with-path",
        "upstream-with-query",
        "upstream-with-userinfo",
        "upstream-with-broken-ipv6",
        "model-name-with-space",
        "learn-not-bool",
    ],
)
def test_jev_values_are_rejected(tmp_path, body, message):
    cfg = tmp_path / "stuntd.toml"
    cfg.write_text(f'upstream = "http://up"\n{body}', encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        load_settings(cfg)


def test_template_documents_jev():
    assert "# learn = true" in CONFIG_TEMPLATE
    assert "[jev]" in CONFIG_TEMPLATE and '# model_name = "stuntd"' in CONFIG_TEMPLATE
