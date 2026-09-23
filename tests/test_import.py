import json
import sqlite3

import pytest

from stuntd.cli import main
from stuntd.decisions.schema import detect_schema

SPAM_SCHEMA = {"type": "object", "properties": {"spam": {"type": "boolean"}}}


def row(text, answer):
    return json.dumps({"text": text, "answer": answer})


def write_rows(tmp_path, lines, name="rows.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def spam_rows(tmp_path, count=10):
    return write_rows(
        tmp_path, [row(f"ticket {i}", "true" if i % 2 else "false") for i in range(count)]
    )


def write_config(data_dir, body):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "stuntd.toml").write_text(body, encoding="utf-8")


def stored(data_dir):
    connection = sqlite3.connect(data_dir / "captures.sqlite")
    try:
        return connection.execute(
            "select site, schema_canonical, kind, input_text, answer, model, latency_ms,"
            " prompt_tokens, completion_tokens from captures order by id"
        ).fetchall()
    finally:
        connection.close()


def fake_trainer_factory(settings):
    def trainer(dataset, head_path):
        head_path.write_bytes(b"h")
        return [[2.0, 0.0] if item.label == 0 else [0.0, 2.0] for item in dataset.holdout]

    return trainer


def test_import_shows_in_status_and_trains(data_dir, tmp_path, capsys, monkeypatch):
    write_config(data_dir, "[training]\nmin_examples = 4\nholdout = 0.3\n")
    path = spam_rows(tmp_path)
    assert main(["import", "spam", path, "--kind", "boolean"]) == 0
    assert "imported 10 rows into spam" in capsys.readouterr().out
    assert main(["status"]) == 0
    assert capsys.readouterr().out.splitlines()[1].split()[:3] == ["spam", "collect", "10"]
    monkeypatch.setattr("stuntd.cli._make_trainer", fake_trainer_factory)
    assert main(["train"]) == 0
    assert "spam  trained" in capsys.readouterr().out
    assert (data_dir / "models" / "spam" / "meta.json").exists()


def test_imported_row_matches_a_captured_one(data_dir, tmp_path, capsys):
    path = write_rows(tmp_path, [row("ticket 1", "true")])
    assert main(["import", "spam", path, "--kind", "boolean"]) == 0
    capsys.readouterr()
    request = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "spam", "schema": SPAM_SCHEMA},
        }
    }
    captured = detect_schema(request)
    assert stored(data_dir) == [
        ("spam", captured.canonical, captured.kind, "ticket 1", "true", "import", 0, None, None)
    ]


def test_labels_build_an_enum_schema(data_dir, tmp_path, capsys):
    path = write_rows(tmp_path, [row("ticket 1", "deny")])
    assert main(["import", "tickets", path, "--labels", "refund, deny"]) == 0
    capsys.readouterr()
    site, canonical, kind = stored(data_dir)[0][:3]
    assert site == "tickets" and kind == "choice"
    assert json.loads(canonical) == {
        "type": "object",
        "properties": {"tickets": {"type": "string", "enum": ["refund", "deny"]}},
    }


def test_a_namespaced_site_is_imported_under_its_folder_name(data_dir, tmp_path, capsys):
    path = write_rows(tmp_path, [row("ticket 1", "deny")])
    assert main(["import", "moderation:verdict", path, "--labels", "refund,deny"]) == 0
    assert "imported 1 rows into moderation.verdict" in capsys.readouterr().out
    site, canonical = stored(data_dir)[0][:2]
    assert site == "moderation.verdict"
    assert list(json.loads(canonical)["properties"]) == ["moderation.verdict"]


@pytest.mark.parametrize(
    ("labels", "enum"),
    [
        ("refund,refund,deny", ["refund", "deny"]),
        ("refund, deny ,refund", ["refund", "deny"]),
        ("refund,,deny", ["refund", "deny"]),
    ],
    ids=["repeated", "repeated-with-spaces", "empty-fragment"],
)
def test_labels_are_deduplicated(data_dir, tmp_path, capsys, labels, enum):
    path = write_rows(tmp_path, [row("ticket 1", "deny")])
    assert main(["import", "tickets", path, "--labels", labels]) == 0
    capsys.readouterr()
    canonical = stored(data_dir)[0][1]
    assert json.loads(canonical)["properties"]["tickets"]["enum"] == enum


def test_a_byte_order_mark_is_not_part_of_the_first_row(data_dir, tmp_path, capsys):
    path = tmp_path / "bom.jsonl"
    path.write_text(row("ticket 1", "true") + "\n", encoding="utf-8-sig")
    assert main(["import", "spam", str(path), "--kind", "boolean"]) == 0
    assert "imported 1 rows into spam" in capsys.readouterr().out
    assert stored(data_dir)[0][3] == "ticket 1"


def test_schema_flag_keeps_the_field_name(data_dir, tmp_path, capsys):
    path = write_rows(tmp_path, [row("ticket 1", "7")])
    schema = json.dumps({"type": "object", "properties": {"score": {"type": "integer"}}})
    assert main(["import", "scores", path, "--schema", schema]) == 0
    capsys.readouterr()
    canonical, kind = stored(data_dir)[0][1:3]
    assert kind == "number" and json.loads(canonical)["properties"]["score"]["type"] == "integer"


def test_blank_lines_are_skipped(data_dir, tmp_path, capsys):
    path = write_rows(tmp_path, [row("a", "true"), "", "   ", row("b", "false")])
    assert main(["import", "spam", path, "--kind", "boolean"]) == 0
    assert "imported 2 rows into spam" in capsys.readouterr().out
    assert len(stored(data_dir)) == 2


def test_import_redacts_the_text(data_dir, tmp_path, capsys):
    path = write_rows(tmp_path, [row("write to me@example.com", "true")])
    assert main(["import", "spam", path, "--kind", "boolean"]) == 0
    capsys.readouterr()
    assert stored(data_dir)[0][3] == "write to [email]"


@pytest.mark.parametrize(
    ("flags", "lines", "message"),
    [
        (
            ["--labels", "refund,deny"],
            [row("a", "refund"), row("b", "maybe")],
            "line 2: answer 'maybe' not in labels",
        ),
        (
            ["--kind", "boolean"],
            [row("a", "true"), row("b", "yes")],
            "line 2: answer 'yes' is not true or false",
        ),
        (
            ["--kind", "number"],
            [row("a", "1"), row("b", "many")],
            "line 2: answer 'many' is not a number",
        ),
        (
            ["--kind", "number"],
            [row("a", "nan")],
            "line 1: answer 'nan' is not a finite number",
        ),
        (["--kind", "boolean"], [row("a", "true"), "{oops}"], "line 2: invalid JSON"),
        (["--kind", "boolean"], [row("a", "true"), "[1]"], "line 2: expected an object"),
        (
            ["--kind", "boolean"],
            [row("", "true")],
            "line 1: text must be a non-empty string",
        ),
        (
            ["--kind", "boolean"],
            [json.dumps({"text": "a", "answer": True})],
            "line 1: answer must be a string",
        ),
    ],
    ids=[
        "outside-labels",
        "not-boolean",
        "not-number",
        "not-finite",
        "broken-json",
        "not-an-object",
        "empty-text",
        "answer-not-string",
    ],
)
def test_invalid_line_writes_nothing(data_dir, tmp_path, capsys, flags, lines, message):
    path = write_rows(tmp_path, lines)
    assert main(["import", "spam", path, *flags]) == 1
    assert message in capsys.readouterr().err
    assert not (data_dir / "captures.sqlite").exists()


def test_learn_false_refuses_the_import(data_dir, tmp_path, capsys):
    write_config(data_dir, "learn = false\n")
    path = spam_rows(tmp_path)
    assert main(["import", "spam", path, "--kind", "boolean"]) == 1
    assert "stuntd: learning is off in" in capsys.readouterr().err
    assert not (data_dir / "captures.sqlite").exists()


def test_invalid_site_name_is_refused(data_dir, tmp_path, capsys):
    path = spam_rows(tmp_path)
    assert main(["import", "../escape", path, "--kind", "boolean"]) == 2
    assert "stuntd: invalid site name '../escape'" in capsys.readouterr().err
    assert not (data_dir / "captures.sqlite").exists()


def test_choice_without_labels_is_refused(data_dir, tmp_path, capsys):
    path = write_rows(tmp_path, [row("a", "refund")])
    assert main(["import", "tickets", path]) == 2
    assert "stuntd: a choice site needs --labels" in capsys.readouterr().err


def test_kind_disagreeing_with_the_schema_is_refused(data_dir, tmp_path, capsys):
    path = write_rows(tmp_path, [row("a", "true")])
    schema = json.dumps(SPAM_SCHEMA)
    assert main(["import", "spam", path, "--schema", schema, "--kind", "number"]) == 2
    assert "stuntd: --schema describes a boolean field, not number" in capsys.readouterr().err


def test_unusable_schema_is_refused(data_dir, tmp_path, capsys):
    path = write_rows(tmp_path, [row("a", "true")])
    assert main(["import", "spam", path, "--schema", "{oops}"]) == 2
    assert "stuntd: --schema is not valid JSON" in capsys.readouterr().err
    assert main(["import", "spam", path, "--schema", '{"type": "object"}']) == 2
    assert "stuntd: schema must describe an object with one typed field" in capsys.readouterr().err


def test_missing_file_is_reported(data_dir, tmp_path, capsys):
    missing = tmp_path / "absent.jsonl"
    assert main(["import", "spam", str(missing), "--kind", "boolean"]) == 2
    assert "stuntd:" in capsys.readouterr().err
    assert not (data_dir / "captures.sqlite").exists()
