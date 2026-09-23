import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from stuntd.jev.schema import JevError, kind_for, parse_request
from stuntd.train.artifacts import site_dir

CHOICE = {"type": "choice", "instructions": "route it", "criteria": {"north": None, "south": "s"}}
NOUL = {"type": "noul", "instructions": "is it spam", "criteria": {"true": "spam"}}
SCORE = {"type": "score", "instructions": "how urgent", "criteria": ["calm", "soon", "now"]}


def body(questions, **extra):
    payload = {"state": "the state", "model": "jev-latest", "questions": questions}
    payload.update(extra)
    return json.dumps(payload).encode("utf-8")


def one(name, question):
    return parse_request(body({name: question})).questions[0]


def test_choice_question_is_parsed():
    question = one("direction", CHOICE)
    assert (question.name, question.type, question.labels) == (
        "direction",
        "choice",
        ("north", "south"),
    )
    assert question.instructions == "route it"
    assert question.criteria == CHOICE["criteria"]
    assert question.site == "direction"
    assert kind_for(question) == "choice"


def test_noul_question_is_parsed():
    question = one("spam", NOUL)
    assert question.labels == ("false", "true")
    assert kind_for(question) == "boolean"


def test_noul_question_without_criteria_is_parsed():
    question = one("spam", {"type": "noul", "instructions": "is it spam"})
    assert question.criteria is None
    assert question.labels == ("false", "true")


@pytest.mark.parametrize(
    "question",
    [{"type": "noul"}, {"type": "noul", "instructions": None}],
    ids=["noul-without-instructions", "noul-with-null-instructions"],
)
def test_instructions_are_optional(question):
    parsed = one("billing", question)
    assert parsed.instructions is None
    assert parsed.canonical == '{"criteria":null,"instructions":null,"type":"noul"}'


def test_the_sdk_docstring_body_is_parsed():
    sent = (
        b'{"state":"Hi, I was double charged.","model":"jev-latest",'
        b'"questions":{"billing":{"type":"noul"},'
        b'"tone":{"type":"choice","instructions":"What is the tone?",'
        b'"criteria":{"calm":null,"angry":null}}}}'
    )
    billing, tone = parse_request(sent).questions
    assert (billing.instructions, billing.labels) == (None, ("false", "true"))
    assert (tone.instructions, tone.labels) == ("What is the tone?", ("calm", "angry"))


def test_score_question_is_parsed():
    question = one("urgency", SCORE)
    assert question.labels == ("0", "1", "2")
    assert kind_for(question) == "number"


def test_structured_instructions_are_kept_as_sent():
    instructions = {"task": "route it", "hints": ["left", "right"]}
    question = one(
        "direction", {"type": "choice", "instructions": instructions, "criteria": {"a": None}}
    )
    assert question.instructions == instructions


def test_request_keeps_state_model_and_question_order():
    request = parse_request(body({"direction": CHOICE, "spam": NOUL, "urgency": SCORE}))
    assert (request.state, request.model) == ("the state", "jev-latest")
    assert tuple(q.name for q in request.questions) == ("direction", "spam", "urgency")


def test_unknown_top_level_keys_are_accepted():
    request = parse_request(body({"spam": NOUL}, stream=False, metadata={"app": "demo"}))
    assert request.questions[0].name == "spam"


def test_object_state_is_parsed():
    request = parse_request(body({"spam": NOUL}, state={"board": [1, 2]}))
    assert request.state == {"board": [1, 2]}


def test_canonical_ignores_key_order_while_labels_keep_it():
    forward = one("direction", CHOICE)
    backward = one(
        "direction",
        {"criteria": {"south": "s", "north": None}, "instructions": "route it", "type": "choice"},
    )
    assert forward.canonical == backward.canonical
    assert forward.labels == ("north", "south")
    assert backward.labels == ("south", "north")


def test_canonical_is_compact_sorted_json():
    question = one("urgency", SCORE)
    assert question.canonical == (
        '{"criteria":["calm","soon","now"],"instructions":"how urgent","type":"score"}'
    )


def test_canonical_keeps_non_ascii_unescaped():
    question = one("spam", {"type": "noul", "instructions": "спам?"})
    assert question.canonical == '{"criteria":null,"instructions":"спам?","type":"noul"}'


def test_site_maps_a_namespaced_name_to_a_folder_name():
    question = one("moderation:verdict", CHOICE)
    assert question.site == "moderation.verdict"
    assert site_dir(Path("models"), question.site).name == "moderation.verdict"


@pytest.mark.parametrize(
    "name",
    ["has a space", "..", "слово", "x" * 65, "a/b", ":"],
    ids=["space", "dots", "non-ascii", "too-long", "slash", "colon"],
)
def test_site_falls_back_to_a_hash_of_the_canonical_form(name):
    question = one(name, CHOICE)
    assert question.site == hashlib.sha256(question.canonical.encode("utf-8")).hexdigest()[:16]
    assert site_dir(Path("models"), question.site).name == question.site


def test_two_questions_with_the_same_name_share_a_site():
    other = {"type": "choice", "instructions": "somewhere else", "criteria": {"up": None}}
    here, there = one("direction", CHOICE), one("direction", other)
    assert here.canonical != there.canonical
    assert here.site == there.site


@pytest.mark.parametrize(
    ("questions", "message"),
    [
        ({}, "questions must be a non-empty object"),
        ([], "questions must be a non-empty object"),
        ({"q": "choice"}, 'Question "q" must be an object'),
        (
            {"q": {"type": "rank", "instructions": "x"}},
            'Question "q" must have a type of choice, noul, or score',
        ),
        (
            {"q": {"instructions": "x", "criteria": {"a": None}}},
            'Question "q" must have a type of choice, noul, or score',
        ),
        (
            {"q": {"type": "noul", "instructions": 7}},
            'Question "q" must have instructions as a string, object, or array',
        ),
        (
            {"q": {"type": "noul", "instructions": True}},
            'Question "q" must have instructions as a string, object, or array',
        ),
        (
            {"q": {"type": "choice", "instructions": "x"}},
            'Question "q": choice criteria must be a non-empty object',
        ),
        (
            {"q": {"type": "choice", "instructions": "x", "criteria": {}}},
            'Question "q": choice criteria must be a non-empty object',
        ),
        (
            {"q": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}},
            'Question "q": choice criteria must be a non-empty object',
        ),
        (
            {"q": {"type": "score", "instructions": "x", "criteria": ["only"]}},
            'Question "q": score criteria must contain between 2 and 10 levels',
        ),
        (
            {"q": {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]}},
            'Question "q": score criteria must contain between 2 and 10 levels',
        ),
        (
            {"q": {"type": "score", "instructions": "x", "criteria": {"a": None, "b": None}}},
            'Question "q": score criteria must contain between 2 and 10 levels',
        ),
        (
            {"q": {"type": "noul", "instructions": "x", "criteria": ["true", "false"]}},
            'Question "q": noul criteria must be an object',
        ),
    ],
    ids=[
        "empty",
        "list",
        "not-object",
        "unknown-type",
        "missing-type",
        "numeric-instructions",
        "boolean-instructions",
        "choice-without-criteria",
        "choice-empty-criteria",
        "choice-list-criteria",
        "score-one-level",
        "score-eleven-levels",
        "score-object-criteria",
        "noul-list-criteria",
    ],
)
def test_malformed_questions_are_refused(questions, message):
    with pytest.raises(JevError, match=re.escape(message)):
        parse_request(body(questions))


def test_choice_with_too_many_options_is_refused():
    criteria = {f"option{i}": None for i in range(256)}
    with pytest.raises(JevError) as caught:
        parse_request(body({"q": {"type": "choice", "instructions": "x", "criteria": criteria}}))
    assert caught.value.message == (
        'Question "q": choice criteria cannot contain more than 255 options'
    )


def test_choice_with_the_maximum_options_is_parsed():
    criteria = {f"option{i}": None for i in range(255)}
    question = one("q", {"type": "choice", "instructions": "x", "criteria": criteria})
    assert len(question.labels) == 255


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not json", "request body must be valid JSON"),
        (b'\xff{"state": "s"}', "request body must be valid JSON"),
        (b'["state"]', "request body must be a JSON object"),
        (b'{"model": "m", "questions": {}}', "request must contain state"),
        (
            b'{"state": 7, "model": "m", "questions": {}}',
            "state must be a string, object, or array",
        ),
        (b'{"state": "s", "questions": {}}', "model must be a string"),
        (b'{"state": "s", "model": 1, "questions": {}}', "model must be a string"),
        (b'{"state": "s", "model": "m"}', "request must contain questions"),
    ],
    ids=[
        "not-json",
        "bad-utf8",
        "not-object",
        "no-state",
        "numeric-state",
        "no-model",
        "numeric-model",
        "no-questions",
    ],
)
def test_malformed_bodies_are_refused(payload, message):
    with pytest.raises(JevError, match=re.escape(message)):
        parse_request(payload)


def test_jev_error_carries_a_bad_request_status():
    with pytest.raises(JevError) as caught:
        parse_request(b"not json")
    assert caught.value.status == 400


def test_kind_for_refuses_an_unknown_type():
    question = replace(one("spam", NOUL), type="rank")
    with pytest.raises(ValueError, match="'rank'"):
        kind_for(question)
