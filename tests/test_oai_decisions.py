import json

import pytest
from decisions_helpers import (
    CHOICE,
    CHOICE_ANSWER,
    CHOICE_REQUEST,
    PREDICATE,
    PREDICATE_ANSWER,
    PREDICATE_REQUEST,
    REFUSAL_ANSWER,
    SCORE,
    SCORE_ANSWER,
    SCORE_REQUEST,
    TEXT,
    UNNAMED_FLAG,
    decision_body,
    usage,
)

from stuntd.jev.answer import answer_label
from stuntd.jev.schema import JevError
from stuntd.oai.decisions import answers_body, error_body, parse_request, provider_answers


def parse(payload):
    return parse_request(json.dumps(payload).encode())


def request_with(*questions, **fields):
    return {"model": "gpt-6-luna", "input": TEXT, "questions": list(questions), **fields}


@pytest.mark.parametrize(
    "payload",
    [PREDICATE_REQUEST, CHOICE_REQUEST, SCORE_REQUEST],
    ids=["string-input", "message-with-string-content", "message-with-text-parts"],
)
def test_the_input_becomes_the_text(payload):
    assert parse(payload).text == TEXT


def test_text_parts_of_every_message_are_joined_by_newlines():
    payload = request_with(
        PREDICATE,
        input=[
            {"role": "user", "content": [{"type": "input_text", "text": "one"}]},
            {"role": "user", "content": "two"},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "three"},
                    {"type": "input_text", "text": "four"},
                ],
            },
        ],
    )
    assert parse(payload).text == "one\ntwo\nthree\nfour"


def test_a_predicate_means_a_noul_question():
    question = parse(PREDICATE_REQUEST).questions[0]
    assert (question.type, question.site) == ("noul", "damaged")
    assert question.canonical == (
        '{"criteria":null,"instructions":"Is the product visibly damaged?","type":"noul"}'
    )
    assert question.labels == ("false", "true")


def test_a_choice_means_a_choice_question_keyed_by_its_values():
    request = parse(CHOICE_REQUEST)
    question = request.questions[0]
    assert (question.type, question.site, question.labels) == (
        "choice",
        "route",
        ("billing", "shipping", "other"),
    )
    assert question.criteria == {"billing": "Payment questions", "shipping": None, "other": None}
    assert request.items[0].options == ("billing", "shipping", "other")


def test_a_boolean_choice_value_trains_as_true_and_false():
    request = parse(request_with(UNNAMED_FLAG))
    assert request.questions[0].labels == ("true", "false")
    assert request.items[0].options == (True, False)


def test_a_score_means_a_score_question_with_the_level_descriptions():
    question = parse(SCORE_REQUEST).questions[0]
    assert (question.type, question.labels) == ("score", ("0", "1", "2"))
    assert question.criteria == ["low", "medium: Needs a reply today", "high"]


def test_an_unnamed_question_lands_on_the_hash_of_its_canonical():
    question = parse(request_with(UNNAMED_FLAG)).questions[0]
    assert len(question.site) == 16
    assert int(question.site, 16) >= 0
    assert question.name == "#0"


def test_a_name_outside_the_site_rule_lands_on_the_hash_of_its_canonical():
    question = parse(request_with(dict(PREDICATE, name="is it damaged?"))).questions[0]
    assert len(question.site) == 16
    assert question.name == "is it damaged?"


def test_a_namespaced_name_keeps_its_site_folder():
    assert parse(request_with(dict(PREDICATE, name="shop:damaged"))).questions[0].site == (
        "shop.damaged"
    )


UNLEARNABLE = {
    "image": request_with(
        PREDICATE,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "look"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                ],
            }
        ],
    ),
    "colliding-values": request_with(
        dict(CHOICE, choices=[{"value": "true"}, {"value": True}, {"value": "other"}])
    ),
    "eleven-levels": request_with(
        dict(SCORE, levels=[{"label": f"level {number}"} for number in range(11)])
    ),
    "one-level": request_with(dict(SCORE, levels=[{"label": "only"}])),
    "no-choices": request_with(dict(CHOICE, choices=[])),
    "duplicate-names": request_with(PREDICATE, PREDICATE),
    "unknown-type": request_with(dict(PREDICATE, type="ranking")),
    "no-instructions": request_with({"type": "predicate", "name": "damaged"}),
    "assistant-message": request_with(PREDICATE, input=[{"role": "assistant", "content": "I see"}]),
    "no-questions": request_with(),
    "no-model": {"input": TEXT, "questions": [PREDICATE]},
}


@pytest.mark.parametrize("payload", UNLEARNABLE.values(), ids=UNLEARNABLE.keys())
def test_a_request_stuntd_cannot_learn_from_is_refused(payload):
    with pytest.raises(JevError) as raised:
        parse(payload)
    assert raised.value.status == 400


@pytest.mark.parametrize("body", [b"{", b"[]"], ids=["junk", "not-an-object"])
def test_a_body_that_is_not_a_request_is_refused(body):
    with pytest.raises(JevError):
        parse_request(body)


def test_choice_answers_come_back_with_the_typed_value_in_question_order():
    request = parse(request_with(UNNAMED_FLAG))
    jev = {
        "type": "choice",
        "choice": "true",
        "confidence": 0.7,
        "probabilities": {"false": 0.3, "true": 0.7},
    }
    answer = json.loads(answers_body("stuntd", request, [jev], 5))["answers"][0]
    assert answer == {
        "type": "choice",
        "name": None,
        "choice": True,
        "confidence": 0.7,
        "probabilities": [
            {"value": True, "probability": 0.7},
            {"value": False, "probability": 0.3},
        ],
    }


@pytest.mark.parametrize(
    ("payload", "jev", "expected"),
    [
        (PREDICATE_REQUEST, {"type": "noul", "noul": 0.92}, PREDICATE_ANSWER),
        (
            CHOICE_REQUEST,
            {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.6,
                "probabilities": {"billing": 0.6, "shipping": 0.3, "other": 0.1},
            },
            CHOICE_ANSWER,
        ),
        (
            SCORE_REQUEST,
            {
                "type": "score",
                "score": 1.4,
                "confidence": 0.5,
                "legend": {"0": "low", "1": "medium", "2": "high"},
                "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
            },
            SCORE_ANSWER,
        ),
    ],
    ids=["predicate", "choice", "score"],
)
def test_a_jev_answer_is_written_as_the_decisions_answer(payload, jev, expected):
    body = json.loads(answers_body("stuntd", parse(payload), [jev], 17))
    assert body == {
        "answers": [expected],
        "model": "stuntd",
        "usage": usage(17),
    }


@pytest.mark.parametrize(
    ("payload", "answer", "label"),
    [
        (PREDICATE_REQUEST, PREDICATE_ANSWER, "true"),
        (PREDICATE_REQUEST, dict(PREDICATE_ANSWER, probability=0.1), "false"),
        (CHOICE_REQUEST, CHOICE_ANSWER, "billing"),
        (SCORE_REQUEST, SCORE_ANSWER, "2"),
    ],
    ids=["predicate-true", "predicate-false", "choice", "score"],
)
def test_a_provider_answer_reads_back_as_the_label_it_stands_for(payload, answer, label):
    request = parse(payload)
    answers = provider_answers(request, json.loads(decision_body([answer])))
    assert answers is not None
    assert answer_label(answers[0]) == label
    assert label in request.questions[0].labels


def test_a_boolean_choice_reads_back_as_its_label():
    request = parse(request_with(UNNAMED_FLAG))
    answered = {"answers": [dict(CHOICE_ANSWER, name=None, choice=False)]}
    answers = provider_answers(request, answered)
    assert answers is not None
    assert answer_label(answers[0]) == "false"


def test_a_refusal_yields_no_label():
    request = parse(PREDICATE_REQUEST)
    assert provider_answers(request, json.loads(decision_body([REFUSAL_ANSWER]))) == [None]


@pytest.mark.parametrize(
    "answered",
    [
        {},
        {"answers": {"damaged": PREDICATE_ANSWER}},
        {"answers": []},
        {"answers": [PREDICATE_ANSWER, PREDICATE_ANSWER]},
    ],
    ids=["missing", "not-a-list", "short", "long"],
)
def test_a_response_without_one_answer_per_question_reads_as_none(answered):
    assert provider_answers(parse(PREDICATE_REQUEST), answered) is None


@pytest.mark.parametrize(
    "answer",
    [
        "yes",
        {"type": "choice", "choice": 3},
        {"type": "score", "probabilities": [{"probability": 1.0}]},
        {"type": "predicate"},
        {"type": "ranking"},
    ],
    ids=["not-an-object", "choice-of-a-number", "score-without-value", "no-probability", "unknown"],
)
def test_a_malformed_provider_answer_reads_as_none(answer):
    assert provider_answers(parse(PREDICATE_REQUEST), {"answers": [answer]}) == [None]


def test_the_error_body_has_the_openai_shape():
    assert json.loads(error_body("nope")) == {
        "error": {"message": "nope", "type": "invalid_request_error", "param": None, "code": None}
    }
