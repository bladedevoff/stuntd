from __future__ import annotations

import json

import pytest

from stuntd.jev.answer import (
    choice_answer,
    error_body,
    noul_answer,
    response_body,
    score_answer,
)


def test_choice_answer_shape():
    answer = choice_answer(["allow", "block"], [0.25, 0.75], 0.75)
    assert answer == {
        "type": "choice",
        "choice": "block",
        "confidence": 0.75,
        "probabilities": {"allow": 0.25, "block": 0.75},
    }


def test_choice_answer_ties_pick_first_label():
    answer = choice_answer(["allow", "block"], [0.5, 0.5], 0.5)
    assert answer["choice"] == "allow"


def test_choice_answer_probabilities_key_order_matches_labels():
    answer = choice_answer(["b", "a", "c"], [0.1, 0.2, 0.7], 0.7)
    assert list(answer["probabilities"].keys()) == ["b", "a", "c"]


def test_choice_answer_unicode_labels():
    answer = choice_answer(["да", "нет"], [0.9, 0.1], 0.9)
    assert answer["choice"] == "да" and answer["probabilities"] == {"да": 0.9, "нет": 0.1}


def test_choice_answer_rounds_probabilities_to_four_decimals():
    answer = choice_answer(["a", "b"], [0.123456, 0.876544], 0.876544)
    assert answer["probabilities"] == {"a": 0.1235, "b": 0.8765}
    assert answer["confidence"] == 0.8765


def test_noul_answer_shape():
    answer = noul_answer([0.3, 0.7])
    assert answer == {"type": "noul", "noul": 0.7}


def test_noul_answer_rounds_to_four_decimals():
    answer = noul_answer([0.666666, 0.333334])
    assert answer == {"type": "noul", "noul": 0.3333}


def test_score_answer_shape():
    answer = score_answer(["low", "mid", "high"], [0.2, 0.3, 0.5], 0.5)
    assert answer == {
        "type": "score",
        "score": pytest.approx(1.3, abs=1e-9),
        "confidence": 0.5,
        "legend": {"0": "low", "1": "mid", "2": "high"},
        "probabilities": {"0": 0.2, "1": 0.3, "2": 0.5},
    }


def test_score_answer_rounds_score_to_four_decimals():
    answer = score_answer(["a", "b", "c"], [0.111111, 0.333333, 0.555556], 0.555556)
    assert answer["score"] == round(0 * 0.111111 + 1 * 0.333333 + 2 * 0.555556, 4)


def test_score_answer_unicode_levels():
    answer = score_answer(["низкий", "высокий"], [0.4, 0.6], 0.6)
    assert answer["legend"] == {"0": "низкий", "1": "высокий"}


def test_score_answer_keeps_level_objects_as_sent():
    level = {"label": "mid", "hint": None}
    answer = score_answer(["low", level], [0.4, 0.6], 0.6)
    assert answer["legend"] == {"0": "low", "1": level}


def test_response_body_is_compact_utf8_json():
    answers = {
        "verdict": {
            "type": "choice",
            "choice": "allow",
            "confidence": 0.9,
            "probabilities": {"allow": 0.9, "block": 0.1},
        }
    }
    raw = response_body("модель", answers, 42)
    assert isinstance(raw, bytes)
    assert b": " not in raw and "модель".encode() in raw
    assert json.loads(raw) == {
        "model": "модель",
        "answers": answers,
        "usage": {"input_tokens": 42, "output_tokens": 0},
    }


def test_response_body_parses_with_typesafe_sdk():
    pytest.importorskip("typesafe_sdk")
    try:
        from typesafe_sdk._core.response_types import SystemOneResponse
    except ImportError:
        from typesafe_sdk._schemas.models import SystemOneResponse
    mid_level = {"label": "mid", "hint": None}
    answers = {
        "verdict": choice_answer(["да", "нет"], [0.9, 0.1], 0.9),
        "flag": noul_answer([0.3, 0.7]),
        "quality": score_answer(["low", mid_level, "high"], [0.2, 0.3, 0.5], 0.5),
    }
    raw = response_body("stuntd", answers, 5)
    parsed = SystemOneResponse.model_validate_json(raw)
    assert parsed.model == "stuntd" and parsed.usage.input_tokens == 5

    choice = parsed.answers["verdict"]
    assert choice.choice == "да" and choice.confidence == 0.9
    assert choice.probabilities == {"да": 0.9, "нет": 0.1}

    noul = parsed.answers["flag"]
    assert noul.noul == 0.7

    score = parsed.answers["quality"]
    assert score.score == pytest.approx(1.3) and score.confidence == 0.5
    assert score.legend == {0: "low", 1: mid_level, 2: "high"}
    assert score.probabilities == {0: 0.2, 1: 0.3, 2: 0.5}


def test_error_body_shape():
    raw = error_body("bad request")
    assert isinstance(raw, bytes)
    assert json.loads(raw) == {"error": {"message": "bad request"}}
