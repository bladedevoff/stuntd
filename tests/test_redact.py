import time

import pytest

from stuntd.store.redact import Redactor


def test_builtin_email_and_phone():
    r = Redactor()
    assert (
        r.apply("mail me at bob@example.com or +1 (555) 010-2233")
        == "mail me at [email] or [phone]"
    )


def test_short_numbers_survive():
    assert Redactor().apply("order 12345 shipped") == "order 12345 shipped"


def test_extra_pattern():
    r = Redactor(extra_patterns=[r"ACC-\d{6}"])
    assert r.apply("account ACC-123456 flagged") == "account [redacted] flagged"


def test_invalid_pattern_is_reported():
    with pytest.raises(ValueError, match=r"bad redaction pattern '\('"):
        Redactor(extra_patterns=["("])


def test_dates_and_spaced_digits_survive():
    text = "release 2026-09-21, ids 1 2 3 4 5 done"
    assert Redactor().apply(text) == text


def test_eleven_digit_number_is_still_a_phone():
    assert Redactor().apply("call +1 (555) 010-2233") == "call [phone]"


@pytest.mark.parametrize("pattern", ["a*", ""], ids=["star", "empty"])
def test_zero_width_pattern_is_rejected(pattern):
    with pytest.raises(ValueError, match="empty string"):
        Redactor(extra_patterns=[pattern])


def test_timestamp_survives():
    text = "at 2026-09-21 10:15:30 ok"
    assert Redactor().apply(text) == text


def test_ipv4_survives():
    text = "host 192.168.10.100 down"
    assert Redactor().apply(text) == text


def test_fewer_than_nine_spaced_digits_survive():
    text = "ids 1 2 3 4 5 6 7 8 done"
    assert Redactor().apply(text) == text


@pytest.mark.parametrize(
    "text",
    ["call +1 4 1 5 5 5 5 2 6 7 1 now", "call 1 4 1 5 5 5 5 2 6 7 1 now"],
    ids=["plus", "bare"],
)
def test_spaced_single_digit_phone_is_redacted(text):
    assert Redactor().apply(text) == "call [phone] now"


@pytest.mark.parametrize(
    "text",
    ["matrix 3 1 4 1 5 9 2 6 5 3 5", "steps 1 2 3 4 5 6 7 8 9 10 done"],
    ids=["digit-row", "numbered-steps"],
)
def test_nine_spaced_digits_are_redacted_even_when_they_are_not_a_phone(text):
    assert "[phone]" in Redactor().apply(text)


def test_long_token_runs_are_fast():
    text = "deadbeef" * 5000
    start = time.perf_counter()
    assert Redactor().apply(text) == text
    assert time.perf_counter() - start < 0.5


def test_phone_next_to_date_is_redacted():
    r = Redactor()
    assert r.apply("at 2026-09-21 555-010-2233 call") == "at [phone] call"
    assert r.apply("de 0221-12-345678 call") == "de [phone] call"


def test_date_range_survives():
    text = "window 2026-09-21 2026-09-28"
    assert Redactor().apply(text) == text


def test_builtin_rules_can_be_disabled():
    text = "bob@example.com +1 (555) 010-2233"
    assert Redactor(builtin=False).apply(text) == text


def test_extra_patterns_apply_without_builtin():
    r = Redactor(extra_patterns=[r"ACC-\d{6}"], builtin=False)
    assert r.apply("bob@example.com ACC-123456") == "bob@example.com [redacted]"


def test_builtin_is_on_by_default():
    assert Redactor().apply("bob@example.com") == "[email]"
