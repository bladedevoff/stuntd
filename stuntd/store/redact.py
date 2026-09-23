from __future__ import annotations

import re
from collections.abc import Sequence

__all__ = ["Redactor"]

# The lookbehind pins the start to a token boundary, so a long run without an @ is scanned once.
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{6,}\d")
_DATES_ONLY = re.compile(r"\d{4}-\d{2}-\d{2}(?:\s+\d{4}-\d{2}-\d{2})*(?:\s+\d{1,2})?")
_IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
# An ISO date carries eight digits, so nine is the first count no date can reach.
_MIN_PHONE_DIGITS = 9


def _phone_or_original(match: re.Match[str]) -> str:
    candidate = match.group()
    if sum(character.isdigit() for character in candidate) < _MIN_PHONE_DIGITS:
        return candidate
    if _DATES_ONLY.fullmatch(candidate) or _IPV4.match(candidate):
        return candidate
    return "[phone]"


class Redactor:
    """Replaces emails, phone numbers and configured patterns before text is stored."""

    def __init__(self, extra_patterns: Sequence[str] = (), builtin: bool = True) -> None:
        self._builtin = builtin
        self._extra: list[re.Pattern[str]] = []
        for pattern in extra_patterns:
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"bad redaction pattern {pattern!r}: {exc}") from exc
            if compiled.match("") is not None:
                raise ValueError(f"redaction pattern {pattern!r} matches the empty string")
            self._extra.append(compiled)

    def __repr__(self) -> str:
        return f"Redactor(builtin={self._builtin}, extra_patterns={len(self._extra)})"

    def apply(self, text: str) -> str:
        if self._builtin:
            text = _EMAIL.sub("[email]", text)
            text = _PHONE.sub(_phone_or_original, text)
        for pattern in self._extra:
            text = pattern.sub("[redacted]", text)
        return text
