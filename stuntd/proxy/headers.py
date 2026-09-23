from __future__ import annotations

import re
from collections.abc import Iterable

__all__ = [
    "DROPPED_WHEN_BUFFERED",
    "DROPPED_WHEN_STREAMED",
    "NOT_FORWARDED",
    "NOT_FORWARDED_RAW",
    "forwardable",
    "forwardable_raw",
    "jev_header",
    "relayed_headers",
    "stuntd_header",
]

# The hop-by-hop headers of RFC 9110 section 7.6.1 plus host, which names this proxy while
# httpx sets the real one from the upstream base URL.
NOT_FORWARDED = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
    }
)
"""Header names that must not travel between the caller and the provider."""

NOT_FORWARDED_RAW = frozenset(name.encode("ascii") for name in NOT_FORWARDED)
"""NOT_FORWARDED as ascii bytes, for undecoded header pairs."""

# A relayed body is chunked, so the upstream length describes a framing we no longer use; a
# buffered body keeps its length. x-stuntd is ours to set either way.
DROPPED_WHEN_STREAMED = frozenset({b"content-length", b"x-stuntd"})
"""Provider headers a streamed relay must not repeat."""

DROPPED_WHEN_BUFFERED = frozenset({b"x-stuntd"})
"""Provider headers a buffered relay must not repeat."""

# A site name reaches this header from a request header of the caller's, so a value carrying
# CR or LF would let them append a header of their own. fullmatch, not $, which would also
# accept a trailing newline.
_TOKEN = re.compile(r"[A-Za-z0-9_.-]+")


def _token(value: str) -> str:
    if _TOKEN.fullmatch(value) is None:
        raise ValueError(f"unsafe X-Stuntd value {value!r}")
    return value


def forwardable(headers: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drops the headers a proxy must not relay, keeping every other pair and its order."""
    return [(k, v) for k, v in headers if k.lower() not in NOT_FORWARDED]


def forwardable_raw(headers: Iterable[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    """Same filter on undecoded pairs, so a header value that is not latin-1 survives the relay."""
    return [(k, v) for k, v in headers if k.lower() not in NOT_FORWARDED_RAW]


def relayed_headers(
    raw: Iterable[tuple[bytes, bytes]], header: str, dropped: frozenset[bytes]
) -> list[tuple[bytes, bytes]]:
    """The provider's own headers as the caller gets them, with this daemon's X-Stuntd appended."""
    pairs = [(name, value) for name, value in forwardable_raw(raw) if name.lower() not in dropped]
    pairs.append((b"x-stuntd", header.encode("ascii")))
    return pairs


def stuntd_header(
    mode: str,
    *,
    site: str | None = None,
    reason: str | None = None,
    confidence: float | None = None,
) -> str:
    """Formats an X-Stuntd value: the mode, then the fields it was given as key=value."""
    parts = [mode]
    if site is not None:
        parts.append(f"site={_token(site)}")
    if reason is not None:
        parts.append(f"reason={_token(reason)}")
    if confidence is not None:
        parts.append(f"confidence={confidence:.2f}")
    return "; ".join(parts)


def jev_header(
    mode: str,
    *,
    questions: int | None = None,
    live: int | None = None,
    shadow: int | None = None,
    zeroshot: int | None = None,
    check: int | None = None,
    learn_off: bool = False,
    reason: str | None = None,
) -> str:
    """Formats the X-Stuntd value of a Jev answer: the mode it was served in and its counts."""
    parts = ["jev", f"mode={_token(mode)}"]
    for name, count in (
        ("questions", questions),
        ("live", live),
        ("shadow", shadow),
        ("zeroshot", zeroshot),
        ("check", check),
    ):
        if count is not None:
            parts.append(f"{name}={count:d}")
    if learn_off:
        parts.append("learn=off")
    if reason is not None:
        parts.append(f"reason={_token(reason)}")
    return "; ".join(parts)
