from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from stuntd.modes import MODE_CHECK, MODE_LIVE, MODE_SHADOW
from stuntd.paths import private_file
from stuntd.store.redact import Redactor

__all__ = ["PRUNE_EVERY", "Capture", "Decision", "Example", "SiteInfo", "Store"]

_SCHEMA = """
create table if not exists captures (
    id integer primary key,
    site text not null,
    schema_canonical text not null,
    kind text not null,
    input_text text not null,
    answer text not null,
    model text not null,
    latency_ms integer not null,
    prompt_tokens integer,
    completion_tokens integer,
    created_at real not null
);
create index if not exists captures_site_time on captures(site, created_at);
create index if not exists captures_created on captures(created_at);
create table if not exists decisions (
    id integer primary key,
    site text not null,
    mode text not null,
    answer text not null,
    confidence real not null,
    agree integer,
    latency_ms integer not null,
    created_at real not null
);
create index if not exists decisions_site_time on decisions(site, created_at);
"""

_INSERT = (
    "insert into captures (site, schema_canonical, kind, input_text, answer, model,"
    " latency_ms, prompt_tokens, completion_tokens, created_at) values (?,?,?,?,?,?,?,?,?,?)"
)

_INSERT_DECISION = (
    "insert into decisions (site, mode, answer, confidence, agree, latency_ms, created_at)"
    " values (?,?,?,?,?,?,?)"
)

_DECISIONS = (
    "select site, mode, answer, confidence, agree, latency_ms, created_at from decisions"
    " where site = ? order by created_at desc, id desc limit ?"
)

_COMPARISONS = (
    "select site, mode, answer, confidence, agree, latency_ms, created_at from decisions"
    " where site = ? and mode in (?, ?) and created_at >= ?"
    " order by created_at desc, id desc limit ?"
)

_DECISION_COUNTS = "select site, mode, count(*) from decisions group by site, mode"

_EXAMPLES = (
    "select input_text, answer, created_at from captures where site = ? order by created_at, id"
)

_SITES = (
    "select latest.site, captures.kind, captures.schema_canonical, latest.total from"
    " (select site, count(*) as total, max(id) as last_id from captures group by site) as latest"
    " join captures on captures.id = latest.last_id order by latest.site"
)

_MODES = (MODE_SHADOW, MODE_CHECK, MODE_LIVE)

_SECONDS_PER_DAY = 86400

PRUNE_EVERY = 100
"""Records between prunes, so no single capture pays for a table scan."""


@dataclass
class Capture:
    """One decision to record, before redaction."""

    site: str
    schema_canonical: str
    kind: str
    input_text: str
    answer: str
    model: str
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(frozen=True)
class Decision:
    """One answer the trained head gave, in shadow, check or live mode; created_at is stamped
    on write, so it carries a time only on rows read back."""

    site: str
    mode: str
    answer: str
    confidence: float
    agree: bool | None
    latency_ms: int
    created_at: float


@dataclass(frozen=True)
class Example:
    """One recorded decision, as training reads it back."""

    input_text: str
    answer: str
    created_at: float


@dataclass(frozen=True)
class SiteInfo:
    """What one decision site looks like, described by its latest capture."""

    site: str
    kind: str
    schema_canonical: str
    count: int


class Store:
    """Owns the capture database: redacts and prunes what it writes, reads it back for training."""

    def __init__(
        self,
        path: Path,
        redactor: Redactor,
        max_rows: int = 100_000,
        max_age_days: int = 30,
    ) -> None:
        private_file(path)
        self._path = path
        self._conn = sqlite3.connect(path)
        try:
            self._conn.executescript(_SCHEMA)
        except sqlite3.DatabaseError:
            # A file that is not a database leaves the caller with no Store to close.
            self._conn.close()
            raise
        self._redactor = redactor
        self._max_rows = max_rows
        self._max_age = max_age_days * _SECONDS_PER_DAY
        # Starts full so the first record after opening prunes whatever an earlier run left behind.
        self._since_prune = PRUNE_EVERY

    def __repr__(self) -> str:
        return f"Store(path={str(self._path)!r}, max_rows={self._max_rows})"

    def redact(self, text: str) -> str:
        """The text as a capture of it would be stored, so a head is asked what it was trained on."""
        return self._redactor.apply(text)

    def record(self, capture: Capture) -> None:
        self._conn.execute(
            _INSERT,
            (
                capture.site,
                capture.schema_canonical,
                capture.kind,
                self._redactor.apply(capture.input_text),
                capture.answer,
                capture.model,
                capture.latency_ms,
                capture.prompt_tokens,
                capture.completion_tokens,
                time.time(),
            ),
        )
        self._count_and_commit()

    def record_decision(self, decision: Decision) -> None:
        self._conn.execute(
            _INSERT_DECISION,
            (
                decision.site,
                decision.mode,
                decision.answer,
                decision.confidence,
                None if decision.agree is None else int(decision.agree),
                decision.latency_ms,
                time.time(),
            ),
        )
        self._count_and_commit()

    def _count_and_commit(self) -> None:
        # One counter for both tables, so writing to either does not double the prune rate.
        self._since_prune += 1
        if self._since_prune >= PRUNE_EVERY:
            self._prune()
            self._since_prune = 0
        self._conn.commit()

    def _prune(self) -> None:
        self._conn.execute(
            "delete from captures where created_at < ?", (time.time() - self._max_age,)
        )
        # The offset subquery reads one id instead of materialising max_rows of them, and yields
        # NULL - deleting nothing - while the table is still smaller than the cap.
        self._conn.execute(
            "delete from captures where id <= "
            "(select id from captures order by id desc limit 1 offset ?)",
            (self._max_rows,),
        )
        self._conn.execute(
            "delete from decisions where created_at < ?", (time.time() - self._max_age,)
        )
        self._conn.execute(
            "delete from decisions where id <= "
            "(select id from decisions order by id desc limit 1 offset ?)",
            (self._max_rows,),
        )

    def examples(self, site: str) -> list[Example]:
        rows = self._conn.execute(_EXAMPLES, (site,)).fetchall()
        return [Example(text, answer, created_at) for text, answer, created_at in rows]

    def decisions(self, site: str, limit: int) -> list[Decision]:
        _check_limit(limit)
        return self._decisions(_DECISIONS, (site, limit))

    def comparisons(self, site: str, limit: int, since: float) -> list[Decision]:
        # A live answer has nothing to agree or disagree with, so a site judged on its recent
        # comparisons would otherwise read a window of rows that cannot be compared.
        _check_limit(limit)
        return self._decisions(_COMPARISONS, (site, MODE_SHADOW, MODE_CHECK, since, limit))

    def _decisions(self, sql: str, params: tuple[object, ...]) -> list[Decision]:
        return [
            Decision(
                row_site,
                mode,
                answer,
                confidence,
                None if agree is None else bool(agree),
                latency_ms,
                created_at,
            )
            for row_site, mode, answer, confidence, agree, latency_ms, created_at in (
                self._conn.execute(sql, params).fetchall()
            )
        ]

    def decision_counts(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for site, mode, total in self._conn.execute(_DECISION_COUNTS):
            counts.setdefault(site, dict.fromkeys(_MODES, 0))[mode] = total
        return counts

    def sites(self) -> list[SiteInfo]:
        rows = self._conn.execute(_SITES).fetchall()
        return [SiteInfo(site, kind, schema, count) for site, kind, schema, count in rows]

    def stats(self) -> list[dict[str, str | int | float]]:
        rows = self._conn.execute(
            "select site, kind, count(*), min(created_at), max(created_at)"
            " from captures group by site, kind"
        ).fetchall()
        return [
            {
                "site": site,
                "kind": kind,
                "count": count,
                "first_at": first,
                "last_at": last,
            }
            for site, kind, count, first, last in rows
        ]

    def close(self) -> None:
        self._conn.close()


def _check_limit(limit: int) -> None:
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit!r}")
