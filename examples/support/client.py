"""Asks a running stuntd the three support questions about fresh tickets, in one request each."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

from generate import DEFAULT_SEED, SITES, Ticket, state, state_text, tickets
from typesafe_sdk import Choice, Noul, Score, SystemOneResponse, TypeSafeClient

DEFAULT_URL = "http://127.0.0.1:8787"
DEFAULT_KEY = "local"
DEFAULT_SENT = 200
DEFAULT_CLIENT_SEED = DEFAULT_SEED + 10_000
TIMEOUT = 60.0
FRESH_MARGIN = 2
DEFAULT_ROW_DIR = "rows"
NOUL_TRUE = 0.5

CATEGORY_CRITERIA = {
    "billing": "An invoice, a charge, a refund or the price of a plan.",
    "bug": "The product does not do what it promises.",
    "feature": "Something the product cannot do yet.",
    "account": "Access, ownership, seats or closing the account.",
    "other": "Anything else, including questions that are not about the product.",
}
URGENCY_CRITERIA = [
    "The ticket says itself that it can wait: no rush, just curious, for future reference.",
    "An ordinary request with nothing in it that makes it urgent.",
    "The customer says they are stuck, or an ordinary request carries a deadline.",
    "An outage, or a customer who is stuck and has a deadline as well.",
]
QUESTIONS = {
    "category": Choice(
        instructions="Which part of the product is this ticket about?",
        criteria=CATEGORY_CRITERIA,
    ),
    "urgency": Score(
        instructions="How soon does this ticket have to be answered?",
        criteria=URGENCY_CRITERIA,
    ),
    "needs_human": Noul(
        instructions="Does this ticket have to be read by a support agent rather than a bot?",
        criteria={
            "true": "It asks for money back, for a person or to end the contract, or it is at"
            " urgency 3: an outage, or a customer who is stuck and has a deadline.",
            "false": "A canned answer or an automation can close it.",
        },
    ),
}

_STUNTD_HEADER = "X-Stuntd"


def answers(response: SystemOneResponse) -> dict[str, str]:
    """The answer to each question as the label the teacher would have written."""
    given = {name: answer.choice for name, answer in response.choices.items()}
    given.update(
        {name: str(math.floor(answer.score + 0.5)) for name, answer in response.scores.items()}
    )
    given.update(
        {
            name: "true" if answer.noul >= NOUL_TRUE else "false"
            for name, answer in response.nouls.items()
        }
    )
    return given


def local_share(response: SystemOneResponse) -> tuple[int, int]:
    """Questions this answer served from a trained head, out of the questions asked."""
    header = response.raw_http_response.headers.get(_STUNTD_HEADER, "")
    counts = dict(_count(part) for part in header.split(";") if "=" in part)
    return counts.get("live", 0), counts.get("questions", 0)


def _count(part: str) -> tuple[str, int]:
    name, _, value = part.strip().partition("=")
    return name, int(value) if value.isdigit() else 0


def trained_texts(folder: Path) -> set[str]:
    """The states the sites were trained on, read back from the rows the generator wrote."""
    path = folder / f"{next(iter(SITES))}.jsonl"
    with path.open(encoding="utf-8") as rows:
        return {json.loads(line)["text"] for line in rows if line.strip()}


def fresh(rows: int, seed: int, trained: set[str]) -> list[Ticket]:
    """The drawn tickets none of those states covers, so the head is asked about states it has
    never seen."""
    unseen = [
        ticket for ticket in tickets(seed, rows * FRESH_MARGIN) if state_text(ticket) not in trained
    ]
    if len(unseen) < rows:
        raise RuntimeError(f"only {len(unseen)} unseen tickets at seed {seed}")
    return unseen[:rows]


def ask(client: TypeSafeClient, rows: int, seed: int, trained: set[str]) -> int:
    """Sends one request per drawn row and prints the answers, the agreement and the latency."""
    agreed = dict.fromkeys(SITES, 0)
    latencies: list[float] = []
    live = asked = 0
    for index, ticket in enumerate(fresh(rows, seed, trained)):
        started = time.perf_counter()
        response = client.system_one(state=state(ticket), questions=QUESTIONS, timeout=TIMEOUT)
        latencies.append((time.perf_counter() - started) * 1000)
        given = answers(response)
        for site, teacher in SITES.items():
            agreed[site] += given[site] == teacher(ticket)
        from_head, questions = local_share(response)
        live += from_head
        asked += questions
        if index == 0:
            _print_first(ticket, given)
    _print_totals(agreed, rows, latencies, live, asked)
    return 0


def _print_first(ticket: Ticket, given: dict[str, str]) -> None:
    print("the first ticket:")
    for key, value in state(ticket).items():
        print(f"  {key}: {value}")
    for site, answer in given.items():
        print(f"  -> {site}: {answer}")
    print()


def _print_totals(
    agreed: dict[str, int], rows: int, latencies: list[float], live: int, asked: int
) -> None:
    print(f"{rows} tickets, p50 {statistics.median(latencies):.1f} ms per request")
    for site, count in agreed.items():
        print(
            f"{site:<12} agrees with the rule teacher on {count}/{rows} ({100 * count / rows:.1f}%)"
        )
    if asked:
        print(f"answered by a trained head: {live}/{asked} questions ({100 * live / asked:.1f}%)")
    else:
        print("no X-Stuntd header came back; run stuntd status --json to see what is serving")


def main(argv: list[str] | None = None) -> int:
    """Sends the fresh tickets to the daemon and prints what it answered."""
    args = _parser().parse_args(argv)
    if args.n < 1:
        print("--n must be at least 1", file=sys.stderr)
        return 2
    try:
        trained = trained_texts(Path(args.trained))
    except OSError as exc:
        print(f"stuntd demo: {exc}; run generate.py --out {args.trained} first", file=sys.stderr)
        return 2
    with TypeSafeClient(api_key=args.api_key, base_url=args.url, timeout=TIMEOUT) as client:
        return ask(client, args.n, args.seed, trained)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, metavar="URL", help="the stuntd to ask")
    parser.add_argument(
        "--api-key",
        default=DEFAULT_KEY,
        metavar="K",
        help="key sent with every request; a local daemon does not check it",
    )
    parser.add_argument("--n", type=int, default=DEFAULT_SENT, metavar="N", help="tickets to send")
    parser.add_argument(
        "--trained",
        default=DEFAULT_ROW_DIR,
        metavar="DIR",
        help="where generate.py wrote its rows; their states are skipped as already trained on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CLIENT_SEED,
        metavar="S",
        help="draw seed; the default is not the seed the training rows were drawn with",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
