"""Asks a running stuntd the intent and the risk of fresh banking requests, in one request each."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from generate import DEFAULT_SEED, SITES, Request, requests, state, state_text
from typesafe_sdk import Choice, SystemOneResponse, TypeSafeClient

DEFAULT_URL = "http://127.0.0.1:8787"
DEFAULT_KEY = "local"
DEFAULT_SENT = 200
DEFAULT_CLIENT_SEED = DEFAULT_SEED + 10_000
TIMEOUT = 60.0
FRESH_MARGIN = 2
DEFAULT_ROW_DIR = "rows"

INTENT_CRITERIA = {
    "card_arrival": "Where a card that was ordered has got to.",
    "card_lost": "A card was lost or stolen and has to be frozen.",
    "card_declined": "A payment with the card was refused.",
    "atm_fee": "Fees for taking cash out of a machine.",
    "transfer_pending": "A transfer that has not arrived yet.",
    "transfer_failed": "A transfer that came back with an error.",
    "exchange_rate": "The rate used to convert a payment.",
    "top_up_failed": "Money added to the account did not appear.",
    "verify_identity": "Documents and identity checks.",
    "close_account": "Closing the account and taking the balance out.",
    "direct_debit_dispute": "A direct debit the customer says was never agreed.",
    "change_pin": "Setting or resetting the PIN of a card.",
}
RISK_INSTRUCTIONS = (
    "Count the payment's points from the context line: two if it is much larger than anything the"
    " customer has sent before, one if it is only larger than their usual payments, and one each"
    " for a first transfer to this payee, an hour outside their usual ones, and a country they"
    " have never sent money to. Then say what should happen to the payment."
)
RISK_CRITERIA = {
    "allow": "One point or none.",
    "review": "Exactly two points.",
    "block": "Three points or more.",
}
QUESTIONS = {
    "intent": Choice(
        instructions="What is the customer asking about in the message?",
        criteria=INTENT_CRITERIA,
    ),
    "risk": Choice(instructions=RISK_INSTRUCTIONS, criteria=RISK_CRITERIA),
}

_STUNTD_HEADER = "X-Stuntd"


def answers(response: SystemOneResponse) -> dict[str, str]:
    """The answer to each question as the label the teacher would have written."""
    return {name: answer.choice for name, answer in response.choices.items()}


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


def fresh(rows: int, seed: int, trained: set[str]) -> list[Request]:
    """The drawn requests none of those states covers, so the head is asked about states it has
    never seen."""
    unseen = [
        request
        for request in requests(seed, rows * FRESH_MARGIN)
        if state_text(request) not in trained
    ]
    if len(unseen) < rows:
        raise RuntimeError(f"only {len(unseen)} unseen requests at seed {seed}")
    return unseen[:rows]


def ask(client: TypeSafeClient, rows: int, seed: int, trained: set[str]) -> int:
    """Sends one request per drawn row and prints the answers, the agreement and the latency."""
    agreed = dict.fromkeys(SITES, 0)
    latencies: list[float] = []
    live = asked = 0
    for index, request in enumerate(fresh(rows, seed, trained)):
        started = time.perf_counter()
        response = client.system_one(state=state(request), questions=QUESTIONS, timeout=TIMEOUT)
        latencies.append((time.perf_counter() - started) * 1000)
        given = answers(response)
        for site, teacher in SITES.items():
            agreed[site] += given[site] == teacher(request)
        from_head, questions = local_share(response)
        live += from_head
        asked += questions
        if index == 0:
            _print_first(request, given)
    _print_totals(agreed, rows, latencies, live, asked)
    return 0


def _print_first(request: Request, given: dict[str, str]) -> None:
    print("the first request:")
    for key, value in state(request).items():
        print(f"  {key}: {value}")
    for site, answer in given.items():
        print(f"  -> {site}: {answer}")
    print()


def _print_totals(
    agreed: dict[str, int], rows: int, latencies: list[float], live: int, asked: int
) -> None:
    print(f"{rows} requests, p50 {statistics.median(latencies):.1f} ms per request")
    for site, count in agreed.items():
        print(
            f"{site:<7} agrees with the rule teacher on {count}/{rows} ({100 * count / rows:.1f}%)"
        )
    if asked:
        print(f"answered by a trained head: {live}/{asked} questions ({100 * live / asked:.1f}%)")
    else:
        print("no X-Stuntd header came back; run stuntd status --json to see what is serving")


def main(argv: list[str] | None = None) -> int:
    """Sends the fresh requests to the daemon and prints what it answered."""
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
    parser.add_argument("--n", type=int, default=DEFAULT_SENT, metavar="N", help="requests to send")
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
