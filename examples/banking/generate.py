"""Seeded banking requests and the labelled rows the intent and risk sites are imported from."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BLOCK_POINTS",
    "FAMILIAR",
    "INTENTS",
    "LARGER",
    "MUCH_LARGER",
    "NEW_COUNTRY",
    "NEW_PAYEE",
    "ODD_HOURS",
    "REVIEW_POINTS",
    "RISKS",
    "SITES",
    "UNFAMILIAR",
    "Request",
    "intent",
    "requests",
    "risk",
    "state",
    "state_text",
]

INTENTS = (
    "card_arrival",
    "card_lost",
    "card_declined",
    "atm_fee",
    "transfer_pending",
    "transfer_failed",
    "exchange_rate",
    "top_up_failed",
    "verify_identity",
    "close_account",
    "direct_debit_dispute",
    "change_pin",
)
"""The classes of the intent question, in the order the site is imported with."""

RISKS = ("allow", "review", "block")
"""The classes of the risk question."""

MUCH_LARGER = "much larger than anything they have sent before"
"""Two risk points."""

LARGER = "larger than their usual payments"
"""One risk point."""

NEW_PAYEE = "first transfer to this payee"
"""One risk point."""

ODD_HOURS = "outside their usual hours"
"""One risk point."""

NEW_COUNTRY = "to a country they have never sent money to"
"""One risk point."""

REVIEW_POINTS = 2
"""Points from which a payment is held for review."""

BLOCK_POINTS = 3
"""Points from which a payment is blocked."""

FAMILIAR = ("DE", "AT", "NL", "FR", "ES", "IT")
"""Countries the customer sends money to regularly."""

UNFAMILIAR = ("US", "GB", "SG", "BR", "JP", "ZA")
"""Countries the context calls new for this customer."""

DEFAULT_ROWS = 3000
DEFAULT_SEED = 1
_DRAWS_PER_REQUEST = 200
_USUAL_AMOUNT = "about the size they usually send"
_KNOWN_PAYEE = "a payee they pay most months"
_MUCH_LARGER_SHARE = 0.2
_LARGER_SHARE = 0.55
_NEW_PAYEE_SHARE = 0.4
_ODD_HOURS_SHARE = 0.3
_NEW_COUNTRY_SHARE = 0.3
_PAYEES = (
    "Ada Nwosu",
    "Marek Jelinek",
    "Sofia Ferrari",
    "Hiroshi Tanaka",
    "Nora Lindqvist",
    "Tomas Pereira",
    "Amina Haddad",
    "Ivan Petrov",
    "Claire Dubois",
    "Sam Okafor",
)
_BANDS = {_USUAL_AMOUNT: (20.0, 400.0), LARGER: (600.0, 3000.0), MUCH_LARGER: (5000.0, 40000.0)}
_DAY_HOURS = (9, 19)
_NIGHT_HOURS = (0, 4)
_CARDS = ("the blue card", "the metal card", "the virtual card", "my second card")
_SHOPS = ("a petrol station", "a hotel in Lisbon", "an online bookshop", "a supermarket")
_DAYS = ("on Monday", "last Thursday", "three days ago", "yesterday", "on the 2nd")


@dataclass(frozen=True)
class _Template:
    intent: str
    message: str


_TEMPLATES = (
    _Template("card_arrival", "I ordered {card} {day} and it still has not arrived."),
    _Template("card_arrival", "How long does a new card take to reach me?"),
    _Template("card_lost", "I left {card} in a taxi {day}, please freeze it."),
    _Template("card_lost", "My wallet was stolen at {shop} and the card is gone."),
    _Template("card_declined", "{card} was declined at {shop} {day} and I had money."),
    _Template("card_declined", "Why does my card keep getting declined at {shop}?"),
    _Template("atm_fee", "I was charged a fee for taking cash out of an ATM {day}."),
    _Template("atm_fee", "Does taking cash out abroad cost me anything?"),
    _Template("transfer_pending", "The transfer I sent {day} still says pending."),
    _Template("transfer_pending", "How long does a payment abroad stay pending?"),
    _Template("transfer_failed", "My transfer {day} came back with an error and no reason."),
    _Template("transfer_failed", "The payment to {payee} failed twice and the money is gone."),
    _Template("exchange_rate", "What rate did you use for my purchase at {shop} {day}?"),
    _Template("exchange_rate", "Your rate looks worse than the one I see online."),
    _Template("top_up_failed", "I tried to top up {day} and the money never appeared."),
    _Template("top_up_failed", "Topping up from my other bank fails every time."),
    _Template("verify_identity", "You asked me for documents, which ones do you accept?"),
    _Template("verify_identity", "My identity check has been in review since {day}."),
    _Template("close_account", "Please close my account, I have moved away."),
    _Template("close_account", "How do I close the account and take the balance out?"),
    _Template("direct_debit_dispute", "A direct debit to {payee} {day} was never agreed."),
    _Template("direct_debit_dispute", "I cancelled a subscription and was billed anyway {day}."),
    _Template("change_pin", "I want to change the PIN on {card}."),
    _Template("change_pin", "I forgot the PIN of {card}, how do I set a new one?"),
)


@dataclass(frozen=True)
class Request:
    """One drawn request: what the customer wrote, the payment in front of them, and how that
    payment compares with the ones they usually make."""

    template: _Template
    payee: str
    amount: float
    amount_phrase: str
    country: str
    new_payee: bool
    odd_hours: bool
    new_country: bool
    hour: int
    minute: int
    card: str
    shop: str
    day: str


def context(request: Request) -> str:
    """How this payment compares with the customer's own history, in words."""
    parts = [request.amount_phrase, NEW_PAYEE if request.new_payee else _KNOWN_PAYEE]
    if request.odd_hours:
        parts.append(ODD_HOURS)
    if request.new_country:
        parts.append(NEW_COUNTRY)
    return "; ".join(parts)


def state(request: Request) -> dict[str, str]:
    """The flat state a client sends about one request."""
    return {
        "message": request.template.message.format(
            card=request.card, shop=request.shop, day=request.day, payee=request.payee
        ),
        "context": context(request),
        "amount": f"{request.amount:,.2f} EUR",
        "payee": request.payee,
        "country": request.country,
        "time": f"{request.hour:02d}:{request.minute:02d}",
    }


def state_text(request: Request) -> str:
    """The text a site sees, written the way stuntd.jev.state.serialize_state writes a state."""
    return json.dumps(state(request), ensure_ascii=False)


def intent(request: Request) -> str:
    """The template family the message was drawn from."""
    return request.template.intent


def risk(request: Request) -> str:
    """Read off the context line: two points if the payment is much larger than anything the
    customer has sent, one if it is merely larger, and one each for a first transfer to the payee,
    an unusual hour and a country they have never sent money to. Three points block it, two hold
    it for review, one or none let it through."""
    written = context(request)
    points = 2 if MUCH_LARGER in written else int(LARGER in written)
    points += NEW_PAYEE in written
    points += ODD_HOURS in written
    points += NEW_COUNTRY in written
    if points >= BLOCK_POINTS:
        return "block"
    return "review" if points >= REVIEW_POINTS else "allow"


SITES = {"intent": intent, "risk": risk}
"""The site each question is imported into and the rule that labels its rows."""


def requests(seed: int, count: int) -> list[Request]:
    """Draws count requests whose states are all different, the same ones for the same seed."""
    rng = random.Random(seed)
    seen: set[str] = set()
    drawn: list[Request] = []
    for _ in range(count * _DRAWS_PER_REQUEST):
        if len(drawn) == count:
            return drawn
        request = _draw(rng)
        text = state_text(request)
        if text not in seen:
            seen.add(text)
            drawn.append(request)
    raise RuntimeError(f"only {len(drawn)} different requests, {count} asked for")


def _draw(rng: random.Random) -> Request:
    roll = rng.random()
    if roll < _MUCH_LARGER_SHARE:
        phrase = MUCH_LARGER
    elif roll < _LARGER_SHARE:
        phrase = LARGER
    else:
        phrase = _USUAL_AMOUNT
    low, high = _BANDS[phrase]
    odd_hours = rng.random() < _ODD_HOURS_SHARE
    new_country = rng.random() < _NEW_COUNTRY_SHARE
    return Request(
        template=rng.choice(_TEMPLATES),
        payee=rng.choice(_PAYEES),
        amount=round(rng.uniform(low, high), 2),
        amount_phrase=phrase,
        country=rng.choice(UNFAMILIAR if new_country else FAMILIAR),
        new_payee=rng.random() < _NEW_PAYEE_SHARE,
        odd_hours=odd_hours,
        new_country=new_country,
        hour=rng.randint(*(_NIGHT_HOURS if odd_hours else _DAY_HOURS)),
        minute=rng.randrange(60),
        card=rng.choice(_CARDS),
        shop=rng.choice(_SHOPS),
        day=rng.choice(_DAYS),
    )


def main(argv: list[str] | None = None) -> int:
    """Writes one JSONL file per site into the output directory."""
    args = _parser().parse_args(argv)
    drawn = requests(args.seed, args.rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for site, teacher in SITES.items():
        path = out / f"{site}.jsonl"
        with path.open("w", encoding="utf-8") as rows:
            for request in drawn:
                row = {"text": state_text(request), "answer": teacher(request)}
                rows.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(drawn)} rows to {path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, metavar="N", help="rows drawn")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, metavar="S", help="draw seed")
    parser.add_argument("--out", default="rows", metavar="DIR", help="where the JSONL files go")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
