"""Seeded support tickets and the labelled rows the three support sites are imported from."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CALM_CUES",
    "CATEGORIES",
    "CHANNELS",
    "DEADLINE_CUES",
    "HUMAN_CUES",
    "OUTAGE_CUES",
    "PLANS",
    "SITES",
    "STUCK_CUES",
    "URGENCY_LEVELS",
    "Ticket",
    "category",
    "needs_human",
    "state",
    "state_text",
    "tickets",
    "urgency",
]

CATEGORIES = ("billing", "bug", "feature", "account", "other")
"""The classes of the category question, in the order the site is imported with."""

URGENCY_LEVELS = 4
"""Levels of the urgency score, labelled 0 to 3."""

CHANNELS = ("email", "chat", "in_app")
"""Where the ticket arrived from."""

PLANS = ("free", "pro", "enterprise")
"""The plan the customer is on; it is context, and no teacher reads it."""

OUTAGE_CUES = (
    "production outage",
    "everyone in the company",
    "the whole team is down",
    "all of our customers",
)
"""Words that put a ticket at the top urgency level."""

STUCK_CUES = (
    "we cannot carry on",
    "blocking the work",
    "cannot get any further",
    "have to do it by hand",
)
"""Words that say the customer is blocked."""

DEADLINE_CUES = ("before our release", "audit is on friday", "renewal is due")
"""Words that add a level to anything but a calm ticket."""

CALM_CUES = ("no rush", "not urgent", "for future reference")
"""Words that put a ticket at the lowest urgency level."""

HUMAN_CUES = (
    "refund",
    "charged twice",
    "money back",
    "speak to someone",
    "cancel our contract",
    "compensation",
)
"""Words that send a ticket to a person whatever its urgency."""

DEFAULT_ROWS = 3000
DEFAULT_SEED = 1
_FIRST_NUMBER = 1000
_NUMBERS = 400
_DRAWS_PER_TICKET = 200
# Only a ticket about something that broke can carry an outage or a blocked line; a feature
# request that says the whole team is down would be nonsense for the reader and for the teacher.
_IMPACT_SHARE = 0.85
_OUTAGE_SHARE = 0.5
_STUCK_DEADLINE_SHARE = 0.55
_TAIL_SHARE = 0.66

_PRODUCTS = (
    "the dashboard",
    "the mobile app",
    "the API",
    "the export page",
    "the billing portal",
    "the team settings",
    "the audit log",
    "the webhook console",
)
_WHEN_POINT = ("this morning", "on Friday", "after the update last night", "yesterday")
_WHEN_SPAN = ("since Friday", "for the last two days", "all week", "since yesterday")
_WHEN_PAST = ("last month", "two weeks ago", "at the end of March", "on Friday")
_WHEN = {"point": _WHEN_POINT, "span": _WHEN_SPAN, "past": _WHEN_PAST, "": ()}
_AMOUNTS = ("19", "49", "99", "149", "240", "480", "1,200", "2,400")

_OUTAGE = (
    "Everyone in the company is affected and no one can work.",
    "This is a production outage for us.",
    "The whole team is down because of it.",
    "All of our customers are seeing it too.",
)
_STUCK = (
    "We cannot carry on until it is sorted.",
    "It is blocking the work we planned for today.",
    "I cannot get any further on my own.",
    "Until it is fixed we have to do it by hand.",
)
_DEADLINE = (
    "We need an answer before our release tomorrow.",
    "Our audit is on Friday, so the timing matters.",
    "The renewal is due at the end of the week.",
)
_CALM = (
    "There is no rush, I am just curious.",
    "Whenever you have time; this is not urgent.",
    "Asking for future reference only.",
)


@dataclass(frozen=True)
class _Template:
    category: str
    subject: str
    body: str
    # Which time phrases the body reads with: a moment, a stretch, something already over, none.
    when: str = ""
    # Whether something is broken, which is what an outage or a blocked line may be added to.
    incident: bool = False


_TEMPLATES = (
    _Template(
        "billing",
        "Charged twice for invoice {number}",
        "We paid invoice {number} {when} and the same {amount} EUR was taken again today, so we"
        " are asking for a refund of the second charge.",
        when="point",
    ),
    _Template(
        "billing",
        "Invoice {number} has no VAT number on it",
        "Our finance team cannot book invoice {number} for {amount} EUR without our VAT number"
        " printed on it.",
    ),
    _Template(
        "billing",
        "The renewal payment failed",
        "The renewal of {amount} EUR was declined {when} and {product} has been locked since.",
        when="point",
        incident=True,
    ),
    _Template(
        "billing",
        "What is included in the {amount} EUR tier?",
        "Before we move up a tier we would like to know whether {product} is part of it.",
    ),
    _Template(
        "bug",
        "Cannot sign in to {product}",
        "Every sign-in attempt {when} comes back with a session error.",
        when="span",
        incident=True,
    ),
    _Template(
        "bug",
        "Export from {product} fails with a 500",
        "Exporting report {number} has returned a 500 {when}; the smaller reports still work.",
        when="span",
        incident=True,
    ),
    _Template(
        "bug",
        "Numbers on the chart in {product} are wrong",
        "The weekly chart has shown last week's totals {when}, while the CSV underneath is right.",
        when="span",
        incident=True,
    ),
    _Template(
        "bug",
        "Payouts are stuck in pending",
        "Payout {number} for {amount} EUR has been pending {when} and our sellers are waiting for"
        " their money back.",
        when="span",
        incident=True,
    ),
    _Template(
        "feature",
        "SSO with our identity provider",
        "We would like SAML sign-in for {product}; our security review has asked for it.",
    ),
    _Template(
        "feature",
        "CSV export for {product}",
        "A scheduled CSV of report {number} would save our analysts a manual step every week.",
    ),
    _Template(
        "feature",
        "Retries for failed webhooks",
        "When {product} drops an event we replay it by hand; a retry would save us that.",
    ),
    _Template(
        "feature",
        "Raise the seat limit",
        "We cannot invite anyone past seat {number} in {product}, and we have more people to add.",
    ),
    _Template(
        "account",
        "Locked out after losing the MFA device",
        "The phone with our codes is gone and no admin has been able to sign in {when}.",
        when="span",
        incident=True,
    ),
    _Template(
        "account",
        "Transfer ownership of the workspace",
        "Our admin left {when}; please move ownership of {product} to the address in copy.",
        when="past",
    ),
    _Template(
        "account",
        "Remove a teammate who left",
        "Seat {number} belongs to someone who left {when} and should be released.",
        when="past",
    ),
    _Template(
        "account",
        "Close the account",
        "We are winding the project down and want to cancel our contract and get the money back"
        " for the {amount} EUR we paid for months we will not use.",
    ),
    _Template(
        "other",
        "Where do I find the documentation?",
        "I am looking for the guide to {product} and the search on the site returns nothing.",
    ),
    _Template(
        "other",
        "Do you publish a status page?",
        "We would like somewhere to check {product} before we open a ticket like this one.",
    ),
    _Template(
        "other",
        "Partnership enquiry",
        "We build tooling around {product} and would like to speak to someone about partnerships.",
    ),
    _Template(
        "other",
        "Security questionnaire before renewal",
        "Procurement needs the questionnaire back before the {amount} EUR renewal.",
    ),
)


@dataclass(frozen=True)
class Ticket:
    """One drawn ticket: the template it came from, the words filled in and the clauses added."""

    template: _Template
    channel: str
    plan: str
    product: str
    when: str
    amount: str
    number: int
    impact: str
    tail: str


def state(ticket: Ticket) -> dict[str, str]:
    """The flat state a client sends about one ticket."""
    slots = {
        "product": ticket.product,
        "when": ticket.when,
        "amount": ticket.amount,
        "number": ticket.number,
    }
    sentences = [ticket.template.body.format(**slots), ticket.impact, ticket.tail]
    return {
        "channel": ticket.channel,
        "plan": ticket.plan,
        "subject": ticket.template.subject.format(**slots),
        "body": " ".join(sentence for sentence in sentences if sentence),
    }


def state_text(ticket: Ticket) -> str:
    """The text a site sees, written the way stuntd.jev.state.serialize_state writes a state."""
    return json.dumps(state(ticket), ensure_ascii=False)


def category(ticket: Ticket) -> str:
    """The template family the ticket was drawn from."""
    return ticket.template.category


def urgency(ticket: Ticket) -> str:
    """Read off the words of the ticket: an outage is 3, a customer who cannot carry on is 2, a
    ticket that says it is not urgent is 0, anything else is 1; a deadline adds one level to all
    but the calm ones."""
    body = state(ticket)["body"].lower()
    if _says(body, OUTAGE_CUES):
        level = 3
    elif _says(body, STUCK_CUES):
        level = 2
    elif _says(body, CALM_CUES):
        return "0"
    else:
        level = 1
    if _says(body, DEADLINE_CUES):
        level += 1
    return str(min(URGENCY_LEVELS - 1, level))


def needs_human(ticket: Ticket) -> str:
    """A person answers a ticket that asks for money back, for a person or for the contract to
    end, and every ticket at the top urgency level."""
    body = state(ticket)["body"].lower()
    top = urgency(ticket) == str(URGENCY_LEVELS - 1)
    return "true" if top or _says(body, HUMAN_CUES) else "false"


SITES = {"category": category, "urgency": urgency, "needs_human": needs_human}
"""The site each question is imported into and the rule that labels its rows."""


def tickets(seed: int, count: int) -> list[Ticket]:
    """Draws count tickets whose states are all different, the same ones for the same seed."""
    rng = random.Random(seed)
    seen: set[str] = set()
    drawn: list[Ticket] = []
    for _ in range(count * _DRAWS_PER_TICKET):
        if len(drawn) == count:
            return drawn
        ticket = _draw(rng)
        text = state_text(ticket)
        if text not in seen:
            seen.add(text)
            drawn.append(ticket)
    raise RuntimeError(f"only {len(drawn)} different tickets, {count} asked for")


def _says(body: str, cues: tuple[str, ...]) -> bool:
    return any(cue in body for cue in cues)


def _clauses(rng: random.Random, template: _Template) -> tuple[str, str]:
    if template.incident and rng.random() < _IMPACT_SHARE:
        if rng.random() < _OUTAGE_SHARE:
            return rng.choice(_OUTAGE), ""
        # A deadline on top of a blocked customer is what makes a ticket the top level.
        tail = rng.choice(_DEADLINE) if rng.random() < _STUCK_DEADLINE_SHARE else ""
        return rng.choice(_STUCK), tail
    if rng.random() >= _TAIL_SHARE:
        return "", ""
    return "", rng.choice(rng.choice((_DEADLINE, _CALM)))


def _draw(rng: random.Random) -> Ticket:
    template = rng.choice(_TEMPLATES)
    impact, tail = _clauses(rng, template)
    when = _WHEN[template.when]
    return Ticket(
        template=template,
        channel=rng.choice(CHANNELS),
        plan=rng.choice(PLANS),
        product=rng.choice(_PRODUCTS),
        when=rng.choice(when) if when else "",
        amount=rng.choice(_AMOUNTS),
        number=_FIRST_NUMBER + rng.randrange(_NUMBERS),
        impact=impact,
        tail=tail,
    )


def main(argv: list[str] | None = None) -> int:
    """Writes one JSONL file per site into the output directory."""
    args = _parser().parse_args(argv)
    drawn = tickets(args.seed, args.rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for site, teacher in SITES.items():
        path = out / f"{site}.jsonl"
        with path.open("w", encoding="utf-8") as rows:
            for ticket in drawn:
                row = {"text": state_text(ticket), "answer": teacher(ticket)}
                rows.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(drawn)} rows to {path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, metavar="N", help="tickets drawn")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, metavar="S", help="draw seed")
    parser.add_argument("--out", default="rows", metavar="DIR", help="where the JSONL files go")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
