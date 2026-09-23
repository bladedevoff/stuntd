import pytest
from demo_helpers import TypeSafeClient, choice_answer, load_demo, recording_transport

from stuntd.jev.state import serialize_state

banking, client = load_demo("banking")

ANSWERS = {
    "intent": choice_answer("card_lost", banking.INTENTS),
    "risk": choice_answer("review", banking.RISKS),
}


def _request(amount_phrase, new_payee=False, odd_hours=False, new_country=False):
    template = banking._Template("card_lost", "I lost {card}, please freeze it.")
    return banking.Request(
        template=template,
        payee="Ada Nwosu",
        amount=420.0,
        amount_phrase=amount_phrase,
        country="US" if new_country else "DE",
        new_payee=new_payee,
        odd_hours=odd_hours,
        new_country=new_country,
        hour=2 if odd_hours else 14,
        minute=30,
        card="the blue card",
        shop="a hotel in Lisbon",
        day="yesterday",
    )


USUAL = "about the size they usually send"


def test_the_same_seed_draws_the_same_requests():
    first = [banking.state_text(request) for request in banking.requests(1, 300)]
    assert first == [banking.state_text(request) for request in banking.requests(1, 300)]
    assert first != [banking.state_text(request) for request in banking.requests(2, 300)]
    assert len(set(first)) == 300


def test_every_teacher_label_is_a_label_of_its_question():
    drawn = banking.requests(1, 400)
    written = {site: {teacher(r) for r in drawn} for site, teacher in banking.SITES.items()}
    assert written["intent"] == set(banking.INTENTS)
    assert written["risk"] == set(banking.RISKS)
    assert list(client.QUESTIONS["intent"].criteria) == list(banking.INTENTS)
    assert list(client.QUESTIONS["risk"].criteria) == list(banking.RISKS)


@pytest.mark.parametrize(
    "request_, risk",
    [
        (_request(USUAL), "allow"),
        (_request(USUAL, new_payee=True), "allow"),
        (_request(banking.LARGER), "allow"),
        (_request(banking.LARGER, new_payee=True), "review"),
        (_request(banking.MUCH_LARGER), "review"),
        (_request(USUAL, new_payee=True, odd_hours=True, new_country=True), "block"),
        (_request(banking.MUCH_LARGER, new_payee=True), "block"),
    ],
    ids=[
        "routine",
        "new-payee",
        "larger",
        "larger-new-payee",
        "much-larger",
        "three-signs",
        "much-larger-new-payee",
    ],
)
def test_the_rule_teacher_reads_the_context_line(request_, risk):
    assert banking.risk(request_) == risk


def test_the_client_sends_the_imported_text_and_both_questions():
    sent, transport = recording_transport(ANSWERS)
    with TypeSafeClient(
        api_key="local", base_url="http://stuntd.invalid", transport=transport
    ) as c:
        assert client.ask(c, 3, 7, set()) == 0
    assert [serialize_state(body["state"]) for body in sent] == [
        banking.state_text(request) for request in client.fresh(3, 7, set())
    ]
    asked = sent[0]["questions"]
    assert set(asked) == set(banking.SITES)
    assert asked["intent"]["type"] == "choice"
    assert list(asked["risk"]["criteria"]) == list(banking.RISKS)
