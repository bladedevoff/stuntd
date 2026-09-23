import pytest
from demo_helpers import (
    TypeSafeClient,
    choice_answer,
    load_demo,
    noul_answer,
    recording_transport,
    score_answer,
)

from stuntd.jev.state import serialize_state

support, client = load_demo("support")

ANSWERS = {
    "category": choice_answer("bug", support.CATEGORIES),
    "urgency": score_answer(2, support.URGENCY_LEVELS),
    "needs_human": noul_answer(0.8),
}


def _ticket(body, category="bug"):
    template = support._Template(category, "Subject {number}", body)
    return support.Ticket(
        template=template,
        channel="email",
        plan="free",
        product="the API",
        when="this morning",
        amount="49",
        number=1001,
        impact="",
        tail="",
    )


def test_the_same_seed_draws_the_same_tickets():
    first = [support.state_text(ticket) for ticket in support.tickets(1, 300)]
    assert first == [support.state_text(ticket) for ticket in support.tickets(1, 300)]
    assert first != [support.state_text(ticket) for ticket in support.tickets(2, 300)]
    assert len(set(first)) == 300


def test_every_teacher_label_is_a_label_of_its_question():
    drawn = support.tickets(1, 400)
    written = {site: {teacher(t) for t in drawn} for site, teacher in support.SITES.items()}
    assert written["category"] == set(support.CATEGORIES)
    assert set(client.QUESTIONS["category"].criteria) == set(support.CATEGORIES)
    assert written["urgency"] == {str(level) for level in range(support.URGENCY_LEVELS)}
    assert len(client.QUESTIONS["urgency"].criteria) == support.URGENCY_LEVELS
    assert written["needs_human"] == {"true", "false"}


@pytest.mark.parametrize(
    "body, urgency, needs_human",
    [
        ("The export is slow.", "1", "false"),
        ("The export is slow. There is no rush, I am just curious.", "0", "false"),
        ("The export is slow. We need an answer before our release tomorrow.", "2", "false"),
        ("The export is slow. We cannot carry on until it is sorted.", "2", "false"),
        (
            "The export is slow. It is blocking the work we planned for today. Our audit is on"
            " Friday, so the timing matters.",
            "3",
            "true",
        ),
        ("The export is slow. This is a production outage for us.", "3", "true"),
        ("We were charged twice and would like a refund.", "1", "true"),
        ("Please have someone speak to someone in billing.", "1", "true"),
    ],
    ids=[
        "plain",
        "calm",
        "deadline",
        "stuck",
        "stuck-deadline",
        "outage",
        "refund",
        "asks-for-a-person",
    ],
)
def test_the_rule_teacher_reads_the_words_of_the_ticket(body, urgency, needs_human):
    ticket = _ticket(body)
    assert support.urgency(ticket) == urgency
    assert support.needs_human(ticket) == needs_human


def test_the_client_skips_the_states_it_was_trained_on(tmp_path):
    assert support.main(["--rows", "40", "--seed", "1", "--out", str(tmp_path)]) == 0
    trained = client.trained_texts(tmp_path)
    assert len(trained) == 40
    drawn = client.fresh(5, 2, trained)
    assert {support.state_text(ticket) for ticket in drawn}.isdisjoint(trained)
    with pytest.raises(RuntimeError, match="unseen tickets at seed 1"):
        client.fresh(5, 1, trained)


def test_the_client_sends_the_imported_text_and_the_three_questions():
    sent, transport = recording_transport(ANSWERS)
    with TypeSafeClient(
        api_key="local", base_url="http://stuntd.invalid", transport=transport
    ) as c:
        assert client.ask(c, 3, 7, set()) == 0
    assert [serialize_state(body["state"]) for body in sent] == [
        support.state_text(ticket) for ticket in client.fresh(3, 7, set())
    ]
    asked = sent[0]["questions"]
    assert set(asked) == set(support.SITES)
    assert asked["category"]["type"] == "choice"
    assert list(asked["category"]["criteria"]) == list(support.CATEGORIES)
    assert asked["urgency"]["type"] == "score"
    assert asked["needs_human"]["type"] == "noul"
