from dataclasses import replace

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
_, check = load_demo("support", "check")

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


def test_a_stuck_payout_without_an_outage_does_not_need_a_human():
    payouts = [
        ticket
        for ticket in support.tickets(1, 3000)
        if ticket.template.subject == "Payouts are stuck in pending"
        and support.urgency(ticket) != "3"
    ]
    assert payouts
    assert {support.needs_human(ticket) for ticket in payouts} == {"false"}


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


def _readings(answers, confident=True, novelty=0.1):
    return {
        site: [check.Reading(answer[site], confident, novelty) for answer in answers]
        for site in support.SITES
    }


def _answers(ticket):
    return {site: teacher(ticket) for site, teacher in support.SITES.items()}


def test_tickets_drawn_from_some_templates_use_only_those():
    chosen = support.TEMPLATES[:3]
    assert {ticket.template for ticket in support.tickets(1, 200, chosen)} == set(chosen)


def test_the_left_out_templates_are_the_first_of_each_category():
    left_out = check.held_out()
    assert [template.category for template in left_out] == list(support.CATEGORIES)
    for template in left_out:
        first = next(one for one in support.TEMPLATES if one.category == template.category)
        assert template is first


def test_unseen_tickets_skip_the_trained_states():
    trained = {support.state_text(ticket) for ticket in support.tickets(1, 60)}
    drawn = check.unseen(trained, support.TEMPLATES, 20)
    assert len(drawn) == 20
    assert {support.state_text(ticket) for ticket in drawn}.isdisjoint(trained)
    assert drawn == check.unseen(trained, support.TEMPLATES, 20)


def test_unseen_tickets_come_from_the_templates_asked_for():
    drawn = check.unseen(set(), check.held_out(), 50)
    assert {ticket.template for ticket in drawn} <= set(check.held_out())


def test_gates_cut_each_site_at_the_quantiles_of_its_holdout_novelty():
    novelties = {site: [index / 100 for index in range(101)] for site in support.SITES}
    gates = check.gates(novelties)
    assert list(gates) == ["no gate", "0.95 quantile", "0.98 quantile", "0.99 quantile"]
    assert gates["no gate"] == dict.fromkeys(support.SITES)
    assert gates["0.95 quantile"]["urgency"] == pytest.approx(0.95)
    assert gates["0.99 quantile"]["category"] == pytest.approx(0.99)


def test_a_ticket_is_local_only_when_all_three_heads_are_sure_and_familiar():
    drawn = support.tickets(1, 4)
    answers = [_answers(ticket) for ticket in drawn]
    readings = _readings(answers)
    readings["urgency"][1] = check.Reading(answers[1]["urgency"], False, 0.1)
    readings["category"][2] = check.Reading(answers[2]["category"], True, 0.5)
    readings["needs_human"][3] = check.Reading("wrong", True, 0.1)
    ungated = check.tally(drawn, readings, dict.fromkeys(support.SITES))
    assert (ungated.total, ungated.local, ungated.right) == (4, 3, 2)
    gated = check.tally(drawn, readings, dict.fromkeys(support.SITES, 0.2))
    assert (gated.total, gated.local, gated.right) == (4, 2, 1)


def test_the_probes_are_stopped_when_no_head_answers_them():
    answers = [{"category": "bug", "urgency": "1", "needs_human": "false"}] * 2
    readings = _readings(answers, confident=False)
    readings["urgency"][1] = check.Reading("1", True, 0.9)
    ungated = dict.fromkeys(support.SITES)
    assert check.stopped(readings, 0, ungated)
    assert not check.stopped(readings, 1, ungated)
    assert check.stopped(readings, 1, dict.fromkeys(support.SITES, 0.5))


def test_swapping_the_context_changes_the_channel_and_the_plan_only():
    ticket = support.tickets(1, 1)[0]
    twin = check.swapped(ticket)
    assert twin.channel != ticket.channel
    assert twin.plan != ticket.plan
    assert replace(twin, channel=ticket.channel, plan=ticket.plan) == ticket


def test_changes_count_the_answers_that_differ_per_site():
    before = {site: [check.Reading("a", True, 0.1)] * 4 for site in support.SITES}
    after = {site: list(readings) for site, readings in before.items()}
    after["urgency"][0] = check.Reading("b", True, 0.1)
    after["urgency"][1] = check.Reading("b", False, 0.9)
    assert check.changes(before, after) == {"category": 0, "urgency": 2, "needs_human": 0}


def test_the_tallies_are_printed_as_shares():
    tallies = {"no gate": check.Tally(1000, 723, 703), "0.99 quantile": check.Tally(1000, 0, 0)}
    assert check.render_tallies("a title", tallies).splitlines() == [
        "a title",
        "  gate             answered locally  all three right when local",
        "  no gate                     72.3%                       97.2%",
        "  0.99 quantile                0.0%                           -",
    ]


def test_the_probes_are_printed_per_gate():
    stopped = {"asdf qwer zxcv": {"no gate": False, "0.99 quantile": True}}
    assert check.render_probes(stopped).splitlines() == [
        "junk and out-of-scope states, stopped if no head answers them",
        "  state           no gate  0.99 quantile",
        "  asdf qwer zxcv  no       yes",
    ]


def test_the_changes_are_printed_per_site():
    text = check.render_changes({"category": 21, "urgency": 102, "needs_human": 35}, 1000)
    assert text == (
        "changing only channel and plan changes the answer of category on 2.1%,"
        " urgency on 10.2%, needs_human on 3.5%"
    )
