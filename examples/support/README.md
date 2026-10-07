# Support triage over the Jev protocol

Every ticket that arrives is one state and three typed decisions: which part of the product it is
about, how soon it has to be answered, and whether a human has to read it. `client.py` asks all
three in a single `POST /v1/systemone`, so one round trip triages the ticket.

`generate.py` is the data: 20 seeded templates over five categories, each one filled with a
product, a timeframe, an amount and a ticket number, and then given the sentences that say how bad
it is -- an outage, a customer who is stuck, a deadline, or a note that there is no rush. Only a
template about something that broke can carry an outage or a blocked line, so no feature request
claims the whole team is down, and each template reads its time phrases from the pool that fits it
("all week" for something still happening, "last month" for an admin who left). Nothing but the
standard library.

The demo needs `pip install typesafe-sdk`; the daemon needs `pip install "stuntd[train]"`. Both come
with `pip install "stuntd[train,jev]"`.

## The questions and the state

The state is a flat object, and the row a site is imported with is that object serialised exactly
the way stuntd serialises a Jev state:

```json
{"channel": "in_app", "plan": "free",
 "subject": "Cannot sign in to the billing portal",
 "body": "Every sign-in attempt since Friday comes back with a session error. We cannot carry on until it is sorted. Our audit is on Friday, so the timing matters."}
```

| question | type | answers |
| --- | --- | --- |
| `category` | choice | billing, bug, feature, account, other |
| `urgency` | score | 4 levels, 0 (says itself it can wait) to 3 (an outage, or stuck with a deadline) |
| `needs_human` | noul | true / false |

**The teacher is a rule, not an LLM, and every rule reads the words of the ticket.** The three
functions live in `generate.py`:

- `category` is the template family the ticket was drawn from.
- `urgency` reads the body: an outage ("production outage", "everyone in the company", "the whole
  team is down", "all of our customers") is 3; a customer who says they are stuck ("we cannot carry
  on", "blocking the work", "cannot get any further", "have to do it by hand") is 2; a ticket that
  says it can wait ("no rush", "not urgent", "for future reference") is 0; anything else is 1. A
  deadline ("before our release", "audit is on Friday", "renewal is due") adds one level to
  everything except the calm ones.
- `needs_human` is true when the ticket asks for money back, for a person or for the contract to
  end ("refund", "charged twice", "money back", "speak to someone", "cancel our contract",
  "compensation"), and for every ticket at urgency 3.

The `channel` and `plan` fields are context: they are in the state because a real helpdesk sends
them, and no rule reads them.

With the paid Jev API in front, the same demo runs with it as the teacher instead: put
`[jev] upstream = "https://api.typesafe.ai"` in `stuntd.toml`, pass your key
(`--api-key "$TYPESAFE_API_KEY"`), skip `stuntd import` and let `client.py` collect the captures.
That variant is not measured here; every number below comes from the rule teacher.

## The demo in three steps

```
python generate.py --rows 3000 --out rows
stuntd import category rows/category.jsonl --labels billing,bug,feature,account,other
stuntd import urgency rows/urgency.jsonl --labels 0,1,2,3
stuntd import needs_human rows/needs_human.jsonl --kind boolean
```

```
stuntd train && stuntd report
stuntd enable category && stuntd enable urgency && stuntd enable needs_human
```

```
stuntd serve
python client.py --n 200
```

`--labels 0,1,2,3` is how a score site is imported: the site's labels have to be the same set the
Jev question carries, and a 4-level score question is answered over the labels `"0"` to `"3"`.

`client.py` draws 200 tickets from a seed the training rows were not drawn with, skipping any
state the training draw already contained, and sends each one. It prints the first ticket with
its three answers, then the agreement with the rule teacher per question, the p50 latency and the share of
questions the trained heads answered, read from the `X-Stuntd` header. With no header on the reply
it points at `stuntd status --json` instead.

The demo sets two keys in `stuntd.toml`: `training.base_model`, which points at a local Laya
checkpoint, and `training.epochs = 24`. Leave `upstream` commented out -- with no OpenAI upstream
the daemon serves the Jev routes alone and says so, `no upstream: only the Jev routes are served`.
Everything else is the default, `target_agreement = 0.99` and the novelty gate included. 24 epochs and 3000 rows are
both measured choices, not guesses -- the numbers below say what each one buys.

`stuntd status --json` reports `"agreement": null` for every site here: in local mode nothing
compares the head with a bigger model, so there is no agreement to report until a Jev provider is
put in front of it.

## What it measured

3000 tickets from seed 1, so 3000 rows per site; `stuntd train` fitted 2400 and held 600 back per
site and took **266 s for the three sites at `epochs = 24` with the encoder cache** on a laptop
RTX 5060, with `convaiinnovations/laya` as the base checkpoint. `stuntd report`:

| site | holdout agreement | ECE | operating point at target 0.99 |
| --- | --- | --- | --- |
| `category` | 0.990 | 0.011 | coverage 1.00 at agreement 0.990 |
| `needs_human` | 0.970 | 0.079 | coverage 0.94 at agreement 0.991 |
| `urgency` | 0.922 | 0.092 | coverage 0.73 at agreement 0.991 |

Then 200 tickets no training row contained, before and after `stuntd enable`. These two runs, and
the 266 s above, were measured with 0.1.3 before the novelty gate existed and were not repeated for
0.1.4; `check.py` does not reproduce them. With the gate on, the share answered by a head is lower
(see the gate numbers below):

| run | category | urgency | needs_human | p50 per request | answered by a head |
| --- | --- | --- | --- | --- | --- |
| base Laya, zero-shot | 69.5% | 34.5% | 69.5% | 32.4 ms | 0/600 |
| trained heads | 99.5% | 84.0% | 95.0% | 82.8 ms | 543/600 (90.5%) |

Run it again and the numbers move a little. The same head retrained on the same rows at three
seeds in the devtools demo scored 0.985, 0.978 and 0.973 on its holdout, at coverage 0.98, 0.98
and 0.95, so a site quoted at 0.97 here is a site that lands near 0.97, not on it.

**Why 24 epochs and 3000 rows.** Both were measured, on `urgency`, the hardest of the three sites:
at 1500 rows it scores 0.323 at 3 epochs (the default), 0.510 at 12 and 0.643 at 24; at 3000 rows
and 24 epochs it reaches 0.890, and 0.935 once the generator stopped pairing outage sentences with
feature requests. Every site gained from the extra rows, so the demo ships at 3000. The cost is the
266 s above: the frozen encoder runs once per site and every epoch after that reads its cached
output. At the default 3 epochs the same three sites answer almost nothing.

`urgency` is the one site under the 0.95 target, and it stays because its operating point is
honest: it answers 73% of the tickets at 99.1% agreement and hands the rest to the provider. Its
four levels are a combination of two independent cues -- how bad it is, and whether there is a
deadline -- which is harder to read off one pooled embedding than the presence of a phrase.

**What these numbers do not measure.** The tickets come from 20 templates, and a label depends only
on the template and the impact and tail sentences added to it: 258 combinations in the 3000 rows.
The 200 test tickets are states the training rows did not contain, but nearly all of them repeat a
combination the heads trained on, with a different product, amount or number. So the table above
measures new wording of known tickets, not new kinds of ticket. Two checks say more, both run on
1000 tickets:

| check | all three heads sure | all three right when sure |
| --- | --- | --- |
| new states of the 20 trained templates | 72.3% | 97.2% |
| one template per category left out of training, tested on those five | 68.7% | 42.4% |

The left-out row moves far more between draws of the training rows than the others, because a head
that has never seen a template guesses: an earlier measurement of the same check, on another draw,
had 40.0% and 61.0%. Changing only `channel` and `plan`, which no rule reads, changes the answer of
`category` on 2.1% of the tickets, `needs_human` on 3.5% and `urgency` on 10.2%.

**The novelty gate.** `serving.novelty_gate` is on by default, with each head's cut-off at the 0.99
quantile of its holdout's novelty. With one template per category held out (heads trained on 3,000
rows of the other 15 templates), the gate let 0.3% of the 1,000 tickets from the held-out templates
be answered locally, against 68.7% without it; at the 0.95 and 0.98 quantiles it let none through.
On 1,000 unseen states of the 15 trained templates (seed 10001), 88.8% were answered locally
without the gate and 85.8% with it, and all three were right on 97.2% and 97.3% of the local ones.
The heads trained on all 20 templates answered 72.3% of the unseen states locally without the gate
and 70.6% with it, and all three were right on 97.2% in both cases. "What's the weather in Paris",
`asdf qwer zxcv` and an empty state were stopped on all three heads at every quantile; without
the gate none of them was. The main README's [novelty gate](../../README.md#the-novelty-gate)
section has the details and what it costs.

Both checks above were suggested by Dipankar Sarkar on the Hugging Face page of these heads, who also
found that the stuck-payout template tripped the "money back" cue without anyone asking for a
refund; its wording is fixed.

## Check it yourself

`check.py` reproduces every generalization number above, and the ones in the main README's
novelty gate section, from an empty data directory:

```
python check.py --data-dir check-data
```

It draws 3,000 tickets at seed 1, imports and trains the three sites through `stuntd import` and
`stuntd train` with `training.epochs = 24` and `training.cache_max_mb = 2048`, then does the same
with the first template of each category left out. It asks the heads about 1,000 tickets at seed
10001 that none of them was trained on, and recomputes each head's novelty cut-off at the 0.95,
0.98 and 0.99 quantiles from the holdout the way `stuntd train` builds it. It needs a GPU or a lot
of patience: the run below took about 23 minutes on the laptop RTX 5060 with
`convaiinnovations/laya`. The progress lines of the imports and of training are left out here;
the rest is what it printed:

```
heads trained on all 20 templates
  gate             answered locally  all three right when local
  no gate                     72.3%                       97.2%
  0.95 quantile               67.1%                       97.6%
  0.98 quantile               69.8%                       97.3%
  0.99 quantile               70.6%                       97.2%
changing only channel and plan changes the answer of category on 2.1%, urgency on 10.2%, needs_human on 3.5%
junk and out-of-scope states, stopped if no head answers them
  state                        no gate  0.95 quantile  0.98 quantile  0.99 quantile
  what's the weather in Paris  no       yes            yes            yes
  asdf qwer zxcv               no       yes            yes            yes
  {}                           no       yes            yes            yes
heads trained on 15 templates
  gate             answered locally  all three right when local
  no gate                     88.8%                       97.2%
  0.95 quantile               78.2%                       97.6%
  0.98 quantile               83.2%                       97.4%
  0.99 quantile               85.8%                       97.3%
tickets from the left-out templates
  gate             answered locally  all three right when local
  no gate                     68.7%                       42.4%
  0.95 quantile                0.0%                           -
  0.98 quantile                0.0%                           -
  0.99 quantile                0.3%                      100.0%
```

"Answered locally" means all three heads are sure and, under a gate, none of them finds the ticket
farther from its training rows than its cut-off; "stopped" means no head would answer the state.
The heads on all 20 templates came out within 0.2 points of the measurement made before this
script existed: 72.3% answered locally and 97.4% right without the gate then, 72.3% and 97.2% here.
`check.py` re-implements the gate rather than calling the daemon; on the same 1,000 tickets a
running daemon answered 708 locally (70.8%) against the 706 above. On other hardware expect small
differences.
