# Banking intents and payment risk over the Jev protocol

One request from a banking app carries what the customer wrote and the payment sitting in front of
them. `client.py` asks two typed questions about it in a single `POST /v1/systemone`: what the
message is about, and what should happen to the payment.

`generate.py` is the data: 24 seeded message templates over 12 intents, and a payment described
both as fields and as a line of context that says how it compares with what this customer usually
does. Nothing but the standard library.

The demo needs `pip install typesafe-sdk`; the daemon needs `pip install "stuntd[train]"`.

## The questions and the state

```json
{"message": "I left the blue card in a taxi yesterday, please freeze it.",
 "context": "much larger than anything they have sent before; first transfer to this payee",
 "amount": "38,916.42 EUR", "payee": "Sam Okafor", "country": "IT", "time": "12:48"}
```

| question | type | answers |
| --- | --- | --- |
| `intent` | choice | card_arrival, card_lost, card_declined, atm_fee, transfer_pending, transfer_failed, exchange_rate, top_up_failed, verify_identity, close_account, direct_debit_dispute, change_pin |
| `risk` | choice | allow, review, block |

**The teacher is a rule, not an LLM, and both rules read words.** The intents are synthetic rather
than Banking77: a Banking77 subset (CC-BY-4.0) would have to be fetched at generation time, which
costs the demo its offline reproducibility, and its messages carry no payment to ask `risk` about.

- `intent` is the template family the message was drawn from.
- `risk` reads the context line, which is the bank's own summary of the payment against the
  customer's history: two points for "much larger than anything they have sent before", one for
  "larger than their usual payments", and one each for "first transfer to this payee", "outside
  their usual hours" and "to a country they have never sent money to". Three points or more is
  `block`, two is `review`, one or none is `allow`.

The `amount`, `country` and `time` fields agree with the context line -- a payment called much
larger is drawn from the top band, one called out of hours is timed at night -- but the rule reads
the words, not the numbers. That is the point of this round: a head on a frozen encoder learns a
decision that lives in the language and not one that lives in arithmetic over fields.

With the paid Jev API in front, the same demo runs with it as the teacher instead: put
`[jev] upstream = "https://api.typesafe.ai"` in `stuntd.toml`, pass your key
(`--api-key "$TYPESAFE_API_KEY"`), skip `stuntd import` and let `client.py` collect the captures.
That variant is not measured here; every number below comes from the rule teacher.

## The demo in three steps

```
python generate.py --rows 3000 --out rows
stuntd import intent rows/intent.jsonl --labels card_arrival,card_lost,card_declined,atm_fee,transfer_pending,transfer_failed,exchange_rate,top_up_failed,verify_identity,close_account,direct_debit_dispute,change_pin
stuntd import risk rows/risk.jsonl --labels allow,review,block
```

```
stuntd train && stuntd report
stuntd enable intent && stuntd enable risk
```

```
stuntd serve
python client.py --n 200
```

`client.py` draws 200 requests from a seed the training rows were not drawn with, skipping any
state the training draw already contained, and sends each one. It prints the first request with
both answers, then the agreement with the rule teacher per question, the p50 latency and the share of
questions the trained heads answered, read from the `X-Stuntd` header. With no header on the reply
it points at `stuntd status --json` instead.

The demo sets two keys in `stuntd.toml`: `training.base_model`, which points at a local Laya
checkpoint, and `training.epochs = 24`. Leave `upstream` commented out -- with no OpenAI upstream
the daemon serves the Jev routes alone and says so, `no upstream: only the Jev routes are served`.
Everything else is the default, `target_agreement = 0.99` included. 24 epochs and 3000 rows are
both measured choices, not guesses -- the numbers below say what each one buys.

`stuntd status --json` reports `"agreement": null` for every site here: in local mode nothing
compares the head with a bigger model, so there is no agreement to report until a Jev provider is
put in front of it.

## What it measured

3000 requests from seed 1, so 3000 rows per site; `stuntd train` fitted 2400 and held 600 back per
site and took **313 s for both sites at `epochs = 24` with the encoder cache** on a laptop RTX
5060, with `convaiinnovations/laya` as the base checkpoint. `stuntd report`:

| site | holdout agreement | ECE | operating point at target 0.99 |
| --- | --- | --- | --- |
| `intent` | 1.000 | 0.000 | coverage 1.00 at agreement 1.000 |
| `risk` | 0.937 | 0.119 | coverage 0.57 at agreement 0.991 |

Then 200 requests no training row contained, before and after `stuntd enable`:

| run | intent | risk | p50 per request | answered by a head |
| --- | --- | --- | --- | --- |
| base Laya, zero-shot | 89.5% | 30.0% | 39.9 ms | 0/400 |
| trained heads | 100.0% | 72.5% | 75.7 ms | 313/400 (78.2%) |

Run it again and the numbers move, `risk` most of all. The same head retrained on the same rows at
three seeds in the devtools demo scored 0.985, 0.978 and 0.973 on its holdout at coverage 0.98,
0.98 and 0.95: a small move in agreement is a large move in coverage, because coverage is the
widest threshold that still reaches the target. An earlier run of this demo read 0.952 and
coverage 0.87 on `risk`, and answered 94% of the questions locally.

**Why 24 epochs and 3000 rows.** Both were measured, on `risk`, the harder of the two sites: at
1500 rows it scores 0.367 at 3 epochs (the default), 0.693 at 12 and 0.840 at 24; at 3000 rows and
24 epochs it reaches 0.937. `intent` went from 0.993 to a holdout it does not get a single question
wrong on. The cost is the 313 s above: the frozen encoder runs once per site and every epoch after
that reads its cached output. At the default 3 epochs `risk` answers nothing at all.

The difference from the first version of this demo is the teacher, not the trainer: when `risk` was
arithmetic over the amount, the country and the hour it reached 0.42 and covered 1% of the holdout.
The same decision, written as a sentence about the customer's own history, is 0.937 at coverage
0.57. The zero-shot row is what the base checkpoint makes of the same question from the criteria
alone, which is why `risk` reads 30% there: counting points is not something it does without
having been shown.
