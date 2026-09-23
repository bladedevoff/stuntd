# stuntd: a local Jev-compatible proxy that learns your typed LLM decisions

[![CI](https://github.com/bladedevoff/stuntd/actions/workflows/ci.yml/badge.svg)](https://github.com/bladedevoff/stuntd/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/stuntd.svg)](https://pypi.org/project/stuntd/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**stuntd is a local, self-hosted proxy that records the typed decisions your app already makes and
learns to answer them itself.** It speaks the Jev System One protocol (`POST /v1/systemone`, with
`choice`, `score` and `noul` questions) and the OpenAI Chat Completions API, distils each decision
site into a small head on a frozen [Laya](https://huggingface.co/convaiinnovations/laya) encoder,
and serves that answer locally with calibrated confidence, handing anything it is unsure about back
to the provider.

Three ways to run it:

- **Local Jev.** No key, no provider, no training. Point `typesafe-sdk` at stuntd and the base Laya
  checkpoint answers your typed questions.
- **In front of OpenAI.** A drop-in replacement for the provider base URL. Typed requests are
  recorded; everything else is relayed byte for byte.
- **In front of the paid Jev API.** stuntd relays, learns from the provider's own answers, and
  takes the decision over once its head is right often enough.

Status: first release, v0.1. Every number below traces to a demo in this repository or to a source
listed at the end.

## Who it is for

- **You have no Jev key.** TypeSafe's Jev is an early-access API at the time of writing, and there
  are no published weights. stuntd answers the same typed questions locally on Laya, with no key
  and no training, so a `typesafe-sdk` client keeps working while you wait.
- **You pay a provider for the same decision thousands of times a day.** Routing, triage,
  moderation, gating. stuntd distils those answers into a local head: tens of milliseconds, no
  tokens, no round trip.
- **Zero-shot Laya is not good enough on your labels.** On banking77's 77 intents it scores 38.2
  where Jev scores 76.4. A few thousand rows of your own traffic is what closes that gap: in the
  banking demo the risk question goes from 30.0% to 72.5%.
- **You want a Claude Code hook that decides locally.** A `PreToolUse` gate answering "may this
  command run" in about 50 ms, without sending every command you type to a provider.
  [`examples/devtools/`](examples/devtools/) ships the hook.

## Quickstart: local Jev without an API key

```
pip install "stuntd[train]"
stuntd config init
stuntd serve
```

`config init` writes a commented `stuntd.toml` with everything at its default, which means no
OpenAI upstream and no Jev provider. `serve` says so and loads the base checkpoint:

```
stuntd listening on http://127.0.0.1:8787
no upstream: only the Jev routes are served
serving jev locally with convaiinnovations/laya
```

Then change the base URL in your client and nothing else:

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

QUESTIONS = {
    "category": Choice(
        instructions="Which part of the product is this ticket about?",
        criteria={"billing": "Payments and refunds.", "bug": "Something is broken."},
    ),
    "needs_human": Noul(instructions="Does a support agent have to read this?"),
}

with TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8787") as client:
    answer = client.system_one(
        state={"subject": "Charged twice", "body": "Two identical charges on Friday."},
        questions=QUESTIONS,
    )
print(answer.choices["category"].choice, answer.nouls["needs_human"].noul)
```

Anything that speaks the protocol plugs in the same way, by changing one base URL wherever the
client exposes it: `typesafe-sdk`, JevRouter, the local-server request of fast-jev-compaction, and
the browser and game agents built on the same SDK.

The key is ignored unless you set `[jev] require_key = true`. To train a head on this daemon you
need labelled rows: either put the paid Jev API in front of it (`[jev] upstream`) and let it teach,
or write the rows yourself and `stuntd import` them.

## Quickstart: OpenAI-compatible proxy that learns

```
pip install "stuntd[train]"
stuntd config init
stuntd serve --upstream https://api.openai.com
```

`--upstream` is the provider origin only, without `/v1`: your client's own path is appended to it
unchanged. In the client, point `base_url` at the daemon:

```python
import os

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key=os.environ["OPENAI_API_KEY"])
```

Everything keeps working exactly as before. A request whose `response_format` (or single tool) asks
for an object with one enum, boolean or number field is a typed decision, and stuntd records it
with the provider's answer. Anything else is relayed untouched, streaming included.

Once a site has a few hundred captures:

```
stuntd status                  # what has been recorded, per site
stuntd train                   # one head per site with enough examples
stuntd report                  # holdout agreement, ECE, the operating point
stuntd enable <site>           # let that site answer locally
```

## Numbers

What a trained head does, measured on a laptop RTX 5060 with `convaiinnovations/laya` as the base
checkpoint, 3000 generated rows per site and 24 epochs (5890 rows for Snake), with
`training.cache_encoder` at its default. "Zero-shot" is the same daemon before training, answering
from the base checkpoint alone.

| demo | questions | zero-shot | trained heads | answered locally | p50 per request |
| --- | --- | --- | --- | --- | --- |
| [Snake](examples/snake/) | 1 choice | 0.70 average score | 11.40 average score | 99.6% | 21.1 ms |
| [support](examples/support/) | choice, score, noul | 69.5 / 34.5 / 66.5% | 99.5 / 85.5 / 97.5% | 90% | 90.3 ms |
| [banking](examples/banking/) | 2 choice | 89.5 / 30.0% | 100.0 / 72.5% | 78% | 75.7 ms |
| [devtools](examples/devtools/) | choice, noul | 45.0 / 84.5% | 97.0 / 100.0% | 98% | 58.0 ms |

Five of the seven sites in the three text demos reach a holdout agreement of 0.95 or better. The
demos run at the default `target_agreement` of 0.99, which picks each head's operating point rather
than being a score it reached. The two that fall short, support's `urgency` at 0.935 and banking's
`risk` at 0.937, stay in the demos anyway: see [limitations](#limitations).

Latency, size and cost:

| | measured |
| --- | --- |
| one head answer on CUDA | p50 22 ms, 26 ms in process |
| the same head on CPU | 60 to 62 ms |
| one live answer through the daemon, over TCP | 28 ms |
| what the relay itself adds when the provider answers | 2.1 ms |
| Jev API round trip, through OpenRouter | p50 376 to 389 ms |
| Jev price | $0.042 per 1M input tokens, output free |
| training one head: 5890 rows, 24 epochs | 186 s |
| training three heads: 3000 rows each, 24 epochs | 362 s |
| a trained head on disk | about 50 MB, fp16 |

The first four rows were measured during development on an RTX 5060 laptop running Windows. They
come from no published benchmark, and there is no link to give for them. The two training rows are
the Snake and support demos of this repository, on the same laptop with the encoder cache on; the
devtools demo measured the same 24-epoch run at 730 s with the cache off against 231 s with it on,
3.2x.

Why train at all, rather than run Laya zero-shot. From
[jevbench](https://github.com/dhruvmehra/jevbench), n=500 per dataset, accuracy:

| dataset | labels | Laya zero-shot | Jev | fine-tuned DistilBERT |
| --- | --- | --- | --- | --- |
| sst2 | 2 | 92.0 | 95.4 | 91.0 |
| agnews | 4 | 90.6 | 84.3 | 91.0 |
| banking77 | 77 | 38.2 | 76.4 | 88.0 |

Zero-shot Laya holds its own at two or four labels and collapses at seventy-seven. A model trained
on a few thousand of your own examples does not: that is the bet stuntd makes, and the four demos
are what it is measured against.

## How stuntd distils decisions into a Laya head

```
   your app
      |
      v
  +-------------------------------------------------+
  |  stuntd                                          |
  |                                                  |
  |   typed decision?  -- no --------------------------->  provider, byte for byte
  |         |                                        |
  |        yes                                       |
  |         v                                        |
  |   head confident?  -- yes -->  local answer      |     20 to 100 ms, no tokens
  |         |                                        |
  |         no, or a 2% check sample                 |
  +---------|----------------------------------------+
            v
        provider  (answer returned to the caller, and recorded as a capture)
```

A **decision site** is one place in your code that makes one decision. On the Jev path the site is
the question's name. On the OpenAI path it is a hash of the response schema and the system prompt,
or whatever you put in the `X-Stuntd-Site` request header.

Each site walks through four modes.

1. **collect.** The provider answers everything. stuntd stores the request text and the answer,
   with the redaction rules applied before anything is written.
2. **train.** `stuntd train` fits one classification head per site on the frozen Laya encoder,
   holds back 20% of the rows, and reports the holdout agreement, the calibration error and the
   confidence threshold at which the head reaches your `target_agreement`:
   `category  trained  holdout agreement 0.992  coverage 1.00 at agreement 0.992  ece 0.024`.
   Below that threshold the head abstains, which is what coverage measures: the share of requests
   it is willing to answer at all. `stuntd enable` refuses a head whose coverage rounds to zero and
   warns below 10%.
3. **shadow.** The caller still gets the provider's answer; the head answers beside it and the two
   are compared. `stuntd status` shows the running agreement.
4. **live.** `stuntd enable <site>` and the head answers the caller. Anything under the threshold
   still goes to the provider. So does a `check_share` sample, 2% by default, and those samples are
   what keeps the head honest: once agreement over the last `window` decisions falls below
   `target_agreement`, the site demotes itself back to shadow and logs why. Promotion the other way
   is manual unless you set `serving.auto_promote`.

In proxy mode the provider behind `[jev] upstream` does not have to be the paid API. Any server
that speaks `POST /v1/systemone` can sit there -- kev, LLM2Jev, the laya-server family -- which
makes it a keyless local teacher: it answers, stuntd records, and the head learns from it.

Nothing is uploaded. Captures live in a SQLite file under your user directory with private
permissions, and there is no telemetry.

### The `X-Stuntd` header

Every answer says how it was served, so a silent fallback cannot hide.

On the OpenAI path: the mode, then the fields that apply.

```
X-Stuntd: live; site=moderation; confidence=0.97
X-Stuntd: collect; site=moderation; reason=low-confidence
X-Stuntd: passthrough; reason=no-schema
```

| field | values |
| --- | --- |
| mode | `live`, `shadow`, `collect`, `passthrough` |
| `site=` | the decision site this request landed on |
| `confidence=` | the head's calibrated confidence, two decimals, on a live answer |
| `reason=` | `no-schema`, `streaming`, `learning-off`, `no-upstream`, `low-confidence`, `check`, `retrained`, `model-error`, `no-runtime`, `state-error`, `upstream-error` |

On the Jev path: the mode it served in, then how the questions were answered.

```
X-Stuntd: jev; mode=local; questions=3; live=2; zeroshot=1
X-Stuntd: jev; mode=proxy; questions=2; live=0; shadow=1; check=0
X-Stuntd: jev; mode=local; questions=1; live=0; zeroshot=1; learn=off
X-Stuntd: jev; mode=local; reason=no-key
```

| field | values |
| --- | --- |
| `mode=` | `local` (no Jev provider behind it) or `proxy` (the paid API is) |
| `questions=` | how many questions the request asked |
| `live=` | answered by a trained head |
| `shadow=` | site in shadow mode, proxy mode only |
| `zeroshot=` | answered by the base checkpoint, local mode only |
| `check=` | sampled for a comparison with the provider, proxy mode only |
| `learn=off` | `learn = false`, so nothing was recorded |
| `reason=` | `no-key`, `bad-request`, `not-parsed`, `no-runtime`, `model-error`, `upstream-error` |

## Configuration

`stuntd config init` writes the file; `stuntd config show` prints every setting with where it came
from. Every key is optional and a missing one keeps the default.

| key | default | what it does |
| --- | --- | --- |
| `upstream` | `""` | Provider origin the OpenAI path forwards to, no path. Empty: only the Jev routes are served. |
| `host` | `127.0.0.1` | Address the daemon listens on. |
| `port` | `8787` | Port the daemon listens on. |
| `learn` | `true` | Whether the daemon records anything at all. `false` is a pure proxy, or a pure local Jev. |
| `redaction.enabled` | `true` | Built-in email and phone rules run before a capture is written. |
| `redaction.patterns` | `[]` | Extra regular expressions, each replaced with `[redacted]`. |
| `storage.db_path` | `captures.sqlite` | Capture database; a relative path resolves against the data directory. |
| `storage.max_rows` | `100000` | Captures kept before the oldest are pruned. |
| `storage.max_age_days` | `30` | Age at which a capture is pruned. |
| `storage.models_dir` | `models` | Where trained heads live. |
| `training.min_examples` | `300` | Captures a site needs before it is worth training. |
| `training.holdout` | `0.2` | Share of the rows held back to measure the head. |
| `training.target_agreement` | `0.99` | Agreement with the teacher the head must reach; it sets the confidence threshold. |
| `training.base_model` | `convaiinnovations/laya` | Checkpoint the heads are trained on. A local directory works too. |
| `training.epochs` | `3` | Passes over the training rows. The demos use 24. |
| `training.device` | `auto` | `auto`, `cpu`, `cuda` or `mps`. |
| `training.cache_encoder` | `true` | Whether the frozen encoder runs once per example instead of once per epoch; off re-encodes. |
| `training.cache_max_mb` | `4096` | Most memory the cached encoder output may take, in megabytes; a bigger site trains uncached. |
| `serving.check_share` | `0.02` | Share of live requests still sent to the provider to check the head. Inert in local Jev mode. |
| `serving.window` | `100` | Recent decisions a site is judged on. |
| `serving.min_window` | `20` | Decisions needed before that judgement counts. |
| `serving.auto_promote` | `false` | Whether a shadow site that holds its target goes live on its own. |
| `serving.auto_promote_after_hours` | `24` | Hours in shadow before it may. |
| `serving.cache_size` | `1000` | Answers kept in memory, one LRU cache shared by every site. |
| `jev.upstream` | `""` | Origin of the paid Jev provider, no path. Empty: the base Laya answers locally. |
| `jev.require_key` | `false` | Whether a Jev request must carry a Bearer token. The token itself is not checked. |
| `jev.model_name` | `stuntd` | Name reported in the `model` field of a Jev answer. |

## Demos

Four runnable demos, each with its own README, its own generated data and its own measured table.

- [`examples/snake/`](examples/snake/) plays 12x12 Snake over the Jev protocol, one `choice` per
  move. Zero-shot Laya scores 0.70 a game; the head scores 11.40 at `target_agreement = 0.95`,
  answering all but 13 of 3050 moves locally at a p50 of 21.1 ms. At the default 0.99 the same head
  scores 3.03 and answers 71% of the moves. That is the difference between "let the head play" and "only when
  it is sure", in one table.
- [`examples/support/`](examples/support/) triages a helpdesk ticket with three questions in one
  request: `category` (choice), `urgency` (score) and `needs_human` (noul). 90% of the questions
  end up answered locally.
- [`examples/banking/`](examples/banking/) asks a banking intent over 12 labels and a payment risk
  verdict. 78% local, and `intent` reaches a holdout agreement of 1.000.
- [`examples/devtools/`](examples/devtools/) gates a coding agent's shell command and calls a test
  failure flaky. 98% local, and it ships the Claude Code `PreToolUse` hook that uses it.

**The teachers in all four are rules, not an LLM.** Three demos use a written rule over the
generated text; Snake uses a BFS oracle. No Jev API key was available when these numbers were
taken, so the zero-shot rows are the base checkpoint, not the paid provider, and the same tables
with the real Jev API as the teacher have not been measured yet. What the demos do show is what a
head learns from a teacher that is consistent: the ceiling of the method, not of any one provider.

## Limitations

**The encoder is frozen, so a head learns text, not arithmetic over fields.** This is the single
most important thing to know. In the banking demo the first version of the `risk` rule compared the
amount, the country and the hour numerically: the head reached 0.42 and covered 1% of the holdout.
The same decision, written as a sentence about the customer's own history, reaches 0.937 at
coverage 0.57. If your decision is really a calculation, write the calculation, not a model.

- **Free text is not a decision.** stuntd learns closed decisions only: an enum, a boolean or a
  number on the OpenAI path, `choice`, `noul` or `score` on the Jev path. A request that asks for
  prose, a summary or code is relayed and never recorded.
- **Some decisions need more than the default 3 epochs.** The default stays 3, and the demos
  train at 24 because each of them measured what that bought. The frozen encoder runs once per
  site and every epoch reads its cached output, so the extra epochs cost seconds, not minutes.
  At 3 epochs the same sites answer almost nothing.
- **Overlapping labels make a head defer.** The weakest site of the three text demos is the
  support demo's `urgency`, a four-level `score`. Its levels overlap in wording, so the head
  reaches a 0.935 holdout agreement and answers 71% of the tickets at 99.1% agreement, handing the
  other 29% to the provider. That is a working site rather than a failure, but it is the shape to
  expect when the labels are not cleanly separated by words.
- **A site needs examples.** `training.min_examples` is 300 and the demos use 3000 rows. A rare
  decision, or one whose option list changes every week, is not a fit.
- **`score` takes 2 to 10 levels and `choice` takes at most 255 options.** Those are stuntd's own
  limits, not a quotation from the Jev API. `instructions` is optional on a question.
- **A question name is a decision site.** `:` becomes `.`, so `a:b` and `a.b` are one site. Two
  questions with the same name but different criteria train one head on mixed data; give them
  different names.
- **`GET /v1/models` is answered by stuntd**, in the Jev shape, naming `jev.model_name`. In proxy
  mode it is relayed to the Jev provider. Either way an OpenAI client does not get the provider's
  model list through stuntd; every other `/v1/models...` path is relayed as usual.
- **Local Jev mode compares nothing.** There is no second opinion to compare against, because the
  zero-shot answers come from the very checkpoint the head was distilled from. No request is
  sampled for a check, `stuntd status` reports `"agreement": null`, and no site is ever demoted.
  `stuntd report`, which measures the head against held-out rows, is the quality signal there. Put
  a Jev provider in front and both come back.
- **Only the OpenAI Chat Completions path learns.** Anthropic Messages, the OpenAI Responses API
  and Gemini pass through untouched. Claude Code, Codex CLI and Gemini CLI run through stuntd as if
  it were not there.
- **One CUDA workload at a time.** Stop the daemon before `stuntd train`, or the two contend for
  the same GPU.
- **Every `serve` loads the base checkpoint** when `[jev] upstream` is empty, because local Jev
  answers from it. The daemon runs one pass through the model before it accepts requests, so
  startup costs seconds and the first request then runs close to the steady-state p50. Lazy
  loading is on the roadmap.

## Compared with

- [**stiermid/laya-serve**](https://github.com/stiermid/laya-serve) is a Jev-compatible HTTP server
  for Laya weights. It is the cleaner choice when zero-shot is all you want: it documents its
  divergences from the Jev wire format carefully, and it carries no database, no training and no
  proxy. stuntd's local mode does the same job; what it adds is learning from traffic.
- The rest of the laya-server family serves those weights behind the same API:
  [noahbclarkson/laya-server](https://github.com/noahbclarkson/laya-server) ships a container, a
  health endpoint and TypeSafe's error shapes, [receptron/laya](https://github.com/receptron/laya)
  runs Laya from Node through ONNX Runtime. All of them answer zero-shot only.
- [**jaredpalmer/kev**](https://github.com/jaredpalmer/kev) trains Jev-like decision models on
  Qwen3.5 and publishes the weights at 0.8B, 4B and 9B, with the training code, frozen eval suites
  and a System One API in front of them. A whole model per decision, where stuntd trains one head
  per site on a frozen encoder.
- [**Yinsongxu/LLM2Jev**](https://github.com/Yinsongxu/LLM2Jev) puts local LLMs behind
  `/v1/systemone` and scores each option during the prefill, so reordering the options does not
  change the answer. stuntd has neither that order independence nor its image inputs; what it has
  is a head that answers without the LLM.
- [**TheoLeeCJ/SemIf**](https://github.com/TheoLeeCJ/SemIf) reads typed option probabilities
  straight out of open 4B models, with a browser demo and no waitlist. It reproduces the interface
  rather than the model, and every decision still costs a pass through the 4B.
- [**TianyuCodings/NanoJev**](https://github.com/TianyuCodings/NanoJev) trains a 0.6B decision
  model from prepared game datasets and plays ViZDoom, a maze and Snake with the one checkpoint.
  Its Snake comes from a model trained for the game; stuntd's comes from a head distilled from
  5890 recorded moves of a BFS oracle.
- [**vinnylarouge/jevlike**](https://github.com/vinnylarouge/jevlike) trains an option-attention
  head from JSONL rows of a context, its options and a label. That is `stuntd train` as a
  standalone library: you bring the rows, and nothing records them for you.
- [**gargpratyush/jev-router**](https://github.com/gargpratyush/jev-router) uses Jev to pick a
  model tier per turn for Claude Code and Codex. A different problem, and a good example of the
  decision stuntd would learn: a routing call on every turn is exactly the repeated typed decision
  that pays for a head.
- [**Santiago-j-s/jev-proxy**](https://github.com/Santiago-j-s/jev-proxy) sits where stuntd sits
  and records every Jev request and answer to SQLite, with its latency, tokens and cost, for seven
  days. It is a debugger rather than a learner: it trains nothing and answers nothing itself.
- [**dhruvmehra/jevbench**](https://github.com/dhruvmehra/jevbench) is the benchmark this README
  quotes for the zero-shot numbers. If you are choosing between Jev, Laya, an LLM and a fine-tuned
  BERT for your labels, run it first. stuntd's answer to "Laya is worse than Jev on 77 labels" is
  to train on your own data, but measuring comes first.
- [**yibie/awesome-jev**](https://github.com/yibie/awesome-jev) is the larger of the two catalogues
  of public Jev projects, integrations and discussions. A good place to check whether your decision
  is a pattern other people already run.

None of them learns from live traffic: recording the provider's own answers, a shadow phase, a
calibrated threshold and a fallback for everything under it is the loop stuntd adds. And every
Jev-compatible server above -- the laya-server family, kev, LLM2Jev -- can sit behind stuntd as
`[jev] upstream`, which makes any of them a local teacher that needs no key.

## Roadmap

Next: support for new formats.

- Adapters for Anthropic Messages, the OpenAI Responses API and Gemini, so those paths learn too.

Later:

- Lazy checkpoint load, so a daemon that serves nothing locally starts at once.
- Encode the state once per request instead of once per question, for multi-question latency.
- Option-order augmentation during training.
- Dynamic candidate options, so a site whose option list changes does not need a new head.
- Export a trained site as a full Laya checkpoint.
- Unfreezing the top encoder layers, for decisions a frozen encoder cannot read.
- `stuntd train` on a schedule.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). `make check` is what CI runs: `ruff format --check`,
`ruff check`, `mypy`, `pyright` and `pytest`. The same file for LLM contributors is
[AGENTS.md](AGENTS.md).

## License

Apache-2.0. See [LICENSE](LICENSE).

The decision heads are trained on [Laya](https://huggingface.co/convaiinnovations/laya) by Convai
Innovations, Apache-2.0.

## Sources

- Jev pricing, context limit and model sheet:
  [docs.typesafe.ai/models](https://docs.typesafe.ai/models). $0.042 per 1M input tokens, output
  free, 64k context, text input only.
- Zero-shot accuracy and API latency: [dhruvmehra/jevbench](https://github.com/dhruvmehra/jevbench),
  the run of 2026-09-22, n=500 per dataset
  ([summary](https://github.com/dhruvmehra/jevbench/blob/main/docs/results/2026-09-22-n500-summary.md)).
- Jev wire format and its documented divergences from Laya:
  [stiermid/laya-serve](https://github.com/stiermid/laya-serve).
- Head latency, relay overhead and head size: measured during development on an RTX 5060 laptop
  running Windows. Not public, so there is no link.
- Every other stuntd number: the four demo READMEs linked above, each of which states the
  hardware, the row count and the epochs it was measured at.
