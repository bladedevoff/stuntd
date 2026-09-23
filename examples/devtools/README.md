# A gate for coding agents over the Jev protocol

A coding agent takes a step: it wants to run a shell command, and it has the output of the last
test run in front of it. `client.py` asks two typed questions about that step in a single
`POST /v1/systemone`: may the command run, and was the last failure a flake.

`generate.py` is the data: 26 seeded commands, 12 test failures, five repositories and five
branches. Nothing but the standard library.

The demo needs `pip install typesafe-sdk`; the daemon needs `pip install "stuntd[train]"`.

## The questions and the state

```json
{"repo": "payments-api", "branch": "feature/webhook-retry",
 "command": "git push --force origin main",
 "last_failure": "tests/test_api.py::test_sync timed out after 30s"}
```

| question | type | answers |
| --- | --- | --- |
| `command_gate` | choice | allow, ask, deny |
| `flaky` | noul | true / false |

**The teacher is a rule, not an LLM, and both rules read one field of the state.**

- `command_gate` reads the command and nothing else. `deny` for anything that deletes outside the
  repository (`rm -rf ~`, `rm -rf /`), force-pushes, pipes a download into a shell (`| sh`), takes
  root (`sudo `), reads a secret (`cat ~/.ssh`, `env | grep`) or drops a database. `ask` for
  anything that installs a package, commits, pushes or reaches the network (`pip install`,
  `npm install`, `apt-get install`, `git push`, `git commit`, `curl `, `gh pr`). `allow` for the
  rest, which is reading the repository and building or testing inside it -- `rm -rf build/`
  included, because it deletes inside the repository.
- `flaky` reads the failure text: a timeout, a refused or reset connection, a port already in use
  or a test that only fails next to others is a flake; a plain assertion, an import error or a
  syntax error is not, and a step with no failure is not either.

The `repo` and `branch` fields are context: a real hook can fill them in, and neither rule turns
on them. One step in five carries no failure at all, because that is the state a `PreToolUse` hook
sends -- the four fields here are exactly what such a hook can say about a command it is about to
run.

With the paid Jev API in front, the same demo runs with it as the teacher instead: put
`[jev] upstream = "https://api.typesafe.ai"` in `stuntd.toml`, pass your key
(`--api-key "$TYPESAFE_API_KEY"`), skip `stuntd import` and let `client.py` collect the captures.
That variant is not measured here; every number below comes from the rule teacher.

## The demo in three steps

```
python generate.py --rows 3000 --out rows
stuntd import command_gate rows/command_gate.jsonl --labels allow,ask,deny
stuntd import flaky rows/flaky.jsonl --kind boolean
```

```
stuntd train && stuntd report
stuntd enable command_gate && stuntd enable flaky
```

```
stuntd serve
python client.py --n 200
```

`client.py` draws 200 steps from a seed the training rows were not drawn with, skipping any
state the training draw already contained, and sends each one. It prints the first step with
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

3000 steps from seed 1, so 3000 rows per site; `stuntd train` fitted 2400 and held 600 back per
site and took **231 s for both sites at `epochs = 24`** on a laptop RTX 5060, with
`convaiinnovations/laya` as the base checkpoint. The frozen encoder runs once per example rather
than once per epoch; with `[training] cache_encoder = false` the same run takes 730 s.
`stuntd report`:

| site | holdout agreement | ECE | operating point at target 0.99 |
| --- | --- | --- | --- |
| `command_gate` | 0.985 | 0.032 | coverage 0.98 at agreement 0.991 |
| `flaky` | 1.000 | 0.000 | coverage 1.00 at agreement 1.000 |

Then 200 steps no training row contained, before and after `stuntd enable`:

| run | command_gate | flaky | p50 per request | answered by a head |
| --- | --- | --- | --- | --- |
| base Laya, zero-shot | 45.0% | 84.5% | 27.9 ms | 0/400 |
| trained heads | 97.0% | 100.0% | 56 ms | 390/400 (97.5%) |

Those are one run of a fit that is seeded but not reproducible to the digit: re-running it lands
around 0.973 to 0.985 holdout agreement and 0.95 to 0.98 coverage on `command_gate`, 56 to 58 ms
p50, with `flaky` at 1.000 throughout.

**Why 24 epochs and 3000 rows.** Both were measured, on `command_gate`, the harder of the two
sites: at 1500 rows it scores 0.473 at 3 epochs (the default), 0.717 at 12 and 0.863 at 24; at 3000
rows and 24 epochs it reaches 0.985. The cost is the 231 s above; at the default 3 epochs the same
two sites train in 95 s and answer 3% of the questions -- at three epochs the one encoder pass is
most of the work, so the cache saves little there and a great deal at 24.

The gate is worth reading twice: the base checkpoint answers 45% of these commands correctly, and
the head that costs four minutes of a laptop GPU answers 97% of them, locally, at 56 ms per
request. That is the case for a hook that does not send every command you type to a provider.

## As a Claude Code hook

`command_gate` is the shape of a `PreToolUse` hook: Claude Code hands the hook the tool input on
stdin, exit code 2 blocks the call and sends stderr back to the model, and exit code 0 leaves the
normal permission flow alone.

```python
#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

from typesafe_sdk import Choice, TypeSafeClient

GATE = {"command_gate": Choice(criteria={"allow": None, "ask": None, "deny": None})}
BRANCH = ["git", "rev-parse", "--abbrev-ref", "HEAD"]

payload = json.load(sys.stdin)
command = payload["tool_input"].get("command", "")
cwd = Path(payload.get("cwd", "."))
branch = subprocess.run(BRANCH, cwd=cwd, capture_output=True, text=True).stdout.strip()
state = {"repo": cwd.name, "branch": branch, "command": command, "last_failure": "none"}

with TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8787") as client:
    verdict = client.system_one(state=state, questions=GATE).choices["command_gate"].choice
if verdict == "deny":
    print(f"stuntd denied this command: {command}", file=sys.stderr)
    sys.exit(2)
if verdict == "ask":
    reason = f"stuntd would ask about this command: {command}"
    decision = {"hookEventName": "PreToolUse", "permissionDecision": "ask"}
    print(json.dumps({"hookSpecificOutput": {**decision, "permissionDecisionReason": reason}}))
```

Save it as `gate.py` and point a `PreToolUse` matcher for `Bash` at it in `settings.json`; the
matcher is what keeps other tools away from the hook. `deny` blocks the call with exit code 2 and
sends its reason back to the model on stderr; `ask` exits 0 and prints the
`permissionDecision: "ask"` object Claude Code reads, which puts the command in front of you
instead of running it. The criteria are left undescribed here for brevity -- they matter only
before the head is trained, when the base checkpoint answers the question from the descriptions
alone. The hook sends the same four fields every row carries, with no test failure, which is why
one row in five is generated that way.
