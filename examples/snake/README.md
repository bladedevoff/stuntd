# Snake over the Jev protocol

Every move of a 12x12 game of Snake is one typed decision. `game.py` is the game, the state a
decision site is asked about and a BFS oracle, with nothing but the standard library. `play.py` is
an ordinary `typesafe-sdk` client pointed at a running stuntd: one `choice` question per move,
named `direction`, with the criteria `up`, `down`, `left`, `right`. The daemon answers it from the
paid Jev API, from the base Laya checkpoint, or from the head it trained for that site.

The demo needs `pip install typesafe-sdk`; the daemon needs `pip install "stuntd[train]"`.

## What the question is asked about

`state_text(game)` is four lines and no picture:

```
dir=right head=(6,6) food=(10,2) length=3
food: 4 right, 4 up
safe: up, down, right
blocked: none
```

The board is drawn by `board_text(game)`, which `play.py` prints at the end of a run and which
never reaches the model:

```
............
............
..........*.
............
............
............
....##@.....
............
............
............
............
............
```

The split is the whole point of this demo. stuntd trains a head on a **frozen** encoder, and a
frozen encoder reads no geometry out of an ASCII map: asked about the map, the head learns to keep
going straight and the snake walks into the wall six moves in. Asked about the four lines above --
the food's offset from the head, the moves that survive the next step, the moves a wall or the body
takes away -- the same head reaches a 0.955 holdout agreement and plays.

## Playing it

Start from `stuntd config init` and leave `upstream` commented out: with no OpenAI upstream the
daemon serves the Jev routes alone and says so, `no upstream: only the Jev routes are served`.

With the Jev API as the teacher, `stuntd.toml` carrying `[jev] upstream = "https://api.typesafe.ai"`
and `TYPESAFE_API_KEY` exported, because the key is what `play.py` sends and what stuntd forwards
to the provider for every move the head does not answer:

```
export TYPESAFE_API_KEY=...
stuntd serve
python play.py --teacher jev --games 30 --seed 1
stuntd train && stuntd report && stuntd enable direction
python play.py --teacher jev --games 30 --seed 1
```

The provider answers every move of the third command and stuntd keeps each answer as a capture;
after `enable`, the last command asks the same questions and the local head answers the ones it is
sure about, the provider the rest.

With the BFS oracle as the teacher, which needs no key and no provider (`[jev] upstream` empty too,
so the base Laya checkpoint answers what the head cannot):

```
stuntd serve
python play.py --teacher none --games 30 --seed 1                       # before
python play.py --teacher oracle --games 30 --seed 1 --record moves.jsonl
                                                                        # stop the daemon: one
                                                                        # CUDA job at a time
stuntd import direction moves.jsonl --labels up,down,left,right
stuntd train && stuntd report && stuntd enable direction
stuntd serve
python play.py --teacher none --games 30 --seed 1                       # after
```

`--teacher oracle` plays the shortest safe path itself and writes `{"text": ..., "answer": ...}`
rows, so it needs no daemon at all; `--teacher jev` and `--teacher none` both send every move to
`POST /v1/systemone` and play what comes back. `--api-key` defaults to `TYPESAFE_API_KEY`, and a
local daemon does not look at it.

What the demo changes in the settings: `training.base_model` points at a local Laya checkpoint,
`training.epochs = 24`, and `training.target_agreement` chooses which of the two rows below you
get -- the default 0.99, or 0.95 for the head that answers every move.

Without a provider behind it stuntd compares nothing: the zero-shot answers come from the very
checkpoint the head was distilled from, so no request is sampled for a check, no comparison is
recorded, and a live site is never demoted for disagreeing with it -- `stuntd status --json`
reports `"agreement": null` and `stuntd report`, which measures the head against the teacher's own
held-out rows, is the quality signal here. Put `[jev] upstream` in front of it and both come back:
the sampled checks are answered by the provider and a head that drifts is demoted on its own.

## What it measured

The oracle variant, 30 games from seed 1, 200 moves at most per game, on a laptop RTX 5060 with
`convaiinnovations/laya` as the base checkpoint. **The teacher here is the BFS oracle, not an
LLM**: no Jev API key was available when these numbers were taken, so the "before" row is the base
Laya checkpoint answering zero-shot. The same table with the real Jev API as the teacher is still
to be filled in.

| run | average score | average moves | p50 per move | answered by the head |
| --- | --- | --- | --- | --- |
| base Laya, zero-shot (before) | 0.70 | 13.3 | 25.8 ms | 0/398 |
| BFS oracle (the teacher) | 21.37 | 196.3 | 0.1 ms | -- |
| head at `target_agreement = 0.99` (default) | 3.03 | 31.8 | 20.4 ms | 680/955 (71.2%) |
| head at `target_agreement = 0.95` | 11.40 | 101.7 | 21.1 ms | 3037/3050 (99.6%) |

Both head rows come from the same 5890 rows at the same 24 epochs, retrained once per target, and
both report the same holdout. The default 0.99 is the safe setting: `at threshold 0.75: coverage
0.82, agreement 0.990`, so the head answers the 82% of states it is sure about and hands the rest
back -- to zero-shot Laya here, which plays Snake badly and ends the game, but in proxy mode to the
Jev provider that taught it, which does not. The 0.95 setting is "let the head play": `at threshold
0.14: coverage 1.00, agreement 0.955`, so all but 13 of 3050 moves were answered on the machine,
and the snake eats 11.4 dots a game instead of 0.7. A game tolerates one wrong move in twenty; a
refund does not, and that is the whole of the difference between the two rows.

`stuntd import` wrote 5890 rows; `stuntd train` fitted the head on 4641 of them, held 1161 back and
took **186 seconds at `epochs = 24` with the encoder cache** for `holdout agreement 0.955
ece 0.075  temperature 1.58`. The frozen encoder runs once over the rows and every epoch after
that reads its cached output, which is why 24 epochs cost three minutes. `epochs = 24` is what the
0.95 row needs: at 12 epochs the head reaches 0.949, just under the target, so its threshold cannot
fall far enough and it goes back to abstaining. `stuntd status --json` after the 0.95 run:
`{"site": "direction", "mode": "live", "captures": 5890, "shadow": 0, "live": 3037,
"agreement": null}` -- 3037 local decisions, no comparison, and not one call to anything else.

The ceiling is worth knowing: a greedy reading of those four lines, always stepping toward the food
among the safe moves, matches the BFS oracle on 97.4% of moves and scores 17.2. The head reaches
11.4 of that 17.2 with a 0.955 agreement, so what is left is the last few points of agreement, not
the state it is asked about. Latency barely moves, because both the head and the zero-shot answer
run the same local encoder; against the paid Jev API the saving would be the round trip that no
longer happens.

One footnote on the table. Thirty games end on one bad move each, so the scores are a noisy
sample: an earlier run of the same protocol read 12.37 at the 0.95 target and 2.03 at 0.99, on a
head whose holdout was 0.957 rather than 0.955. The 12-epoch figure comes from that same earlier
tree. The head moves a little on a re-run as well: the same rows retrained at three seeds in the
devtools demo scored 0.985, 0.978 and 0.973 on its holdout, at coverage 0.98, 0.98 and 0.95.
