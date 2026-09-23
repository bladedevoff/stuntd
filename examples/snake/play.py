"""Plays Snake against a running stuntd, asking it for one direction per move.

`--teacher jev` and `--teacher none` send every move to `POST /v1/systemone`; the difference is
what the daemon has behind it, a paid Jev provider or the base Laya checkpoint alone.
`--teacher oracle` needs no daemon: it plays a BFS search itself and writes the moves as rows for
`stuntd import`.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from game import Game, board_text, oracle_move, state_text
from typesafe_sdk import Choice, SystemOneResponse, TypeSafeClient

DEFAULT_URL = "http://127.0.0.1:8787"
DEFAULT_KEY = "local"
QUESTION = "direction"
INSTRUCTIONS = (
    "You are the snake in a game of Snake on a 12x12 board. The state gives the direction you are"
    " travelling in, how many steps the food lies away from your head on each axis, the moves that"
    " are safe to take next and the moves that are blocked by a wall or by your body. Answer with"
    " the direction that takes the food without leaving the board or hitting your body."
)
CRITERIA: dict[str, None] = {"up": None, "down": None, "left": None, "right": None}
TIMEOUT = 60.0
_STUNTD_HEADER = "X-Stuntd"
_LIVE = "live="


@dataclass(frozen=True)
class GameResult:
    """What one game came to: the food eaten, the moves played and their latencies."""

    score: int
    moves: int
    latencies_ms: list[float]
    moves_from_a_head: int

    @property
    def p50_ms(self) -> float:
        """Median latency of one move, in milliseconds."""
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0


class _Oracle:
    """Plays the shortest safe path itself and records what it saw with what it answered."""

    def __init__(self, rows: TextIO | None) -> None:
        self._rows = rows

    def move(self, game: Game) -> tuple[str, float, bool]:
        text = state_text(game)
        started = time.perf_counter()
        move = oracle_move(game)
        latency_ms = (time.perf_counter() - started) * 1000
        if self._rows is not None:
            self._rows.write(json.dumps({"text": text, "answer": move}, ensure_ascii=False) + "\n")
        return move, latency_ms, False


class _Daemon:
    """Asks a running stuntd for every move over the Jev protocol."""

    def __init__(self, client: TypeSafeClient) -> None:
        self._client = client
        self._question = Choice(instructions=INSTRUCTIONS, criteria=CRITERIA)

    def move(self, game: Game) -> tuple[str, float, bool]:
        started = time.perf_counter()
        answer = self._client.system_one(
            state=state_text(game), questions={QUESTION: self._question}, timeout=TIMEOUT
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return answer.choices[QUESTION].choice, latency_ms, _served_by_a_head(answer)


def _served_by_a_head(answer: SystemOneResponse) -> bool:
    header = answer.raw_http_response.headers.get(_STUNTD_HEADER, "")
    for part in header.split(";"):
        part = part.strip()
        if part.startswith(_LIVE):
            return part[len(_LIVE) :] != "0"
    return False


def play(game: Game, player: _Oracle | _Daemon, max_moves: int) -> GameResult:
    """Plays one game to its death or to max_moves, whichever comes first."""
    latencies: list[float] = []
    heads = 0
    while game.alive and len(latencies) < max_moves:
        move, latency_ms, from_head = player.move(game)
        latencies.append(latency_ms)
        heads += int(from_head)
        game.step(move)
    return GameResult(game.score, len(latencies), latencies, heads)


def _print_game(index: int, result: GameResult) -> None:
    print(
        f"game {index:>3}  score {result.score:>3}  moves {result.moves:>4}"
        f"  p50 {result.p50_ms:7.1f} ms"
    )


def _print_totals(results: list[GameResult], teacher: str) -> None:
    latencies = [latency for result in results for latency in result.latencies_ms]
    moves = sum(result.moves for result in results)
    heads = sum(result.moves_from_a_head for result in results)
    print(
        f"\n{len(results)} games  average score {statistics.mean(r.score for r in results):.2f}"
        f"  average moves {moves / len(results):.1f}"
        f"  p50 {statistics.median(latencies):.1f} ms"
    )
    if teacher == "oracle":
        return
    print(f"answered by a trained head: {heads}/{moves} moves ({100 * heads / moves:.1f}%)")


def _play_all(args: argparse.Namespace, player: _Oracle | _Daemon) -> None:
    results = []
    for index in range(args.games):
        game = Game(args.seed + index)
        result = play(game, player, args.max_moves)
        _print_game(index + 1, result)
        results.append(result)
    print()
    print("the last game ended here:")
    print(board_text(game))
    _print_totals(results, args.teacher)


def main(argv: list[str] | None = None) -> int:
    """Runs the games the arguments ask for and prints the score and the latencies."""
    args = _parser().parse_args(argv)
    if args.games < 1 or args.max_moves < 1:
        print("--games and --max-moves must be at least 1", file=sys.stderr)
        return 2
    if args.teacher != "oracle":
        if args.record:
            print("--record only has moves to write with --teacher oracle", file=sys.stderr)
            return 2
        with TypeSafeClient(api_key=args.api_key, base_url=args.url, timeout=TIMEOUT) as client:
            _play_all(args, _Daemon(client))
        return 0
    if not args.record:
        _play_all(args, _Oracle(None))
        return 0
    with Path(args.record).open("w", encoding="utf-8") as rows:
        _play_all(args, _Oracle(rows))
    print(f"recorded the moves in {args.record}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=30, metavar="N", help="games to play")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="seed of the first game")
    parser.add_argument("--url", default=DEFAULT_URL, metavar="URL", help="the stuntd to ask")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TYPESAFE_API_KEY", DEFAULT_KEY),
        metavar="K",
        help="key sent with every request, TYPESAFE_API_KEY by default;"
        " a local daemon does not check it",
    )
    parser.add_argument(
        "--teacher",
        choices=("jev", "oracle", "none"),
        default="jev",
        help="who plays: the daemon in front of a Jev provider, a BFS search, or the daemon alone",
    )
    parser.add_argument(
        "--record", metavar="FILE", help="write the oracle's moves as JSONL for stuntd import"
    )
    parser.add_argument(
        "--max-moves", type=int, default=200, metavar="M", help="moves before a game is cut short"
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
