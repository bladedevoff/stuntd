import importlib.util
import re
import sys
from pathlib import Path

import pytest

_DEMO = Path(__file__).resolve().parents[1] / "examples" / "snake"
_SOURCE = _DEMO / "game.py"


def _load_game():
    spec = importlib.util.spec_from_file_location("snake_game", _SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_client(monkeypatch):
    pytest.importorskip("typesafe_sdk")
    monkeypatch.setitem(sys.modules, "game", snake)
    spec = importlib.util.spec_from_file_location("snake_play", _DEMO / "play.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


snake = _load_game()


@pytest.mark.parametrize(
    "argv", [["--games", "0"], ["--max-moves", "0"]], ids=["games", "max-moves"]
)
def test_the_client_refuses_a_run_with_nothing_to_play(argv, capsys, monkeypatch):
    assert _load_client(monkeypatch).main(argv) == 2
    assert "must be at least 1" in capsys.readouterr().err


def test_the_client_refuses_a_negative_delay(capsys, monkeypatch):
    assert _load_client(monkeypatch).main(["--delay", "-1"]) == 2
    assert "cannot be negative" in capsys.readouterr().err


def test_show_draws_the_board_and_leaves_the_recorded_moves_alone(capsys, monkeypatch, tmp_path):
    client = _load_client(monkeypatch)
    quiet = tmp_path / "quiet.jsonl"
    loud = tmp_path / "loud.jsonl"
    argv = ["--teacher", "oracle", "--games", "1", "--max-moves", "5", "--record"]
    assert client.main([*argv, str(quiet)]) == 0
    capsys.readouterr()
    assert client.main([*argv, str(loud), "--show", "--delay", "0"]) == 0
    shown = capsys.readouterr().out
    assert loud.read_text(encoding="utf-8") == quiet.read_text(encoding="utf-8")
    assert "\x1b[2J\x1b[H" in shown
    assert re.search(r"game 1/1 · move 5 · score \d+ · \d+ ms · teacher", shown)


def test_a_move_advances_the_head_and_keeps_the_length():
    game = snake.Game(1)
    game.step("up")
    assert game.head == (6, 5)
    assert game.direction == "up"
    assert len(game.body) == 3
    assert snake.state_text(game).splitlines()[0] == "dir=up head=(6,5) food=(10,2) length=3"


def test_the_state_names_the_food_offset_and_what_each_move_hits():
    game = snake.Game(1)
    game.body = [(0, 5), (0, 4), (1, 4)]
    game.direction = "left"
    game.food = (3, 2)
    lines = snake.state_text(game).splitlines()
    assert lines[1] == "food: 3 right, 3 up"
    assert lines[2] == "safe: down"
    assert lines[3] == "blocked: up (body), left (wall)"
    assert len(lines) == 4
    assert snake.board_text(game).splitlines()[5] == "@..........."


def test_the_board_draws_the_position_the_state_describes():
    game = snake.Game(4)
    for _ in range(12):
        game.step(snake.oracle_move(game))
    header = re.fullmatch(
        r"dir=\w+ head=\((\d+),(\d+)\) food=\((\d+),(\d+)\) length=(\d+)",
        snake.state_text(game).splitlines()[0],
    )
    assert header is not None
    head_x, head_y, food_x, food_y, length = (int(value) for value in header.groups())
    rows = snake.board_text(game).splitlines()
    assert rows[head_y][head_x] == "@"
    assert rows[food_y][food_x] == "*"
    assert sum(row.count("#") for row in rows) == length - 1


def test_a_reversal_keeps_the_current_direction():
    game = snake.Game(1)
    game.step("left")
    assert game.direction == "right"
    assert game.head == (7, 6)


def test_food_grows_the_snake_and_scores():
    game = snake.Game(1)
    game.food = (7, 6)
    game.step("right")
    assert game.score == 1
    assert len(game.body) == 4
    assert game.food not in game.body


@pytest.mark.parametrize(
    "body, direction",
    [
        ([(11, 6), (10, 6), (9, 6)], "right"),
        ([(6, 6), (5, 6), (5, 5), (6, 5), (7, 5)], "up"),
    ],
    ids=["wall", "self"],
)
def test_the_snake_dies(body, direction):
    game = snake.Game(1)
    game.body = body
    game.direction = direction
    game.food = (0, 0)
    game.step(direction)
    assert not game.alive
    with pytest.raises(RuntimeError):
        game.step(direction)


def test_the_same_seed_replays_the_same_game():
    def played(seed):
        game = snake.Game(seed)
        states = []
        while game.alive and len(states) < 60:
            states.append(snake.state_text(game))
            game.step(snake.oracle_move(game))
        return states

    assert played(7) == played(7)
    assert played(7) != played(8)


def test_the_oracle_reaches_the_food_on_an_open_board():
    game = snake.Game(3)
    for _ in range(40):
        if game.score:
            break
        game.step(snake.oracle_move(game))
    assert game.score == 1
    assert game.alive
