"""A deterministic 12x12 Snake, the text a decision site sees, and a BFS oracle to learn from."""

from __future__ import annotations

import random
from collections import deque

__all__ = ["BOARD", "DIRECTIONS", "Game", "board_text", "oracle_move", "state_text"]

BOARD = 12
"""Side of the square board, in cells."""

DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
"""Every move a player may answer with, mapped to its step in board coordinates."""

_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
_START_LENGTH = 3
_START_DIRECTION = "right"

_Cell = tuple[int, int]


class Game:
    """One game of Snake, fully determined by its seed and the moves it is given."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        centre = BOARD // 2
        self.body: list[_Cell] = [(centre - step, centre) for step in range(_START_LENGTH)]
        self.direction = _START_DIRECTION
        self.alive = True
        self.score = 0
        self.food: _Cell | None = self._new_food()

    def __repr__(self) -> str:
        return f"Game(head={self.body[0]}, length={len(self.body)}, score={self.score})"

    @property
    def head(self) -> _Cell:
        """Where the snake's head stands now."""
        return self.body[0]

    def step(self, move: str) -> None:
        """Plays one move, keeping the current direction when the move reverses it."""
        if not self.alive:
            raise RuntimeError("the game is over")
        if move not in DIRECTIONS:
            raise ValueError(f"unknown move: {move!r}")
        if move != _OPPOSITE[self.direction]:
            self.direction = move
        head = _ahead(self.head, self.direction)
        if not _inside(head) or head in _blocked(self.body):
            self.alive = False
            return
        eats = head == self.food
        self.body.insert(0, head)
        if not eats:
            self.body.pop()
            return
        self.score += 1
        self.food = self._new_food()
        if self.food is None:
            self.alive = False

    def _new_food(self) -> _Cell | None:
        taken = set(self.body)
        free = [(x, y) for y in range(BOARD) for x in range(BOARD) if (x, y) not in taken]
        return self._rng.choice(free) if free else None


def state_text(game: Game) -> str:
    """The text the direction question is asked about: where the food lies from the head, which
    moves survive the next step, and which do not.

    The board itself is not in it. stuntd trains a head on a frozen encoder, and a frozen encoder
    reads no geometry out of a 12x12 ASCII map: given the map alone the head learns to keep going
    straight, given these four lines it learns to play. `board_text` draws the map for the eye.
    """
    head_x, head_y = game.head
    safe = _safe_moves(game)
    blocked = [
        f"{move} ({'wall' if not _inside(cell) else 'body'})"
        for move, cell in _candidates(game).items()
        if move not in safe
    ]
    food = "none" if game.food is None else f"({game.food[0]},{game.food[1]})"
    return "\n".join(
        [
            f"dir={game.direction} head=({head_x},{head_y}) food={food} length={len(game.body)}",
            f"food: {_food_offset(game)}",
            f"safe: {', '.join(safe) if safe else 'none'}",
            f"blocked: {', '.join(blocked) if blocked else 'none'}",
        ]
    )


def oracle_move(game: Game) -> str:
    """The first step of a shortest safe path to the food, any safe move when the food is cut
    off, and straight on when nothing is safe."""
    safe = _safe_moves(game)
    if not safe:
        return game.direction
    if game.food is None:
        return next(iter(safe))
    distances = _distances_to(game.food, _blocked(game.body))
    reachable = {move: distances[cell] for move, cell in safe.items() if cell in distances}
    if not reachable:
        return next(iter(safe))
    return min(reachable, key=reachable.__getitem__)


def board_text(game: Game) -> str:
    """The board drawn for a reader: `.` empty, `#` body, `@` head, `*` food."""
    cells = [["."] * BOARD for _ in range(BOARD)]
    for x, y in game.body[1:]:
        cells[y][x] = "#"
    if game.food is not None:
        cells[game.food[1]][game.food[0]] = "*"
    cells[game.head[1]][game.head[0]] = "@"
    return "\n".join("".join(row) for row in cells)


def _food_offset(game: Game) -> str:
    if game.food is None:
        return "none"
    steps_x = game.food[0] - game.head[0]
    steps_y = game.food[1] - game.head[1]
    offsets = []
    if steps_x:
        offsets.append(f"{abs(steps_x)} {'right' if steps_x > 0 else 'left'}")
    if steps_y:
        offsets.append(f"{abs(steps_y)} {'down' if steps_y > 0 else 'up'}")
    return ", ".join(offsets)


def _candidates(game: Game) -> dict[str, _Cell]:
    return {
        move: _ahead(game.head, move) for move in DIRECTIONS if move != _OPPOSITE[game.direction]
    }


def _safe_moves(game: Game) -> dict[str, _Cell]:
    blocked = _blocked(game.body)
    return {
        move: cell
        for move, cell in _candidates(game).items()
        if _inside(cell) and cell not in blocked
    }


def _blocked(body: list[_Cell]) -> set[_Cell]:
    # The tail cell empties as the snake moves into it, and food never lies on the body, so the
    # one move that keeps the tail in place can never end on it.
    return set(body[:-1])


def _ahead(cell: _Cell, direction: str) -> _Cell:
    step_x, step_y = DIRECTIONS[direction]
    return cell[0] + step_x, cell[1] + step_y


def _inside(cell: _Cell) -> bool:
    return 0 <= cell[0] < BOARD and 0 <= cell[1] < BOARD


def _distances_to(target: _Cell, blocked: set[_Cell]) -> dict[_Cell, int]:
    distances = {target: 0}
    queue = deque([target])
    while queue:
        cell = queue.popleft()
        for direction in DIRECTIONS:
            neighbour = _ahead(cell, direction)
            if neighbour in distances or not _inside(neighbour) or neighbour in blocked:
                continue
            distances[neighbour] = distances[cell] + 1
            queue.append(neighbour)
    return distances
