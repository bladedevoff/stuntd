"""Draws the two charts README.md embeds, with the standard library alone.

`python docs/charts.py` rewrites both files; `--check` exits 1 when the committed ones differ from
what this script would write now.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["main"]

DOCS = Path(__file__).resolve().parent
"""Directory the two SVG files live in, beside this script."""

WIDTH = 720
"""Width of both charts in pixels, the width README.md renders them at."""

ACCENT = "#2F5BEA"
GREY = "#A3ADBF"
INK = "#14213D"
MUTED = "#6B7690"
GRID = "#E1E6EF"
WHITE = "#ffffff"


@dataclass(frozen=True)
class _Site:
    demo: str
    question: str
    zero_shot: float
    trained: float


@dataclass(frozen=True)
class _Run:
    label: str
    score: float
    trained: bool


SITES = [
    _Site("support", "category", 69.5, 99.5),
    _Site("support", "urgency", 34.5, 85.5),
    _Site("support", "needs_human", 66.5, 97.5),
    _Site("banking", "intent", 89.5, 100.0),
    _Site("banking", "risk", 30.0, 72.5),
    _Site("devtools", "command_gate", 45.0, 97.0),
    _Site("devtools", "flaky", 84.5, 100.0),
]
"""Accuracy against the teacher, in percent, from the table of each demo's README."""

SNAKE_RUNS = [
    _Run("base Laya, zero-shot", 0.70, False),
    _Run("head at target 0.99", 3.03, True),
    _Run("head at target 0.95", 11.40, True),
    _Run("BFS oracle", 21.37, False),
]
"""Average score over 30 games from seed 1, from examples/snake/README.md."""


def main(argv: list[str] | None = None) -> int:
    """Writes both charts, or checks the committed ones against a fresh render."""
    args = _parser().parse_args(argv)
    charts = {DOCS / "before-after.svg": _before_after(), DOCS / "snake.svg": _snake()}
    if args.check:
        stale = [path for path, svg in charts.items() if not _matches(path, svg)]
        for path in stale:
            print(f"{path.name} is out of date, run python docs/charts.py", file=sys.stderr)
        return 1 if stale else 0
    for path, svg in charts.items():
        path.write_text(svg, encoding="utf-8", newline="\n")
        print(f"wrote {path}")
    return 0


def _matches(path: Path, svg: str) -> bool:
    return path.exists() and path.read_text(encoding="utf-8") == svg


def _before_after() -> str:
    height = 384
    left, top, baseline = 52.0, 88.0, 316.0
    right_edge = WIDTH - 24
    body = [
        _text(left, 36, "Accuracy against the teacher, before and after training", 18, INK, "bold"),
        _text(
            left, 58, "one bar pair per decision site, 200 held-out requests per demo", 12, MUTED
        ),
    ]
    for index, (label, fill) in enumerate((("zero-shot", GREY), ("trained head", ACCENT))):
        swatch = right_edge - 210 + index * 110
        body.append(_rect(swatch, 40, 12, 12, fill))
        body.append(_text(swatch + 18, 50, label, 12, MUTED))
    for percent in (0, 25, 50, 75, 100):
        line_y = baseline - (baseline - top) * percent / 100
        body.append(_line(left, line_y, right_edge, line_y))
        body.append(_text(left - 10, line_y + 4, f"{percent}", 11, MUTED, anchor="end"))
    group = (right_edge - left) / len(SITES)
    bar = 28.0
    for index, site in enumerate(SITES):
        centre = left + group * (index + 0.5)
        pair = ((site.zero_shot, GREY), (site.trained, ACCENT))
        for offset, (value, fill) in enumerate(pair):
            x = centre - bar - 3 + offset * (bar + 6)
            bar_top = baseline - (baseline - top) * value / 100
            body.append(_rect(x, bar_top, bar, baseline - bar_top, fill))
            body.append(_text(x + bar / 2, bar_top - 6, f"{value:.1f}", 11, INK, anchor="middle"))
        body.append(_text(centre, baseline + 21, site.question, 12, INK, anchor="middle"))
        body.append(_text(centre, baseline + 37, site.demo, 11, MUTED, anchor="middle"))
    return _svg(height, body)


def _snake() -> str:
    top, row_height, bar_left, bar_max = 96.0, 46.0, 206.0, 438.0
    height = int(top + row_height * len(SNAKE_RUNS) + 26)
    scale = bar_max / max(run.score for run in SNAKE_RUNS)
    body = [
        _text(24, 36, "Snake: average score over 30 games", 18, INK, "bold"),
        _text(
            24,
            58,
            "12x12 board, 200 moves at most per game, one typed decision per move",
            12,
            MUTED,
        ),
        _line(bar_left, top - 12, bar_left, height - 14),
    ]
    for index, run in enumerate(SNAKE_RUNS):
        y = top + row_height * index
        width = run.score * scale
        body.append(_rect(bar_left, y, width, 26, ACCENT if run.trained else GREY))
        body.append(_text(bar_left - 14, y + 18, run.label, 13, INK, anchor="end"))
        score = f"{run.score:.2f}"
        if width >= 90:
            body.append(_text(bar_left + width - 10, y + 18, score, 13, WHITE, "bold", "end"))
        else:
            body.append(_text(bar_left + width + 10, y + 18, score, 13, INK, "bold"))
    return _svg(height, body)


def _svg(height: int, body: list[str]) -> str:
    opening = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}"'
        f' viewBox="0 0 {WIDTH} {height}" role="img">'
    )
    return "\n".join([opening, _rect(0, 0, WIDTH, height, WHITE), *body, "</svg>", ""])


def _rect(x: float, y: float, width: float, height: float, fill: str) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{fill}"/>'
    )


def _line(x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
        f' stroke="{GRID}" stroke-width="1"/>'
    )


def _text(
    x: float,
    y: float,
    value: str,
    size: int,
    fill: str,
    weight: str = "normal",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Bahnschrift, DIN Alternate, Segoe UI, sans-serif" font-size="{size}"'
        f' font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{value}</text>'
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="fail when a committed chart is out of date"
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
