from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["Layout", "choose_layout", "question_for"]

_OPTION_TOKENS = 48
"""Tokens laya keeps of one option's text, after its marker."""

_HEAD_ROOM = 16
"""Tokens laya keeps free for the instruction before it starts cutting options."""


@dataclass(frozen=True)
class Layout:
    """How one site's input is laid out for the head: sequence and option budget, label text."""

    max_len: int
    head_max_len: int
    spaced_labels: bool


def question_for(field: str, labels: Sequence[str], spaced_labels: bool = False) -> dict[str, Any]:
    """The laya choice question for one site's field, asked the same way in training and serving."""
    options = [label.replace("_", " ") for label in labels] if spaced_labels else labels
    return {"t": "choice", "ins": f"Choose {field}", "crit": dict.fromkeys(options)}


def choose_layout(
    field: str,
    labels: Sequence[str],
    count_tokens: Callable[[str], int],
    checkpoint: Layout,
    max_option_tokens: int,
    max_positions: int,
) -> Layout:
    """The checkpoint's layout, widened so every label fits whole, within max_option_tokens and the encoder's max_positions."""
    spaced_labels = len({label.replace("_", " ") for label in labels}) == len(labels)
    question = question_for(field, labels, spaced_labels)
    instruction = count_tokens(f"{question['t']} question: {question['ins']}")
    options = sum(1 + min(_OPTION_TOKENS, count_tokens(" " + text)) for text in question["crit"])
    needed = options + max(_HEAD_ROOM, instruction)
    encoder_cap = checkpoint.head_max_len + max_positions - checkpoint.max_len
    head_max_len = max(checkpoint.head_max_len, min(needed, max_option_tokens, encoder_cap))
    grown = head_max_len - checkpoint.head_max_len
    return Layout(checkpoint.max_len + grown, head_max_len, spaced_labels)
