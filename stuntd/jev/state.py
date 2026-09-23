from __future__ import annotations

import json

__all__ = ["serialize_state"]


def serialize_state(state: object) -> str:
    """The text a Jev state becomes, byte for byte as laya.common.serialize_state writes it."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)
