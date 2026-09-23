"""Sends one typed decision through a running stuntd and prints the verdict it got back."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

PROXY_URL = "http://127.0.0.1:8787/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
VERDICTS = ["allow", "review", "block"]
COMMENT = "Buy cheap watches at this link, limited offer, click now."
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "moderation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"verdict": {"type": "string", "enum": VERDICTS}},
            "required": ["verdict"],
            "additionalProperties": False,
        },
    },
}


def request_payload(model: str, comment: str) -> dict[str, Any]:
    """The chat completion stuntd recognises as a decision with one enum field."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "Moderate the comment and answer with one verdict."},
            {"role": "user", "content": comment},
        ],
        "response_format": RESPONSE_FORMAT,
    }


def main() -> None:
    """Posts the decision to the proxy and prints the answer and the X-Stuntd header."""
    response = httpx.post(
        PROXY_URL,
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json=request_payload(os.environ.get("STUNTD_MODEL", DEFAULT_MODEL), COMMENT),
        timeout=60.0,
    )
    response.raise_for_status()
    answer = json.loads(response.json()["choices"][0]["message"]["content"])
    print(f"verdict: {answer['verdict']}")
    print(f"X-Stuntd: {response.headers.get('X-Stuntd', '')}")


if __name__ == "__main__":
    main()
