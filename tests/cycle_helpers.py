import json

import httpx
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route

from stuntd.proxy.app import build_app
from stuntd.settings import config_path, load_settings

SITE = "refund"
UPSTREAM = "http://upstream"
WANTS_REFUND = "I was charged twice for invoice {n}, please send the money back."
ASKS_WHERE = "Where do I find the invoice {n} for last month?"
COLLECTED = 40


class Provider:
    def __init__(self):
        self.calls = 0

    async def handle(self, request):
        self.calls += 1
        payload = json.loads(await request.body())
        refund = "charged twice" in payload["messages"][-1]["content"]
        body = json.dumps(
            {
                "id": "upstream",
                "model": "gpt-x",
                "choices": [
                    {"message": {"role": "assistant", "content": json.dumps({"refund": refund})}}
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 2},
            }
        ).encode()
        return Response(content=body, media_type="application/json")


def message(index):
    return (WANTS_REFUND if index % 2 else ASKS_WHERE).format(n=1000 + index)


def request_for(text):
    return {
        "model": "gpt-x",
        "messages": [
            {"role": "system", "content": "Decide refunds."},
            {"role": "user", "content": text},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "refund",
                "schema": {"type": "object", "properties": {"refund": {"type": "boolean"}}},
            },
        },
    }


def app_for(provider, decider=None):
    upstream = Starlette(routes=[Route("/{path:path}", provider.handle, methods=["POST"])])
    return build_app(
        load_settings(config_path()),
        transport=httpx.ASGITransport(app=upstream),
        decider=decider,
    )


async def post(app, text):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        return await client.post(
            "/v1/chat/completions", json=request_for(text), headers={"X-Stuntd-Site": SITE}
        )
