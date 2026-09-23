import json

import httpx
import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from stuntd.cli import main
from stuntd.proxy.app import build_app
from stuntd.settings import Settings

DATABASE = "captures.sqlite"
CONFIG = 'upstream = "http://upstream"\nlearn = false\n'

TYPED_REQUEST = {
    "model": "gpt-x",
    "messages": [{"role": "user", "content": "buy pills"}],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "mod",
            "schema": {
                "type": "object",
                "properties": {"verdict": {"type": "string", "enum": ["allow", "block"]}},
            },
        },
    },
}

COMPLETION = {
    "id": "upstream",
    "model": "gpt-x",
    "choices": [{"message": {"role": "assistant", "content": '{"verdict": "block"}'}}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 2},
}

STREAM_CHUNKS = (b'data: {"i": 0}\n\n', b"data: [DONE]\n\n")

STATE = "Hi, I was double charged."
BILLING = {"type": "noul"}
NOUL_ANSWER = {"type": "noul", "noul": 0.8}
INPUT_TOKENS = 11

JEV_UPSTREAM = "http://jev-provider"
SYSTEMONE_PATH = "/v1/systemone"
MODELS_PATH = "/v1/models"
JEV_BODY = json.dumps(
    {
        "model": "jev-1",
        "answers": {"billing": NOUL_ANSWER},
        "usage": {"input_tokens": 120, "output_tokens": 12},
    }
).encode()
MODELS_BODY = json.dumps(
    {
        "models": [
            {"name": "jev-1", "description": "The provider's own.", "release_date": "2026-09-15"}
        ]
    }
).encode()


class FakeDecider:
    def decide(self, model, head_path, text):
        raise AssertionError("no head answers while learning is off")

    def answer(self, state, questions):
        return {
            "answers": dict.fromkeys(questions, NOUL_ANSWER),
            "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": 0},
        }

    def warm_base(self):
        pass


class Provider:
    """Fake OpenAI and Jev provider that counts what reached it."""

    def __init__(self):
        self.calls = 0

    async def handle(self, request):
        self.calls += 1
        if request.url.path == MODELS_PATH:
            return Response(content=MODELS_BODY, media_type="application/json")
        if request.url.path == SYSTEMONE_PATH:
            return Response(content=JEV_BODY, media_type="application/json")
        payload = json.loads(await request.body())
        if payload.get("stream") is True:

            async def events():
                for chunk in STREAM_CHUNKS:
                    yield chunk

            return StreamingResponse(events(), media_type="text/event-stream")
        return Response(content=json.dumps(COMPLETION).encode(), media_type="application/json")


@pytest.fixture
async def proxy(data_dir):
    apps = []

    def make(decider=None, provider=None, **overrides):
        handle = (provider or Provider()).handle
        upstream = Starlette(routes=[Route("/{path:path}", handle, methods=["GET", "POST"])])
        app = build_app(
            Settings(upstream="http://upstream", learn=False, **overrides),
            transport=httpx.ASGITransport(app=upstream),
            decider=decider,
        )
        apps.append(app)
        return app

    yield make
    for app in apps:
        await app.state.proxy.aclose()


async def post(app, path, payload):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        return await client.post(path, json=payload)


async def get(app, path):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        return await client.get(path)


def write_config(data_dir, body=CONFIG):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "stuntd.toml").write_text(body, encoding="utf-8")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "payload",
    [
        TYPED_REQUEST,
        {**TYPED_REQUEST, "stream": True},
        {"model": "gpt-x", "messages": [{"role": "user", "content": "buy pills"}]},
    ],
    ids=["typed", "streaming", "untyped"],
)
async def test_an_openai_request_is_passed_through(proxy, payload):
    response = await post(proxy(), "/v1/chat/completions", payload)
    assert response.status_code == 200
    assert response.headers["x-stuntd"] == "passthrough; reason=learning-off"


@pytest.mark.anyio
async def test_a_typed_request_keeps_the_provider_answer(proxy):
    response = await post(proxy(), "/v1/chat/completions", TYPED_REQUEST)
    assert json.loads(response.content) == COMPLETION


@pytest.mark.anyio
async def test_a_streaming_request_keeps_the_provider_chunks(proxy):
    response = await post(proxy(), "/v1/chat/completions", {**TYPED_REQUEST, "stream": True})
    assert response.content == b"".join(STREAM_CHUNKS)


@pytest.mark.anyio
async def test_a_jev_question_is_answered_zero_shot(proxy):
    app = proxy(FakeDecider())
    payload = {"state": STATE, "model": "jev-latest", "questions": {"billing": BILLING}}
    response = await post(app, "/v1/systemone", payload)
    assert response.status_code == 200
    assert json.loads(response.content) == {
        "model": "stuntd",
        "answers": {"billing": NOUL_ANSWER},
        "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": 0},
    }
    assert response.headers["x-stuntd"] == (
        "jev; mode=local; questions=1; live=0; zeroshot=1; learn=off"
    )


@pytest.mark.anyio
async def test_a_proxied_jev_request_is_relayed_whole(proxy, data_dir):
    provider = Provider()
    app = proxy(provider=provider, jev_upstream=JEV_UPSTREAM)
    payload = {"state": STATE, "model": "jev-latest", "questions": {"billing": BILLING}}
    response = await post(app, SYSTEMONE_PATH, payload)
    assert provider.calls == 1
    assert response.content == JEV_BODY
    assert response.headers["x-stuntd"] == (
        "jev; mode=proxy; questions=1; live=0; shadow=0; check=0; learn=off"
    )
    assert not (data_dir / DATABASE).exists()


@pytest.mark.anyio
async def test_the_proxied_model_list_is_relayed(proxy):
    provider = Provider()
    app = proxy(provider=provider, jev_upstream=JEV_UPSTREAM)
    response = await get(app, MODELS_PATH)
    assert provider.calls == 1
    assert response.content == MODELS_BODY
    assert response.headers["x-stuntd"] == "jev; mode=proxy"


@pytest.mark.anyio
async def test_neither_path_opens_a_database(proxy, data_dir):
    app = proxy(FakeDecider())
    await post(app, "/v1/chat/completions", TYPED_REQUEST)
    await post(app, "/v1/systemone", {"state": STATE, "questions": {"billing": BILLING}})
    assert app.state.store is None
    assert app.state.runtime is None
    assert not (data_dir / DATABASE).exists()


@pytest.mark.parametrize(
    "command",
    [
        ["train"],
        ["report"],
        ["enable", "spam"],
        ["disable", "spam"],
        ["import", "spam", "rows.jsonl"],
    ],
    ids=["train", "report", "enable", "disable", "import"],
)
def test_a_command_that_would_learn_is_refused(data_dir, capsys, command):
    write_config(data_dir)
    assert main(command) == 1
    assert f"stuntd: learning is off in {data_dir / 'stuntd.toml'}" in capsys.readouterr().err
    assert not (data_dir / DATABASE).exists()


def test_status_reports_learning_off(data_dir, capsys):
    write_config(data_dir)
    assert main(["status"]) == 0
    assert capsys.readouterr().out == "learning off\n"


def test_status_json_reports_learning_off(data_dir, capsys):
    write_config(data_dir)
    assert main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"learning": False, "sites": []}
    assert not (data_dir / DATABASE).exists()


def test_config_show_reports_learn(data_dir, capsys):
    write_config(data_dir)
    assert main(["config", "show"]) == 0
    assert "learn = False  (file)" in capsys.readouterr().out


def test_serve_reports_learning_off_on_its_third_line(data_dir, capsys, monkeypatch):
    write_config(data_dir)
    recorded = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: recorded.update(app=app))
    monkeypatch.setattr("stuntd.cli._make_decider", lambda settings: FakeDecider())
    assert main(["serve"]) == 0
    assert recorded["app"].state.store is None
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("stuntd listening on http://127.0.0.1:")
    assert lines[1] == "serving jev locally with convaiinnovations/laya"
    assert lines[2] == "learning off"
