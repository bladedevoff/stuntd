import importlib.util
import json
import sys
from pathlib import Path

import pytest

typesafe_sdk = pytest.importorskip("typesafe_sdk")
httpx2 = pytest.importorskip("httpx2")

TypeSafeClient = typesafe_sdk.TypeSafeClient
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load_demo(name):
    directory = EXAMPLES / name
    generate = _load(f"{name}_generate", directory / "generate.py")
    kept = sys.modules.get("generate")
    sys.modules["generate"] = generate
    try:
        client = _load(f"{name}_client", directory / "client.py")
    finally:
        if kept is None:
            del sys.modules["generate"]
        else:
            sys.modules["generate"] = kept
    return generate, client


def recording_transport(answers):
    sent = []
    header = f"jev; mode=local; questions={len(answers)}; live={len(answers)}; zeroshot=0"
    body = {"model": "stuntd", "answers": answers, "usage": {"input_tokens": 3, "output_tokens": 0}}

    def handle(request):
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=body, headers={"X-Stuntd": header})

    return sent, httpx2.MockTransport(handle)


def choice_answer(label, labels):
    others = (1.0 - 0.9) / max(1, len(labels) - 1)
    probabilities = {name: (0.9 if name == label else others) for name in labels}
    return {"type": "choice", "choice": label, "confidence": 0.9, "probabilities": probabilities}


def score_answer(level, levels):
    probabilities = {str(i): (0.9 if i == level else 0.1 / (levels - 1)) for i in range(levels)}
    legend = {str(i): f"level {i}" for i in range(levels)}
    return {
        "type": "score",
        "score": float(level),
        "confidence": 0.9,
        "legend": legend,
        "probabilities": probabilities,
    }


def noul_answer(value):
    return {"type": "noul", "noul": value}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
