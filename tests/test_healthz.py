import httpx
import pytest
from test_modes import model

from stuntd.proxy.app import build_app
from stuntd.serve.modes import MODE_LIVE, write_mode
from stuntd.settings import Settings
from stuntd.train.artifacts import save_model

pytestmark = pytest.mark.anyio


class UnusedDecider:
    def decide(self, model, head_path, text):
        raise AssertionError("healthz decided something")

    def warm(self, model, head_path):
        raise AssertionError("healthz loaded a checkpoint")

    def warm_base(self):
        raise AssertionError("healthz loaded the base checkpoint")


async def healthz(settings):
    app = build_app(settings, decider=UnusedDecider())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        response = await client.get("/healthz")
    await app.state.client.aclose()
    if app.state.store is not None:
        app.state.store.close()
    return response


async def test_healthz_counts_the_sites_per_mode(data_dir):
    models = data_dir / "models"
    for site in ("a", "b"):
        save_model(models, model(site))
    write_mode(models / "a", MODE_LIVE, 1.0)
    response = await healthz(Settings())
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "learning": True, "sites": {"live": 1, "shadow": 1}}


async def test_healthz_reports_no_sites_while_learning_is_off(data_dir):
    save_model(data_dir / "models", model("a"))
    response = await healthz(Settings(learn=False))
    assert response.json() == {"status": "ok", "learning": False, "sites": {}}
