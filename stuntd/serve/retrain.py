from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

from anyio import to_thread

from stuntd.paths import data_dir, private_dir, write_private
from stuntd.serve.runtime import Runtime
from stuntd.settings import Settings
from stuntd.store.db import Store

__all__ = ["Retrainer", "last_result"]

_LOGS = "logs"

_log = logging.getLogger(__name__)


def last_result(site: str) -> str | None:
    """How the latest automatic training run of a site ended, or None when it has had none."""
    try:
        return (data_dir() / _LOGS / f"train-{site}.result").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


class Retrainer:
    """Starts `stuntd train` for a site once enough new captures have arrived, one run at a time."""

    def __init__(
        self,
        settings: Settings,
        store: Store,
        runtime: Runtime,
        config: Path | None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._runtime = runtime
        self._command = [sys.executable, "-m", "stuntd", "train"]
        self._config = [] if config is None else ["--config", str(config)]
        self._pending: dict[str, int] = {}
        self._training: asyncio.Task[None] | None = None

    def __repr__(self) -> str:
        return f"Retrainer(after={self._settings.auto_retrain}, pending={self._pending})"

    def record(self, site: str) -> None:
        """Counts one capture of site and starts a training run once enough have arrived."""
        model = self._runtime.state(site).model
        if site in self._pending:
            self._pending[site] += 1
        else:
            since = 0.0 if model is None else model.trained_at
            self._pending[site] = self._store.captures_since(site, since)
        if self._training is not None or self._pending[site] < self._settings.auto_retrain:
            return
        if model is None and self._pending[site] < self._settings.min_examples:
            return
        self._training = asyncio.create_task(self._train(site))

    async def _train(self, site: str) -> None:
        logs = data_dir() / _LOGS
        log = logs / f"train-{site}.log"
        result = "ok"
        try:
            private_dir(logs)
            code = await to_thread.run_sync(self._execute, site, log)
            if code != 0:
                result = f"failed (exit {code})"
                _log.error("training %s exited with %s, see %s", site, code, log)
        except OSError:
            _log.exception("training %s could not start, see %s", site, log)
            result = "failed (not started)"
        finally:
            self._training = None
            self._pending[site] = 0
        write_private(logs / f"train-{site}.result", result)

    def _execute(self, site: str, log: Path) -> int:
        with open(log, "ab") as output:
            done = subprocess.run(
                [*self._command, site, *self._config], stdout=output, stderr=subprocess.STDOUT
            )
        return done.returncode
