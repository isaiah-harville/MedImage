"""The demo inference service: ``opendiagnostics-serve``.

It serves whatever ``catalog.json`` in the weights bucket names. At startup, and then every
``REFRESH_SECONDS``, it downloads new runs into ``WEIGHTS_CACHE`` and loads them, so a newly
published model appears without a new image or a restart. It starts and answers with no
models when nothing is published yet.

Configuration comes from the environment, read and checked once at startup:

| Variable | Default | Meaning |
|---|---|---|
| ``WEIGHTS_CACHE`` | required | writable folder for downloaded runs and checkpoints |
| ``WEIGHTS_BASE_URL`` | the public weights URL | where the bucket is read from |
| ``REFRESH_SECONDS`` | ``600`` | how often to check the catalog; ``0`` checks only at startup |
| ``LISTEN_ADDRESS`` | ``:8080`` | ``host:port`` to listen on |
| ``ATTRIBUTION_STEPS`` | ``16`` | Integrated Gradients steps per explanation |
| ``MAX_CONCURRENT`` | ``2`` | predictions that may run at once |
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl, ValidationError

from opendiagnostics.publish import WEIGHTS_URL

log = logging.getLogger("opendiagnostics.serve")


class Settings(BaseModel):
    weights_cache: Path
    weights_base_url: HttpUrl = HttpUrl(WEIGHTS_URL)
    refresh_seconds: int = Field(600, ge=0)
    listen_address: str = ":8080"
    attribution_steps: int = Field(16, ge=1, le=256)
    max_concurrent: int = Field(2, ge=1, le=64)

    @property
    def host_port(self) -> tuple[str, int]:
        host, _, port = self.listen_address.rpartition(":")
        return host or "0.0.0.0", int(port)


def settings_from(environ: dict[str, str]) -> Settings:
    values = {n: environ[n.upper()] for n in Settings.model_fields if n.upper() in environ}
    try:
        return Settings.model_validate(values)
    except ValidationError as error:
        problems = "; ".join(f"{e['loc'][0]}: {e['msg']}" for e in error.errors())
        raise SystemExit(f"invalid configuration: {problems}. WEIGHTS_CACHE is required.") from None


def main() -> None:
    import uvicorn

    from opendiagnostics.serve.app import create_app
    from opendiagnostics.serve.models import refresh
    from opendiagnostics.serve.weights import prune, sync

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = settings_from(dict(os.environ))
    cache, base_url = settings.weights_cache, str(settings.weights_base_url)
    served = [{}]

    def update() -> None:
        folders = sync(cache, base_url)
        served[0] = refresh(served[0], folders, cache)
        prune(cache, folders)
        log.info("serving %d models", len(served[0]))

    def safely_update() -> None:
        try:
            update()
        except Exception:
            log.exception("refreshing the models failed; serving the previous set")

    def keep_updating() -> None:
        while not stop.wait(settings.refresh_seconds):
            safely_update()

    safely_update()

    stop = threading.Event()
    if settings.refresh_seconds:
        threading.Thread(target=keep_updating, name="refresh", daemon=True).start()
    app = create_app(lambda: served[0], settings.attribution_steps, settings.max_concurrent)
    host, port = settings.host_port
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)
    stop.set()
