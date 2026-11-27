"""The demo inference service: ``opendiagnostics-serve``, built on LitServe.

It serves whatever ``catalog.json`` in the weights bucket names. At startup it downloads the
cataloged runs into ``WEIGHTS_CACHE`` and starts a LitServe server with one worker that
holds every model. Every ``REFRESH_SECONDS`` it checks the catalog again; when it changes,
it downloads the new runs and exits so its supervisor (Kubernetes) restarts it on the new
set. It starts and answers with no models when nothing is published yet.

Configuration comes from the environment, read and checked once at startup:

| Variable | Default | Meaning |
|---|---|---|
| ``WEIGHTS_CACHE`` | required | writable folder for downloaded runs and checkpoints |
| ``WEIGHTS_BASE_URL`` | the public weights URL | where the bucket is read from |
| ``REFRESH_SECONDS`` | ``600`` | how often to check the catalog; ``0`` checks only at startup |
| ``LISTEN_ADDRESS`` | ``:8080`` | ``host:port`` to listen on |
| ``ATTRIBUTION_STEPS`` | ``16`` | Integrated Gradients steps per explanation |
| ``WORKERS`` | ``1`` | LitServe worker processes; each holds every model |
"""

from __future__ import annotations

import logging
import os
import signal
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
    workers: int = Field(1, ge=1, le=8)

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


def _synced(cache: Path, base_url: str) -> list[Path] | None:
    from opendiagnostics.serve.weights import sync

    try:
        return sync(cache, base_url)
    except Exception:
        log.exception("reading the weights bucket failed")
        return None


def watch(settings: Settings, served: list[Path], stop: threading.Event) -> None:
    """Exit the process once the catalog names a different set of runs."""
    cache, base_url = settings.weights_cache, str(settings.weights_base_url)
    while not stop.wait(settings.refresh_seconds):
        folders = _synced(cache, base_url)
        if folders is not None and set(folders) != set(served):
            log.info("the catalog changed; restarting to serve %d models", len(folders))
            os.kill(os.getpid(), signal.SIGTERM)
            return


def main() -> None:
    from opendiagnostics.serve.app import ClassifierAPI, create_server, describe
    from opendiagnostics.serve.weights import prune

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # LitServe stops its workers cleanly on KeyboardInterrupt; Kubernetes and watch() send TERM.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    settings = settings_from(dict(os.environ))
    cache = settings.weights_cache
    folders = _synced(cache, str(settings.weights_base_url)) or []
    prune(cache, folders)
    listings = describe(folders, cache)
    log.info("serving %d models", len(listings))
    api = ClassifierAPI(folders, cache, settings.attribution_steps)
    server = create_server(listings, api, settings.workers)
    stop = threading.Event()
    if settings.refresh_seconds:
        threading.Thread(target=watch, args=(settings, folders, stop), daemon=True).start()
    host, port = settings.host_port
    server.run(host=host, port=port, generate_client_file=False, access_log=False)
    stop.set()
