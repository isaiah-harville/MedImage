"""The demo inference service: ``opendiagnostics-serve``.

Configuration comes from the environment, read and checked once at startup:

| Variable | Default | Meaning |
|---|---|---|
| ``WEIGHTS_DIR`` | required | folder of bundled models (``scripts/bundle_models.py``) |
| ``LISTEN_ADDRESS`` | ``:8080`` | ``host:port`` to listen on |
| ``ATTRIBUTION_STEPS`` | ``16`` | Integrated Gradients steps per explanation |
| ``MAX_CONCURRENT`` | ``2`` | predictions that may run at once |
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator


class Settings(BaseModel):
    weights_dir: Path
    listen_address: str = ":8080"
    attribution_steps: int = Field(16, ge=1, le=256)
    max_concurrent: int = Field(2, ge=1, le=64)

    @field_validator("weights_dir")
    @classmethod
    def _exists(cls, value: Path) -> Path:
        if not value.is_dir():
            raise ValueError(f"{value} is not a directory; run scripts/bundle_models.py first")
        return value

    @property
    def host_port(self) -> tuple[str, int]:
        host, _, port = self.listen_address.rpartition(":")
        return host or "0.0.0.0", int(port)


def settings_from(environ: dict[str, str]) -> Settings:
    names = ("weights_dir", "listen_address", "attribution_steps", "max_concurrent")
    values = {n: environ[n.upper()] for n in names if n.upper() in environ}
    try:
        return Settings.model_validate(values)
    except ValidationError as error:
        problems = "; ".join(f"{e['loc'][0]}: {e['msg']}" for e in error.errors())
        raise SystemExit(f"invalid configuration: {problems}. WEIGHTS_DIR is required.") from None


def main() -> None:
    import uvicorn

    from opendiagnostics.serve.app import create_app
    from opendiagnostics.serve.models import load_models

    settings = settings_from(dict(os.environ))
    app = create_app(
        load_models(settings.weights_dir), settings.attribution_steps, settings.max_concurrent
    )
    host, port = settings.host_port
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)
