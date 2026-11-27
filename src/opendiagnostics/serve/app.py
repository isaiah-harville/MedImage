"""The inference API, served by LitServe. The contract is deploy/api.md.

One ``LitAPI`` at ``POST /predict`` runs every model, in a LitServe worker process that loads
them once. The API process answers the model listing and demo images from each run's files.

Uploads arrive as base64 inside a JSON body, which LitServe's size limit caps before it is
all read. They are decoded in memory, handed to the worker over an in-memory queue, and never
written to disk, stored, or logged. Every response carries ``Cache-Control: no-store``.
"""

# No `from __future__ import annotations` here: LitServe builds the /predict endpoint from
# decode_request's type hint at runtime, and FastAPI cannot resolve a string annotation.
import base64
import binascii
import io
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import litserve as ls
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, model_validator

from opendiagnostics.serve.models import (
    DemoImage,
    Listing,
    NotServableError,
    ServedModel,
    load_listing,
    load_models,
    predict,
)

log = logging.getLogger("opendiagnostics.serve")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Base64 grows the image by a third; the JSON around it adds a little.
MAX_BODY_BYTES = MAX_UPLOAD_BYTES * 4 // 3 + 64 * 1024
MAX_SIDE = 4096
FORMATS = {"PNG", "JPEG"}
Image.MAX_IMAGE_PIXELS = MAX_SIDE * MAX_SIDE


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    sample_id: str | None = None
    image: str | None = None

    @model_validator(mode="after")
    def _one_input(self) -> Self:
        if (self.sample_id is None) == (self.image is None):
            raise ValueError("send exactly one of sample_id (a demo image) or image (base64)")
        return self


def describe(folders: list[Path]) -> dict[str, Listing]:
    """Each model's listing, from its run files. The API process loads no weights: only the
    worker keeps models in memory."""
    listings = []
    for folder in folders:
        try:
            listings.append(load_listing(folder))
        except NotServableError as reason:
            log.info("not serving %s: %s", folder.parent.name, reason)
        except Exception:
            log.exception("could not list %s/%s", folder.parent.name, folder.name)
    return {listing.description["id"]: listing for listing in listings}


def _decode(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
    except UnidentifiedImageError as error:
        raise HTTPException(415, "upload a PNG or JPEG image") from error
    if image.format not in FORMATS:
        raise HTTPException(415, f"upload a PNG or JPEG image, not {image.format}")
    if max(image.size) > MAX_SIDE:
        raise HTTPException(413, f"images may be at most {MAX_SIDE} x {MAX_SIDE} pixels")
    try:
        image.load()
    except (OSError, Image.DecompressionBombError) as error:
        raise HTTPException(422, "the file could not be decoded as an image") from error
    return image


def _upload(encoded: str) -> Image.Image:
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(422, "image must be base64") from error
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "files may be at most 10 MB")
    return _decode(data)


class ClassifierAPI(ls.LitAPI):
    """Every published classifier behind one endpoint; the request names the model."""

    def __init__(self, folders: list[Path], cache: Path, attribution_steps: int) -> None:
        super().__init__(api_path="/predict")
        self.folders, self.cache, self.attribution_steps = folders, cache, attribution_steps
        self.models: dict[str, ServedModel] = {}

    def setup(self, device: str) -> None:
        self.models = load_models(self.folders, self.cache)

    def decode_request(
        self, request: PredictRequest, **kwargs: Any
    ) -> tuple[ServedModel, Image.Image, str | None]:
        model = self.models.get(request.model)
        if model is None:
            raise HTTPException(404, f"no model {request.model!r}")
        if request.image is not None:
            return model, _upload(request.image), None
        demo = model.demos.get(request.sample_id or "")
        if demo is None:
            raise HTTPException(404, f"no demo image {request.sample_id!r} for {model.id}")
        with Image.open(demo.path) as image:
            image.load()
            return model, image.copy(), demo.truth

    def predict(
        self, x: tuple[ServedModel, Image.Image, str | None], **kwargs: Any
    ) -> dict[str, Any]:
        model, image, truth = x
        return predict(model, image, truth, self.attribution_steps)

    def encode_response(self, output: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return output


def create_server(
    listings: Mapping[str, Listing], api: ClassifierAPI, workers: int = 1
) -> ls.LitServer:
    server = ls.LitServer(
        api,
        accelerator="cpu",
        workers_per_device=workers,
        timeout=120,
        max_payload_size=MAX_BODY_BYTES,
        disable_openapi_url=True,
    )
    app = server.app

    def listing_for(model_id: str) -> Listing:
        listing = listings.get(model_id)
        if listing is None:
            raise HTTPException(404, f"no model {model_id!r}")
        return listing

    def demo_for(model_id: str, sample_id: str) -> DemoImage:
        demo = listing_for(model_id).demos.get(sample_id)
        if demo is None:
            raise HTTPException(404, f"no demo image {sample_id!r} for {model_id}")
        return demo

    @app.get("/models")
    def list_models() -> list[dict[str, Any]]:
        return [listing.description for listing in listings.values()]

    @app.get("/models/{model_id}/demo")
    def demos(model_id: str) -> list[dict[str, Any]]:
        return [
            {
                "sample_id": d.sample_id,
                "truth": d.truth,
                "image_url": f"/models/{model_id}/demo/{d.sample_id}.png",
            }
            for d in listing_for(model_id).demos.values()
        ]

    @app.get("/models/{model_id}/demo/{sample_id}.png")
    def demo_image(model_id: str, sample_id: str) -> FileResponse:
        return FileResponse(demo_for(model_id, sample_id).path, media_type="image/png")

    @app.middleware("http")
    async def no_store(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    return server
