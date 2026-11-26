"""The inference HTTP API. The contract is docs/demo-api.md.

Uploads are decoded in memory and never written to disk or logged. Predictions run in a
bounded number of worker threads so a burst of requests cannot exhaust memory.
"""

from __future__ import annotations

import io
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from opendiagnostics.serve.models import ServedModel, predict

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_SIDE = 4096
FORMATS = {"PNG", "JPEG"}
Image.MAX_IMAGE_PIXELS = MAX_SIDE * MAX_SIDE


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


def create_app(
    models: dict[str, ServedModel], attribution_steps: int = 16, max_concurrent: int = 2
) -> FastAPI:
    app = FastAPI(title="OpenDiagnostics demo API", docs_url=None, redoc_url=None)
    slots = threading.BoundedSemaphore(max_concurrent)

    def model_for(model_id: str) -> ServedModel:
        if model_id not in models:
            raise HTTPException(404, f"no model {model_id!r}")
        return models[model_id]

    def run(model: ServedModel, image: Image.Image, truth: str | None) -> dict[str, Any]:
        with slots:
            return predict(model, image, truth, attribution_steps)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "models": len(models)}

    @app.get("/models")
    def list_models() -> list[dict[str, Any]]:
        return [m.description for m in models.values()]

    @app.get("/models/{model_id}/demo")
    def demos(model_id: str) -> list[dict[str, Any]]:
        model = model_for(model_id)
        return [
            {
                "sample_id": d.sample_id,
                "truth": d.truth,
                "image_url": f"/models/{model_id}/demo/{d.sample_id}.png",
            }
            for d in model.demos.values()
        ]

    def demo_for(model: ServedModel, sample_id: str):
        if sample_id not in model.demos:
            raise HTTPException(404, f"no demo image {sample_id!r} for {model.id}")
        return model.demos[sample_id]

    @app.get("/models/{model_id}/demo/{sample_id}.png")
    def demo_image(model_id: str, sample_id: str) -> FileResponse:
        demo = demo_for(model_for(model_id), sample_id)
        return FileResponse(demo.path, media_type="image/png")

    @app.post("/models/{model_id}/demo/{sample_id}/predict")
    def predict_demo(model_id: str, sample_id: str) -> dict[str, Any]:
        model = model_for(model_id)
        demo = demo_for(model, sample_id)
        with Image.open(demo.path) as image:
            image.load()
            return run(model, image, demo.truth)

    @app.post("/models/{model_id}/predict")
    def predict_upload(model_id: str, file: UploadFile) -> dict[str, Any]:
        model = model_for(model_id)
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "files may be at most 10 MB")
        return run(model, _decode(data), None)

    return app
