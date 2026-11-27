"""The inference HTTP API. The contract is deploy/api.md.

Uploads stay in memory for the length of one request and are never written to disk, stored,
or logged: the body is read into memory with a size cap and parsed there, and Starlette's
multipart parser is set never to spool a part to a temporary file. Every response carries
``Cache-Control: no-store``. Predictions run in a bounded number of threads so a burst of
requests cannot exhaust memory.
"""

from __future__ import annotations

import io
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.formparsers import MultiPartParser

from opendiagnostics.serve.models import ServedModel, predict

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# The multipart envelope around the file adds a few hundred bytes.
MAX_BODY_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
# Keep every multipart part in memory; the body cap above bounds how much that can be.
MultiPartParser.spool_max_size = MAX_BODY_BYTES
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


async def _read_limited(request: Request) -> bytes:
    """The request body, refusing anything over the cap before it is all read."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise HTTPException(413, "files may be at most 10 MB")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise HTTPException(413, "files may be at most 10 MB")
        chunks.append(chunk)
    return b"".join(chunks)


def _buffered(request: Request, body: bytes) -> Request:
    """A copy of ``request`` whose body is the in-memory ``body``."""
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(request.scope, receive)


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
    async def predict_upload(model_id: str, request: Request) -> dict[str, Any]:
        model = model_for(model_id)
        body = await _read_limited(request)
        form = await _buffered(request, body).form(max_files=1, max_fields=0)
        upload = form.get("file")
        if not isinstance(upload, StarletteUploadFile):
            raise HTTPException(422, "send the image as a multipart form field named file")
        data = await upload.read()
        await form.close()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "files may be at most 10 MB")
        image = _decode(data)
        del data, body
        return await run_in_threadpool(run, model, image, None)

    @app.middleware("http")
    async def no_store(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    return app
