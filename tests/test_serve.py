import io
import tempfile

import numpy as np
import pytest
from conftest import MemoryBucket, smoke_project
from fastapi.testclient import TestClient
from PIL import Image

from opendiagnostics.publish import WEIGHTS_URL
from opendiagnostics.serve import settings_from
from opendiagnostics.serve.app import create_app
from opendiagnostics.serve.models import DISCLAIMER, load_models, refresh
from opendiagnostics.serve.weights import sync


@pytest.fixture(scope="module")
def synced(finished_run, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("serve")
    bucket = MemoryBucket()
    smoke_project(finished_run, tmp / "models", bucket=bucket)
    cache = tmp / "cache"
    return sync(cache, WEIGHTS_URL, bucket.open_url), cache


@pytest.fixture(scope="module")
def client(synced):
    models = load_models(*synced)
    return TestClient(create_app(lambda: models, attribution_steps=4))


def _png(size=(32, 32), mode="L", fmt="PNG"):
    buffer = io.BytesIO()
    Image.fromarray(np.full((*size, 3) if mode == "RGB" else size, 90, np.uint8)).save(buffer, fmt)
    return buffer.getvalue()


def test_health_and_model_listing(client):
    assert client.get("/healthz").json() == {"status": "ok", "models": 1}
    [model] = client.get("/models").json()
    assert model["id"] == "smoke-project"
    assert model["label_mode"] == "binary"
    assert model["classes"] == ["class_0", "class_1"]
    assert model["demo_count"] == 8
    assert model["run_id"]
    assert set(model["test_auroc"]) == {"value", "lower", "upper"}
    assert model["weights"]["url"].endswith(f"/smoke-project/{model['run_id']}/checkpoint.ckpt")
    assert len(model["weights"]["sha256"]) == 64


def test_demo_images_are_listed_served_and_predicted(client):
    demos = client.get("/models/smoke-project/demo").json()
    assert len(demos) == 8
    first = demos[0]
    image = client.get(first["image_url"])
    assert image.headers["content-type"] == "image/png"
    result = client.post(f"/models/smoke-project/demo/{first['sample_id']}/predict").json()
    assert result["truth"] == first["truth"]
    assert result["disclaimer"] == DISCLAIMER
    assert [p["class"] for p in result["probabilities"]] in (
        ["class_0", "class_1"],
        ["class_1", "class_0"],
    )
    assert sum(p["probability"] for p in result["probabilities"]) == pytest.approx(1)
    assert result["attribution_png"].startswith("data:image/png;base64,")


def test_upload_is_predicted_and_not_given_a_truth(client):
    response = client.post(
        "/models/smoke-project/predict", files={"file": ("x.jpg", _png(fmt="JPEG"), "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["truth"] is None
    assert response.json()["predicted"][0] in ("class_0", "class_1")


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        (_png(fmt="GIF"), 415),
        (b"not an image", 415),
        (_png(size=(4097, 8)), 413),
        (b"0" * (10 * 1024 * 1024 + 1), 413),
    ],
)
def test_bad_uploads_are_refused(client, payload, status):
    response = client.post("/models/smoke-project/predict", files={"file": ("f", payload)})
    assert response.status_code == status


def test_unknown_model_and_sample_are_404(client):
    assert client.get("/models/nope/demo").status_code == 404
    assert client.post("/models/smoke-project/demo/nope/predict").status_code == 404


def test_settings_require_a_cache_and_default_to_the_public_bucket(tmp_path):
    settings = settings_from({"WEIGHTS_CACHE": str(tmp_path)})
    assert settings.host_port == ("0.0.0.0", 8080)
    assert str(settings.weights_base_url) == WEIGHTS_URL
    assert settings.refresh_seconds == 600
    with pytest.raises(SystemExit, match="WEIGHTS_CACHE is required"):
        settings_from({})


def test_the_service_answers_with_no_models_before_anything_is_published():
    client = TestClient(create_app(lambda: {}))
    assert client.get("/healthz").json() == {"status": "ok", "models": 0}
    assert client.get("/models").json() == []
    assert client.get("/models/smoke-project/demo").status_code == 404


def test_refresh_keeps_unchanged_models_and_drops_retired_ones(synced):
    folders, cache = synced
    models = refresh({}, folders, cache)
    assert list(models) == ["smoke-project"]
    assert refresh(models, folders, cache)["smoke-project"] is models["smoke-project"]
    assert refresh(models, [], cache) == {}


def test_every_response_is_marked_no_store(client):
    assert client.get("/models").headers["cache-control"] == "no-store"
    response = client.post(
        "/models/smoke-project/predict", files={"file": ("x.png", _png(), "image/png")}
    )
    assert response.headers["cache-control"] == "no-store"


def test_large_uploads_never_touch_disk(client, monkeypatch):
    def refuse(self, *args, **kwargs):
        raise AssertionError("an upload was spooled to a temporary file on disk")

    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", refuse)
    noise = np.random.default_rng(0).integers(0, 256, (1024, 1024, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(noise).save(buffer, "PNG")
    assert buffer.tell() > 2 * 1024 * 1024
    response = client.post(
        "/models/smoke-project/predict", files={"file": ("x.png", buffer.getvalue(), "image/png")}
    )
    assert response.status_code == 200
