import io
import tempfile

import numpy as np
import pytest
from conftest import bundle_published, bundler, serve_checkpoint, smoke_project
from fastapi.testclient import TestClient
from PIL import Image

from opendiagnostics.publish import WEIGHTS_URL
from opendiagnostics.serve import settings_from
from opendiagnostics.serve.app import create_app
from opendiagnostics.serve.models import DISCLAIMER, load_models
from opendiagnostics.serve.weights import fetch


@pytest.fixture(scope="module")
def client(finished_run, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("serve")
    weights = bundle_published(finished_run, tmp)
    fetch(weights, tmp / "cache", WEIGHTS_URL, serve_checkpoint(finished_run / "checkpoint.ckpt"))
    return TestClient(create_app(load_models(weights, tmp / "cache"), attribution_steps=4))


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


def test_settings_require_an_existing_weights_dir_and_a_cache(tmp_path):
    settings = settings_from({"WEIGHTS_DIR": str(tmp_path), "WEIGHTS_CACHE": str(tmp_path)})
    assert settings.host_port == ("0.0.0.0", 8080)
    assert str(settings.weights_base_url) == WEIGHTS_URL
    with pytest.raises(SystemExit, match="WEIGHTS_DIR and WEIGHTS_CACHE"):
        settings_from({"WEIGHTS_DIR": str(tmp_path)})


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


def test_bundler_skips_runs_without_a_public_checkpoint_and_exits_3_when_none_remain(
    finished_run, tmp_path
):
    smoke_project(finished_run, tmp_path / "models", include_checkpoint=False)
    weights = tmp_path / "weights"
    (weights / "removed-project").mkdir(parents=True)
    assert bundler().main([str(weights), "--models", str(tmp_path / "models")]) == 3
    assert list(weights.iterdir()) == []
