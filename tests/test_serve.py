import importlib.util
import io
import shutil
import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from opendiagnostics.publish import publish_run
from opendiagnostics.serve import settings_from
from opendiagnostics.serve.app import create_app
from opendiagnostics.serve.models import DISCLAIMER, load_models


def _bundler(root):
    spec = importlib.util.spec_from_file_location("bundle", root / "scripts/bundle_models.py")
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def client(finished_run, tmp_path_factory, root):
    models = tmp_path_factory.mktemp("models")
    project = models / "smoke-project"
    project.mkdir()
    shutil.copyfile(root / "configs/smoke/classification.yaml", project / "config.yaml")
    publish_run(finished_run, project, upload=lambda path, key: None)
    weights = tmp_path_factory.mktemp("weights")
    runs = finished_run.parent.parent
    assert _bundler(root).main([str(weights), "--runs", str(runs), "--models", str(models)]) == 0
    return TestClient(create_app(load_models(weights), attribution_steps=4))


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


def test_bundling_again_reuses_a_checkpoint_that_matches(finished_run, tmp_path, root):
    models = tmp_path / "models"
    project = models / "smoke-project"
    project.mkdir(parents=True)
    shutil.copyfile(root / "configs/smoke/classification.yaml", project / "config.yaml")
    publish_run(finished_run, project, include_checkpoint=False)
    weights = tmp_path / "weights"
    bundler = _bundler(root)
    runs = finished_run.parent.parent
    assert bundler.main([str(weights), "--runs", str(runs), "--models", str(models)]) == 0
    (weights / "removed-project").mkdir()
    assert (
        bundler.main([str(weights), "--runs", str(tmp_path / "none"), "--models", str(models)]) == 0
    )
    assert (weights / "smoke-project" / "checkpoint.ckpt").is_file()
    assert not (weights / "removed-project").exists()


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


def test_settings_require_an_existing_weights_dir(tmp_path):
    assert settings_from({"WEIGHTS_DIR": str(tmp_path)}).host_port == ("0.0.0.0", 8080)
    with pytest.raises(SystemExit, match="WEIGHTS_DIR"):
        settings_from({})


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


def test_bundler_exits_3_only_when_nothing_is_published(tmp_path, root):
    (tmp_path / "models" / "empty-project").mkdir(parents=True)
    args = [str(tmp_path / "weights"), "--models", str(tmp_path / "models")]
    assert _bundler(root).main(args) == 3
