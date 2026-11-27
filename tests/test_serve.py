import base64
import io
import tempfile
import time

import numpy as np
import pytest
from conftest import MemoryBucket, smoke_project
from fastapi.testclient import TestClient
from litserve.utils import wrap_litserve_start
from PIL import Image

from opendiagnostics.publish import WEIGHTS_URL
from opendiagnostics.serve import settings_from
from opendiagnostics.serve.app import ClassifierAPI, create_server, describe
from opendiagnostics.serve.models import DISCLAIMER
from opendiagnostics.serve.weights import sync


def _ready(client):
    deadline = time.monotonic() + 120
    while client.get("/health").status_code != 200:
        assert time.monotonic() < deadline, "the LitServe worker never became ready"
        time.sleep(0.2)
    return client


@pytest.fixture(scope="module")
def client(finished_run, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("serve")
    bucket = MemoryBucket()
    smoke_project(finished_run, tmp / "models", bucket=bucket)
    cache = tmp / "cache"
    folders = sync(cache, WEIGHTS_URL, bucket.open_url)
    server = create_server(describe(folders), ClassifierAPI(folders, cache, 4))
    with wrap_litserve_start(server) as started, TestClient(started.app) as client:
        yield _ready(client)


def _png(size=(32, 32), mode="L", fmt="PNG"):
    buffer = io.BytesIO()
    Image.fromarray(np.full((*size, 3) if mode == "RGB" else size, 90, np.uint8)).save(buffer, fmt)
    return buffer.getvalue()


def _upload(client, data, model="smoke-project"):
    image = base64.b64encode(data).decode()
    return client.post("/predict", json={"model": model, "image": image})


def test_health_and_model_listing(client):
    assert client.get("/health").status_code == 200
    [model] = client.get("/models").json()
    assert model["id"] == "smoke-project"
    assert model["label_mode"] == "binary"
    assert model["classes"] == ["class_0", "class_1"]
    assert model["demo_count"] == 8
    assert model["category"] == "Other"
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
    result = client.post(
        "/predict", json={"model": "smoke-project", "sample_id": first["sample_id"]}
    ).json()
    assert result["truth"] == first["truth"]
    assert result["disclaimer"] == DISCLAIMER
    assert [p["class"] for p in result["probabilities"]] in (
        ["class_0", "class_1"],
        ["class_1", "class_0"],
    )
    assert sum(p["probability"] for p in result["probabilities"]) == pytest.approx(1)
    assert result["attribution_png"].startswith("data:image/png;base64,")


def test_upload_is_predicted_and_not_given_a_truth(client):
    response = _upload(client, _png(fmt="JPEG"))
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
    assert _upload(client, payload).status_code == status


def test_a_body_over_the_cap_is_refused_before_it_is_read(client):
    response = client.post(
        "/predict",
        content=b"{" + b" " * (14 * 1024 * 1024) + b"}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413


@pytest.mark.parametrize(
    "body",
    [
        {"model": "smoke-project"},
        {"model": "smoke-project", "sample_id": "a", "image": "AAAA"},
        {"model": "smoke-project", "image": "not base64!"},
    ],
)
def test_malformed_requests_are_422(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_unknown_model_and_sample_are_404(client):
    assert client.get("/models/nope/demo").status_code == 404
    assert client.get("/models/smoke-project/demo/nope.png").status_code == 404
    assert _upload(client, _png(), model="nope").status_code == 404
    response = client.post("/predict", json={"model": "smoke-project", "sample_id": "nope"})
    assert response.status_code == 404


def test_settings_require_a_cache_and_default_to_the_public_bucket(tmp_path):
    settings = settings_from({"WEIGHTS_CACHE": str(tmp_path)})
    assert settings.host_port == ("0.0.0.0", 8080)
    assert str(settings.weights_base_url) == WEIGHTS_URL
    assert settings.refresh_seconds == 600
    assert settings.workers == 1
    with pytest.raises(SystemExit, match="WEIGHTS_CACHE is required"):
        settings_from({})


def test_the_service_answers_with_no_models_before_anything_is_published(tmp_path):
    server = create_server({}, ClassifierAPI([], tmp_path, 4))
    with wrap_litserve_start(server) as started, TestClient(started.app) as client:
        _ready(client)
        assert client.get("/models").json() == []
        assert _upload(client, _png()).status_code == 404


def test_every_response_is_marked_no_store(client):
    assert client.get("/models").headers["cache-control"] == "no-store"
    assert _upload(client, _png()).headers["cache-control"] == "no-store"


def test_large_uploads_never_touch_disk(client, monkeypatch):
    def refuse(self, *args, **kwargs):
        raise AssertionError("an upload was spooled to a temporary file on disk")

    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", refuse)
    noise = np.random.default_rng(0).integers(0, 256, (1024, 1024, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(noise).save(buffer, "PNG")
    assert buffer.tell() > 2 * 1024 * 1024
    assert _upload(client, buffer.getvalue()).status_code == 200


def test_the_watcher_restarts_only_when_the_catalog_changes(tmp_path, monkeypatch):
    import threading

    from opendiagnostics import serve
    from opendiagnostics.serve.weights import Catalog

    served = Catalog(models={"a": "r1"}, bundles={"a": "x"})
    answers = iter(
        [
            served,
            RuntimeError("bucket unreachable"),
            Catalog(models={"a": "r1"}, bundles={"a": "y"}),
        ]
    )

    def read(base_url, open_url):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    kills = []
    monkeypatch.setattr(serve, "read_catalog", read)
    monkeypatch.setattr(serve, "_synced", lambda cache, base_url: [])
    monkeypatch.setattr(serve.os, "kill", lambda pid, sig: kills.append(sig))
    settings = settings_from({"WEIGHTS_CACHE": str(tmp_path), "REFRESH_SECONDS": "1"})
    stop = threading.Event()
    monkeypatch.setattr(stop, "wait", lambda timeout: False)
    serve.watch(settings, served, stop)
    assert kills == [serve.signal.SIGTERM]


def test_a_listing_matches_the_model_and_needs_no_weights(finished_run, tmp_path):
    import shutil

    from opendiagnostics.serve.models import load_listing, load_model

    bucket = MemoryBucket()
    smoke_project(finished_run, tmp_path / "models", bucket=bucket)
    cache = tmp_path / "cache"
    [folder] = sync(cache, WEIGHTS_URL, bucket.open_url)
    model = load_model(folder, cache)
    shutil.rmtree(cache / "checkpoints")
    listing = load_listing(folder)
    assert listing.description == model.description
    assert listing.demos.keys() == model.demos.keys()


def test_runs_the_demo_cannot_run_are_left_out_of_the_listing(finished_run, tmp_path):
    import yaml

    bucket = MemoryBucket()
    smoke_project(finished_run, tmp_path / "models", bucket=bucket)
    [folder] = sync(tmp_path / "cache", WEIGHTS_URL, bucket.open_url)
    config = yaml.safe_load((folder / "config.yaml").read_text())
    config["model"]["class_path"] = "opendiagnostics.modules.ObjectDetector"
    (folder / "config.yaml").write_text(yaml.safe_dump(config))
    assert describe([folder]) == {}
