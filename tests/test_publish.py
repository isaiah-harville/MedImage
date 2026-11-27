import json
import shutil

import pytest
import yaml
from boto3.exceptions import S3UploadFailedError

from opendiagnostics import publish
from opendiagnostics.cli import main
from opendiagnostics.publish import WEIGHTS_URL, PublishError, publish_run


class Bucket:
    def __init__(self):
        self.objects = {}

    def __call__(self, path, key):
        self.objects[key] = path.read_bytes()


@pytest.fixture
def bucket():
    return Bucket()


def _artifacts(target):
    return yaml.safe_load((target / "artifacts.yaml").read_text())["files"]


def test_publish_copies_the_run_and_uploads_its_checkpoint_for_open_data(
    finished_run, project, bucket
):
    target = publish_run(finished_run, project, upload=bucket)
    assert target == project / "runs" / finished_run.name
    names = {p.name for p in target.iterdir()}
    assert {"config.yaml", "run.json", "metrics.json", "metrics.csv", "splits.json"} <= names
    assert {"predictions_val.csv", "predictions_test.csv", "artifacts.yaml"} <= names
    assert "checkpoint.ckpt" not in names
    key = f"{project.name}/{finished_run.name}/checkpoint.ckpt"
    assert bucket.objects == {key: (finished_run / "checkpoint.ckpt").read_bytes()}
    [checkpoint] = _artifacts(target)
    assert checkpoint["file"] == "checkpoint.ckpt"
    assert checkpoint["bytes"] == (finished_run / "checkpoint.ckpt").stat().st_size
    assert checkpoint["url"] == f"{WEIGHTS_URL}/{key}"


def test_republish_uploads_again_only_when_the_checkpoint_changed(
    finished_run, project, tmp_path, bucket
):
    publish_run(finished_run, project, upload=bucket)
    bucket.objects.clear()
    publish_run(finished_run, project, upload=bucket)
    assert bucket.objects == {}

    changed = tmp_path / finished_run.name
    shutil.copytree(finished_run, changed)
    (changed / "checkpoint.ckpt").write_bytes(b"different weights")
    publish_run(changed, project, upload=bucket)
    assert list(bucket.objects.values()) == [b"different weights"]


def test_predictions_can_be_left_out(finished_run, project, bucket):
    target = publish_run(finished_run, project, include_predictions=False, upload=bucket)
    assert not list(target.glob("predictions_*.csv"))


def test_dirty_runs_are_refused_unless_allowed(finished_run, project, tmp_path, bucket):
    dirty = tmp_path / finished_run.name
    shutil.copytree(finished_run, dirty)
    record = json.loads((dirty / "run.json").read_text())
    record["git"]["dirty"] = True
    (dirty / "run.json").write_text(json.dumps(record))
    with pytest.raises(PublishError, match="dirty working tree"):
        publish_run(dirty, project, upload=bucket)
    assert bucket.objects == {}
    assert publish_run(dirty, project, allow_dirty=True, upload=bucket).exists()


def test_example_images_are_published_for_open_data(finished_run, project, bucket):
    target = publish_run(finished_run, project, upload=bucket)
    assert (target / "examples" / "examples.json").is_file()
    assert list((target / "examples").glob("*.png"))


@pytest.mark.parametrize(
    "license_", ["PhysioNet Credentialed Health Data License 1.5.0", "unknown"]
)
def test_example_images_stay_private_unless_the_license_allows_sharing(
    finished_run, project, tmp_path, license_, bucket
):
    restricted = tmp_path / finished_run.name
    shutil.copytree(finished_run, restricted)
    record = json.loads((restricted / "run.json").read_text())
    record["dataset"]["license"] = license_
    (restricted / "run.json").write_text(json.dumps(record))
    assert not (publish_run(restricted, project, upload=bucket) / "examples").exists()


def test_example_images_are_left_out_with_the_predictions(finished_run, project, bucket):
    target = publish_run(finished_run, project, include_predictions=False, upload=bucket)
    assert not (target / "examples").exists()


@pytest.mark.parametrize(
    "license_", ["PhysioNet Credentialed Health Data License 1.5.0", "unknown"]
)
def test_checkpoints_are_never_uploaded_unless_the_license_allows_sharing(
    finished_run, project, tmp_path, license_, bucket
):
    restricted = tmp_path / finished_run.name
    shutil.copytree(finished_run, restricted)
    record = json.loads((restricted / "run.json").read_text())
    record["dataset"]["license"] = license_
    (restricted / "run.json").write_text(json.dumps(record))
    [checkpoint] = _artifacts(publish_run(restricted, project, upload=bucket))
    assert bucket.objects == {}
    assert checkpoint["sha256"]
    assert checkpoint["url"] is None


def test_checkpoint_upload_can_be_skipped(finished_run, project, bucket):
    [checkpoint] = _artifacts(
        publish_run(finished_run, project, include_checkpoint=False, upload=bucket)
    )
    assert bucket.objects == {}
    assert checkpoint["url"] is None


def test_publish_explains_how_to_fix_a_rejected_upload(finished_run, project, monkeypatch):
    class RejectingClient:
        def upload_file(self, *args, **kwargs):
            raise S3UploadFailedError("An error occurred (InvalidAccessKeyId)")

    monkeypatch.setattr(publish.boto3, "client", lambda *args, **kwargs: RejectingClient())
    with pytest.raises(PublishError, match="--no-checkpoint"):
        publish_run(finished_run, project)


def test_no_checkpoint_flag_publishes_without_uploading(finished_run, project, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("uploaded despite --no-checkpoint")

    monkeypatch.setattr(publish.boto3, "client", refuse)
    assert main(["publish", str(finished_run), str(project), "--no-checkpoint"]) == 0
    [checkpoint] = _artifacts(project / "runs" / finished_run.name)
    assert checkpoint["url"] is None
