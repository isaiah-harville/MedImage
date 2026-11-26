import json
import shutil

import pytest
import yaml

from opendiagnostics.publish import PublishError, publish_run


def _artifacts(target):
    return yaml.safe_load((target / "artifacts.yaml").read_text())["files"]


def test_publish_copies_the_run_and_its_checkpoint_for_open_data(finished_run, project):
    target = publish_run(finished_run, project)
    assert target == project / "runs" / finished_run.name
    names = {p.name for p in target.iterdir()}
    assert {"config.yaml", "run.json", "metrics.json", "metrics.csv", "splits.json"} <= names
    assert {"predictions_val.csv", "predictions_test.csv", "artifacts.yaml"} <= names
    assert (target / "checkpoint.ckpt").read_bytes() == (
        finished_run / "checkpoint.ckpt"
    ).read_bytes()
    [checkpoint] = _artifacts(target)
    assert checkpoint["file"] == "checkpoint.ckpt"
    assert checkpoint["bytes"] == (finished_run / "checkpoint.ckpt").stat().st_size
    assert checkpoint["url"] is None


def test_republish_keeps_a_recorded_url_while_the_checkpoint_is_unchanged(
    finished_run, project, tmp_path
):
    target = publish_run(finished_run, project)
    files = _artifacts(target)
    files[0]["url"] = "https://example.org/checkpoint.ckpt"
    (target / "artifacts.yaml").write_text(yaml.safe_dump({"files": files}))
    assert (
        _artifacts(publish_run(finished_run, project))[0]["url"]
        == "https://example.org/checkpoint.ckpt"
    )

    changed = tmp_path / finished_run.name
    shutil.copytree(finished_run, changed)
    (changed / "checkpoint.ckpt").write_bytes(b"different weights")
    assert _artifacts(publish_run(changed, project))[0]["url"] is None


def test_predictions_can_be_left_out(finished_run, project):
    target = publish_run(finished_run, project, include_predictions=False)
    assert not list(target.glob("predictions_*.csv"))


def test_dirty_runs_are_refused_unless_allowed(finished_run, project, tmp_path):
    dirty = tmp_path / finished_run.name
    shutil.copytree(finished_run, dirty)
    record = json.loads((dirty / "run.json").read_text())
    record["git"]["dirty"] = True
    (dirty / "run.json").write_text(json.dumps(record))
    with pytest.raises(PublishError, match="dirty working tree"):
        publish_run(dirty, project)
    assert publish_run(dirty, project, allow_dirty=True).exists()


def test_example_images_are_published_for_open_data(finished_run, project):
    target = publish_run(finished_run, project)
    assert (target / "examples" / "examples.json").is_file()
    assert list((target / "examples").glob("*.png"))


@pytest.mark.parametrize(
    "license_", ["PhysioNet Credentialed Health Data License 1.5.0", "unknown"]
)
def test_example_images_stay_private_unless_the_license_allows_sharing(
    finished_run, project, tmp_path, license_
):
    restricted = tmp_path / finished_run.name
    shutil.copytree(finished_run, restricted)
    record = json.loads((restricted / "run.json").read_text())
    record["dataset"]["license"] = license_
    (restricted / "run.json").write_text(json.dumps(record))
    assert not (publish_run(restricted, project) / "examples").exists()


def test_example_images_are_left_out_with_the_predictions(finished_run, project):
    target = publish_run(finished_run, project, include_predictions=False)
    assert not (target / "examples").exists()


@pytest.mark.parametrize(
    "license_", ["PhysioNet Credentialed Health Data License 1.5.0", "unknown"]
)
def test_checkpoints_stay_out_of_git_unless_the_license_allows_sharing(
    finished_run, project, tmp_path, license_
):
    restricted = tmp_path / finished_run.name
    shutil.copytree(finished_run, restricted)
    record = json.loads((restricted / "run.json").read_text())
    record["dataset"]["license"] = license_
    (restricted / "run.json").write_text(json.dumps(record))
    target = publish_run(restricted, project)
    assert not (target / "checkpoint.ckpt").exists()
    assert _artifacts(target)[0]["sha256"]


def test_checkpoint_can_be_left_out(finished_run, project):
    publish_run(finished_run, project)
    target = publish_run(finished_run, project, include_checkpoint=False)
    assert not (target / "checkpoint.ckpt").exists()


def test_checkpoints_are_committed_through_git_lfs(root):
    attributes = (root / ".gitattributes").read_text()
    assert "models/*/runs/*/checkpoint.ckpt filter=lfs" in attributes
