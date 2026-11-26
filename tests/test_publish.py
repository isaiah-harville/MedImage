import json
import shutil

import pytest
import yaml

from opendiagnostics.publish import PublishError, publish_run


def _artifacts(target):
    return yaml.safe_load((target / "artifacts.yaml").read_text())["files"]


def test_publish_copies_small_files_and_lists_the_checkpoint(finished_run, project):
    target = publish_run(finished_run, project)
    assert target == project / "runs" / finished_run.name
    names = {p.name for p in target.iterdir()}
    assert {"config.yaml", "run.json", "metrics.json", "history.jsonl", "splits.json"} <= names
    assert {"predictions_val.csv", "predictions_test.csv", "artifacts.yaml"} <= names
    assert "checkpoint.pt" not in names
    [checkpoint] = _artifacts(target)
    assert checkpoint["file"] == "checkpoint.pt"
    assert checkpoint["bytes"] == (finished_run / "checkpoint.pt").stat().st_size
    assert checkpoint["url"] is None


def test_republish_keeps_a_recorded_url_while_the_checkpoint_is_unchanged(
    finished_run, project, tmp_path
):
    target = publish_run(finished_run, project)
    files = _artifacts(target)
    files[0]["url"] = "https://example.org/checkpoint.pt"
    (target / "artifacts.yaml").write_text(yaml.safe_dump({"files": files}))
    assert (
        _artifacts(publish_run(finished_run, project))[0]["url"]
        == "https://example.org/checkpoint.pt"
    )

    changed = tmp_path / finished_run.name
    shutil.copytree(finished_run, changed)
    (changed / "checkpoint.pt").write_bytes(b"different weights")
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
