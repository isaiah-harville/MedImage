import importlib.util
import json
import sys

import pytest
import yaml


@pytest.fixture
def queue_module(root):
    spec = importlib.util.spec_from_file_location("run_queue", root / "scripts/run_queue.py")
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _queue(tmp_path, jobs):
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump({"jobs": jobs}))
    return path


def _say(text):
    return [sys.executable, "-c", f"print({text!r})"]


def test_jobs_run_in_order_and_a_failed_download_skips_only_its_job(queue_module, tmp_path):
    queue = _queue(
        tmp_path,
        [
            {"name": "first", "run": _say("finished: runs/a/r1/metrics.json")},
            {
                "name": "broken",
                "config": "unused.yaml",
                "prepare": [sys.executable, "-c", "raise SystemExit(3)"],
            },
            {"name": "last", "run": _say("done")},
        ],
    )
    logs = tmp_path / "logs"
    assert queue_module.main([str(queue), "--logs", str(logs)]) == 1
    status = json.loads((logs / "queue-status.json").read_text())
    assert status["first"]["state"] == "done"
    assert status["first"]["run_dir"] == "runs/a/r1"
    assert status["broken"] == {"state": "failed", "step": "prepare", "exit": 3}
    assert status["last"]["state"] == "done"
    assert not (logs / "broken-fit.log").exists()


def test_rerunning_a_queue_skips_jobs_that_succeeded(queue_module, tmp_path):
    queue = _queue(tmp_path, [{"name": "once", "run": _say("hello")}])
    logs = tmp_path / "logs"
    assert queue_module.main([str(queue), "--logs", str(logs)]) == 0
    (logs / "once-run.log").write_text("kept")
    assert queue_module.main([str(queue), "--logs", str(logs)]) == 0
    assert (logs / "once-run.log").read_text() == "kept"


@pytest.mark.parametrize(
    ("jobs", "message"),
    [
        ([{"name": "a", "run": ["x"], "config": "c.yaml"}], "exactly one of config or run"),
        ([{"name": "a", "run": ["x"]}, {"name": "a", "run": ["y"]}], "appears twice"),
        ([{"name": "bad name", "run": ["x"]}], "needs a name"),
    ],
)
def test_a_malformed_queue_is_refused(queue_module, tmp_path, jobs, message):
    with pytest.raises(SystemExit, match=message):
        queue_module.load_jobs(_queue(tmp_path, jobs))
