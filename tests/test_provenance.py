import subprocess

import pytest

from opendiagnostics.provenance import git_info


@pytest.fixture
def repo(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    (tmp_path / "models/p/runs/r1").mkdir(parents=True)
    (tmp_path / "models/p/runs/r1/metrics.json").write_text("{}")
    (tmp_path / "models/p/config.yaml").write_text("a: 1\n")
    git("add", ".")
    git("commit", "-qm", "init")
    return tmp_path


def test_a_clean_tree_is_not_dirty(repo):
    assert git_info(repo)["dirty"] is False


def test_changed_run_outputs_do_not_make_a_run_dirty(repo):
    (repo / "models/p/runs/r1/metrics.json").write_text('{"rescored": true}')
    assert git_info(repo)["dirty"] is False


def test_a_changed_config_makes_a_run_dirty(repo):
    (repo / "models/p/config.yaml").write_text("a: 2\n")
    assert git_info(repo)["dirty"] is True
