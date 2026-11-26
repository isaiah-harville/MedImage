import pytest
import yaml

from medimage.cli import main
from medimage.notebook import MARKER, NotebookError, build_notebook
from medimage.publish import publish_run

BASELINES = {
    "benchmark": "smoke benchmark",
    "protocol": "test AUROC.",
    "baselines": [
        {
            "method": "Perfect",
            "metrics": {"auroc": 1.0},
            "source": "Doe 2020",
            "url": "https://example.org/a",
        },
        {
            "method": "Chance",
            "metrics": {"auroc": 0.3},
            "source": "Doe 2020",
            "url": "https://example.org/b",
        },
    ],
}


@pytest.fixture
def notebook(finished_run, project, tmp_path):
    publish_run(finished_run, project)
    (project / "baselines.yaml").write_text(yaml.safe_dump(BASELINES))
    (project / "MODEL_CARD.md").write_text("# Smoke model\n\nA card.\n")
    out = tmp_path / "docs" / "notebook"
    build_notebook(project.parent, out)
    return out, finished_run.name


def test_run_page_shows_metrics_plots_and_downloads(notebook):
    out, run_id = notebook
    page = (out / "smoke-project" / run_id / "index.md").read_text()
    assert "| auroc |" in page
    assert "![Training curves](files/training.svg)" in page
    assert "![ROC and precision-recall on test](files/roc_pr.svg)" in page
    assert "[metrics.json](files/metrics.json)" in page
    assert "checkpoint.pt (not uploaded yet)" in page
    for name in ("metrics.json", "predictions_test.csv", "training.svg", "roc_pr.svg"):
        assert (out / "smoke-project" / run_id / "files" / name).exists()


def test_rebuilding_gives_identical_files(notebook, project):
    out, _ = notebook
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    build_notebook(project.parent, out)
    after = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    assert before == after


def test_refuses_to_replace_a_directory_it_did_not_make(project, tmp_path):
    out = tmp_path / "docs"
    out.mkdir()
    (out / "design.md").write_text("hand-written")
    with pytest.raises(NotebookError, match="refusing to replace"):
        build_notebook(project.parent, out)
    assert (out / "design.md").exists()
    assert not (out / MARKER).exists()


def test_cli_builds_the_notebook(notebook, project, tmp_path, capsys):
    out = tmp_path / "cli-notebook"
    assert main(["notebook", "--models", str(project.parent), "--out", str(out)]) == 0
    assert "wrote 1 projects and 1 runs" in capsys.readouterr().out
