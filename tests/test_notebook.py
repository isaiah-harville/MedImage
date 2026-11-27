import pytest
import yaml
from conftest import MemoryBucket

from opendiagnostics.cli import main
from opendiagnostics.notebook import MARKER, NotebookError, build_notebook
from opendiagnostics.publish import WEIGHTS_URL, publish_run

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
    publish_run(finished_run, project, bucket=MemoryBucket())
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
    url = f"{WEIGHTS_URL}/smoke-project/{run_id}/checkpoint.ckpt"
    assert f"[checkpoint.ckpt]({url})" in page
    assert "[Download weights (" in page
    assert f"]({url}){{ .md-button .md-button--primary }}" in page
    assert not (out / "smoke-project" / run_id / "files" / "checkpoint.ckpt").exists()
    assert "## Examples" in page
    assert "](files/examples/01-input.png)" in page
    assert (out / "smoke-project" / run_id / "files" / "examples" / "01-attribution.png").exists()
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


def test_samples_page_shows_one_correct_test_image_per_class(notebook):
    out, run_id = notebook
    page = (out / "samples.md").read_text()
    assert "## SyntheticClassificationDataModule" in page
    assert "License generated." in page
    images = [line for line in page.splitlines() if line.startswith("-   ![")]
    assert len(images) == 2
    assert all(f"smoke-project/{run_id}/files/examples/" in line for line in images)


def test_samples_page_says_when_there_are_no_examples(project, tmp_path):
    out = tmp_path / "notebook"
    build_notebook(project.parent, out)
    assert "No published run has example images yet." in (out / "samples.md").read_text()


def test_roc_plot_is_skipped_for_a_multilabel_run(tmp_path):
    from opendiagnostics.notebook import plots

    csv = tmp_path / "predictions_test.csv"
    csv.write_text("id,patient_id,target/a,target/b,prob/a,prob/b\ns1,p1,1.0,0.0,0.9,0.2\n")
    assert plots.roc_and_pr(csv, tmp_path / "roc.svg") is False
    assert not (tmp_path / "roc.svg").exists()
