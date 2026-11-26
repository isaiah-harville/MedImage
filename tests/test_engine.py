import csv
import json

import numpy as np
import pytest
from PIL import Image

from opendiagnostics.cli import main
from opendiagnostics.config import load_config
from opendiagnostics.data.splits import LeakageError
from opendiagnostics.engine import build, evaluate_run, train

RUN_FILES = {
    "config.yaml",
    "run.json",
    "splits.json",
    "history.jsonl",
    "metrics.json",
    "predictions_val.csv",
    "predictions_test.csv",
    "checkpoint.pt",
    "train.log",
}


def _metrics(run_dir):
    return json.loads((run_dir / "metrics.json").read_text())


@pytest.mark.parametrize("name", ["classification", "classification_3d", "detection"])
def test_smoke_config_trains_end_to_end(name, root, tmp_path):
    config = load_config(root / f"configs/smoke/{name}.yaml", [f"output_dir={tmp_path}"])
    run_dir = train(config, root)
    assert {p.name for p in run_dir.iterdir()} >= RUN_FILES
    metrics = _metrics(run_dir)
    monitor = "map50" if name == "detection" else "auroc"
    test = metrics["test"][monitor]
    assert test["lower"] <= test["value"] <= test["upper"]
    assert test["unit"] == "patient"
    record = json.loads((run_dir / "run.json").read_text())
    assert record["git"]["commit"]
    assert record["environment"]["packages"]["torch"]
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert len(history) == config.train.epochs


def test_runs_are_bitwise_reproducible(root, tmp_path):
    config = load_config(root / "configs/smoke/classification.yaml", [f"output_dir={tmp_path}"])
    first = _metrics(train(config, root))
    second = _metrics(train(config, root))
    assert first == second


def test_different_seeds_differ(root, tmp_path):
    a = load_config(
        root / "configs/smoke/classification.yaml", [f"output_dir={tmp_path}", "seed=0"]
    )
    b = load_config(
        root / "configs/smoke/classification.yaml", [f"output_dir={tmp_path}", "seed=1"]
    )
    assert _metrics(train(a, root))["test"] != _metrics(train(b, root))["test"]


def test_evaluate_run_reproduces_test_metrics(root, tmp_path):
    config = load_config(root / "configs/smoke/classification.yaml", [f"output_dir={tmp_path}"])
    run_dir = train(config, root)
    assert evaluate_run(run_dir)["test"] == _metrics(run_dir)["test"]


def test_predefined_split_with_shared_patient_is_refused(tmp_path):
    rows = [
        ("a.png", "p1", "1", "train"),
        ("b.png", "p1", "0", "test"),
        ("c.png", "p2", "0", "val"),
    ]
    for path, *_ in rows:
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path / path)
    with (tmp_path / "labels.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "patient_id", "label", "split"])
        writer.writerows(rows)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "name: leak\ntask: classification\nmodel: simple_cnn\n"
        f"data: {{dataset: {{name: csv_classification, params: {{manifest: {tmp_path / 'labels.csv'}, split_column: split}}}}}}\n"
    )
    with pytest.raises(LeakageError, match="patient 'p1'"):
        build(load_config(config_path))


def test_model_params_cannot_override_dataset_sizing(root):
    config = load_config(root / "configs/smoke/classification.yaml", ["model.params.num_classes=5"])
    with pytest.raises(ValueError, match="derives from the dataset"):
        build(config, root)


def test_cli_train_and_list(root, tmp_path, capsys):
    assert (
        main(
            [
                "train",
                str(root / "configs/smoke/classification.yaml"),
                "train.epochs=1",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert (tmp_path / "smoke-classification").is_dir()
    assert main(["list"]) == 0
    assert "medmnist" in capsys.readouterr().out
