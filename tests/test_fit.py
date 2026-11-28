import csv
import json
import shutil

import numpy as np
import pytest
from conftest import fit_smoke
from PIL import Image

from opendiagnostics.cli import evaluate_run, fit, main
from opendiagnostics.data import LeakageError

RUN_FILES = {
    "config.yaml",
    "run.json",
    "splits.json",
    "metrics.csv",
    "metrics.json",
    "predictions_val.csv",
    "predictions_test.csv",
    "checkpoint.ckpt",
    "train.log",
    "examples",
}


def _metrics(run_dir):
    return json.loads((run_dir / "metrics.json").read_text())


@pytest.mark.parametrize("name", ["classification", "classification_3d", "detection"])
def test_smoke_config_trains_end_to_end(name, tmp_path):
    run_dir = fit_smoke(name, tmp_path)
    assert {p.name for p in run_dir.iterdir()} >= RUN_FILES
    manifest = json.loads((run_dir / "examples" / "examples.json").read_text())
    assert len(manifest) == 8
    for example in manifest:
        assert (run_dir / "examples" / example["input"]).is_file()
        assert (run_dir / "examples" / example["overlay"]).is_file()
    metrics = _metrics(run_dir)
    monitor = "map_50" if name == "detection" else "auroc"
    test = metrics["test"][monitor]
    assert test["lower"] <= test["value"] <= test["upper"]
    assert test["unit"] == "patient"
    record = json.loads((run_dir / "run.json").read_text())
    assert record["git"]["commit"]
    assert record["environment"]["packages"]["lightning"]
    with (run_dir / "metrics.csv").open() as handle:
        epochs = {row["epoch"] for row in csv.DictReader(handle) if row["epoch"]}
    expected = 3 if name == "classification" else 2
    assert len(epochs) == expected


def test_runs_are_bitwise_reproducible_on_cpu(tmp_path):
    first = _metrics(fit_smoke("classification", tmp_path))
    second = _metrics(fit_smoke("classification", tmp_path))
    assert first == second


def test_different_seeds_differ(tmp_path):
    a = _metrics(fit_smoke("classification", tmp_path, "--seed_everything=0"))
    b = _metrics(fit_smoke("classification", tmp_path, "--seed_everything=1"))
    assert a["test"] != b["test"]


def test_evaluate_run_reproduces_test_metrics_and_records_the_rescore(finished_run, tmp_path):
    run_dir = tmp_path / finished_run.name
    shutil.copytree(finished_run, run_dir)
    before = _metrics(run_dir)
    assert evaluate_run(run_dir)["test"] == before["test"]
    after = _metrics(run_dir)
    assert after["test"] == before["test"]
    assert after["selection"] == before["selection"]
    assert after["rescored"]["git"]["commit"]


def test_a_run_with_a_number_typed_transform_argument_can_be_evaluated(tmp_path):
    run_dir = fit_smoke(
        "classification",
        tmp_path,
        "--data.init_args.train_transforms=torchvision.transforms.v2.RandomAffine",
        "--data.init_args.train_transforms.init_args.degrees=10",
    )
    assert "degrees: 10\n" in (run_dir / "config.yaml").read_text()
    assert evaluate_run(run_dir)["test"] == _metrics(run_dir)["test"]


def test_a_number_typed_transform_argument_refuses_text(tmp_path, capsys):
    with pytest.raises(SystemExit):
        fit_smoke(
            "classification",
            tmp_path,
            "--data.init_args.train_transforms=torchvision.transforms.v2.RandomAffine",
            "--data.init_args.train_transforms.init_args.degrees=ten",
        )
    assert "expected a number, got 'ten'" in capsys.readouterr().err


def test_unknown_config_key_is_an_error(tmp_path, capsys):
    with pytest.raises(SystemExit):
        fit_smoke("classification", tmp_path, "--data.init_args.batch_sise=4")
    assert "batch_sise" in capsys.readouterr().err


def test_network_with_wrong_output_size_is_explained(root, tmp_path):
    config = (root / "configs/smoke/classification.yaml").read_text()
    (tmp_path / "two.yaml").write_text(config.replace("num_classes: 1", "num_classes: 2"))
    with pytest.raises(ValueError, match="needs \\(batch, 1\\)"):
        fit(["-c", str(tmp_path / "two.yaml"), f"--output_dir={tmp_path}"])


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
    with pytest.raises(LeakageError, match="patient 'p1'"):
        fit_smoke(
            "classification",
            tmp_path,
            "--data=opendiagnostics.data.CSVImageDataModule",
            f"--data.manifest={tmp_path / 'labels.csv'}",
            "--data.split_column=split",
            "--data.split=predefined",
        )


def test_cli_fit(tmp_path):
    out = tmp_path / "runs"
    argv = ["fit", "-c", "configs/smoke/classification.yaml", "--trainer.max_epochs=1"]
    assert main([*argv, f"--output_dir={out}"]) == 0
    assert (out / "smoke-classification").is_dir()


def test_prepare_readies_a_configs_data_without_training(root, capsys):
    assert main(["prepare", "-c", str(root / "configs/smoke/classification.yaml")]) == 0
    assert "prepared the data for smoke-classification" in capsys.readouterr().out


def test_checkpoint_holds_the_averaged_weights_that_selection_scored(tmp_path):
    averaging = tmp_path / "ema.yaml"
    averaging.write_text(
        "trainer:\n"
        "  callbacks:\n"
        "    - class_path: lightning.pytorch.callbacks.EMAWeightAveraging\n"
        "      init_args: {decay: 0.9}\n"
    )
    run_dir = fit_smoke("classification", tmp_path, "-c", str(averaging))
    metrics = _metrics(run_dir)
    epoch = metrics["selection"]["best_epoch"]
    with (run_dir / "metrics.csv").open() as handle:
        logged = [
            float(row["val/auroc"])
            for row in csv.DictReader(handle)
            if row["epoch"] == str(epoch) and row["val/auroc"]
        ]
    assert metrics["val"]["auroc"]["value"] == pytest.approx(logged[0], abs=1e-6)
