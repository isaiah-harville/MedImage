"""Recompute the paper's numbers from the run's committed files and write generated/numbers.tex.

Run from the repository root:

    uv run python papers/synapsemnist3d/scripts/make_numbers.py

The script recomputes every test and validation metric from predictions_*.csv with the
repository's own bootstrap, and fails if any value differs from metrics.json.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from opendiagnostics.metrics import classification as cm
from opendiagnostics.metrics.bootstrap import bootstrap_draws

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "models" / "synapsemnist3d-resnet18"
RUN = PROJECT / "runs" / "20261127-051633-f5029e64"
OUT = Path(__file__).resolve().parents[1] / "generated" / "numbers.tex"


def read_predictions(split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    with (RUN / f"predictions_{split}.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [r["id"] for r in rows]
    target = np.array([int(r["target"]) for r in rows])
    prob = np.array([float(r["prob/excitatory synapse"]) for r in rows])
    return ids, target, prob


def statistic_for(target: np.ndarray, prob: np.ndarray, threshold: float):
    def statistic(index: np.ndarray) -> dict[str, float]:
        t, p = target[index], prob[index]
        return {
            "accuracy_at_0.5": cm.threshold_metrics(t, p, 0.5)["accuracy"],
            "auroc": cm.auroc(t, p),
            "auprc": cm.auprc(t, p),
            "brier": cm.brier(t, p),
            "ece": cm.expected_calibration_error(t, p, 15),
        } | cm.threshold_metrics(t, p, threshold)

    return statistic


def interval(draws: np.ndarray) -> tuple[float, float]:
    values = draws[~np.isnan(draws)]
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def recompute(split: str, metrics: dict, threshold: float) -> dict[str, np.ndarray]:
    ids, target, prob = read_predictions(split)
    assert len(ids) == len(set(ids)) == metrics[split]["n_samples"], split
    point, draws = bootstrap_draws(statistic_for(target, prob, threshold), len(ids), ids, 1000, 0)
    for key, value in point.items():
        stored = metrics[split][key]
        lower, upper = interval(draws[key])
        for name, mine, theirs in (
            ("value", value, stored["value"]),
            ("lower", lower, stored["lower"]),
            ("upper", upper, stored["upper"]),
        ):
            if not math.isclose(mine, theirs, abs_tol=1e-9):
                sys.exit(f"{split} {key} {name}: recomputed {mine}, metrics.json has {theirs}")
    return draws


def f4(value: float) -> str:
    return f"{value:.4f}"


def f3(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    metrics = json.loads((RUN / "metrics.json").read_text())
    run = json.loads((RUN / "run.json").read_text())
    splits = json.loads((RUN / "splits.json").read_text())
    baselines = yaml.safe_load((PROJECT / "baselines.yaml").read_text())["baselines"]
    threshold = metrics["calibration"]["threshold"]

    recompute("val", metrics, threshold)
    test_draws = recompute("test", metrics, threshold)

    ids = {k: set(v) for k, v in splits["ids"].items()}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    sizes = {k: len(v) for k, v in ids.items()}
    assert sizes == {"train": 1230, "val": 177, "test": 352}, sizes

    _, target, _ = read_predictions("test")
    _, val_target, _ = read_predictions("val")

    with (RUN / "metrics.csv").open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["val/auroc"]]
    epochs = len(rows)
    selected = metrics["checkpoint_epoch"]
    logged = next(float(r["val/auroc"]) for r in rows if int(r["epoch"]) == selected)

    started = datetime.fromisoformat(run["started"])
    finished = datetime.fromisoformat(run["finished"])
    minutes = (finished - started).total_seconds() / 60

    by_method = {b["method"]: b["metrics"] for b in baselines}
    park = by_method["Axial-centric cross-plane attention (64³)"]
    r50 = by_method["ResNet-50 + 3D (28³), mean of at least 3 trials"]
    r18 = by_method["ResNet-18 + 3D (28³), mean of at least 3 trials"]
    test = metrics["test"]
    auroc_draws = test_draws["auroc"][~np.isnan(test_draws["auroc"])]
    below_park = int(np.sum(auroc_draws <= park["auroc"]))

    macros = {
        "testAuroc": f4(test["auroc"]["value"]),
        "testAurocLo": f4(test["auroc"]["lower"]),
        "testAurocHi": f4(test["auroc"]["upper"]),
        "testAccHalf": f4(test["accuracy_at_0.5"]["value"]),
        "testAccHalfLo": f4(test["accuracy_at_0.5"]["lower"]),
        "testAccHalfHi": f4(test["accuracy_at_0.5"]["upper"]),
        "testAuprc": f4(test["auprc"]["value"]),
        "testAuprcLo": f4(test["auprc"]["lower"]),
        "testAuprcHi": f4(test["auprc"]["upper"]),
        "testBrier": f4(test["brier"]["value"]),
        "testBrierLo": f4(test["brier"]["lower"]),
        "testBrierHi": f4(test["brier"]["upper"]),
        "testEce": f4(test["ece"]["value"]),
        "testEceLo": f4(test["ece"]["lower"]),
        "testEceHi": f4(test["ece"]["upper"]),
        "testSens": f4(test["sensitivity"]["value"]),
        "testSensLo": f4(test["sensitivity"]["lower"]),
        "testSensHi": f4(test["sensitivity"]["upper"]),
        "testSpec": f4(test["specificity"]["value"]),
        "testSpecLo": f4(test["specificity"]["lower"]),
        "testSpecHi": f4(test["specificity"]["upper"]),
        "valAuroc": f4(metrics["val"]["auroc"]["value"]),
        "valAurocLo": f4(metrics["val"]["auroc"]["lower"]),
        "valAurocHi": f4(metrics["val"]["auroc"]["upper"]),
        "valAurocLogged": f4(logged),
        "threshold": f3(threshold),
        "nTrain": str(sizes["train"]),
        "nVal": str(sizes["val"]),
        "nTest": str(sizes["test"]),
        "nTestExc": str(int(target.sum())),
        "nTestInh": str(int(len(target) - target.sum())),
        "nValExc": str(int(val_target.sum())),
        "nValInh": str(int(len(val_target) - val_target.sum())),
        "prevalence": f3(test["prevalence"]),
        "bootReps": str(metrics["bootstrap"]["replicates"]),
        "bootBelowPark": str(below_park),
        "selectedEpoch": str(selected),
        "epochs": str(epochs),
        "trainMinutes": str(round(minutes)),
        "params": f"{run['parameters'] / 1e6:.1f}",
        "parkAuroc": f"{park['auroc']:.2f}",
        "parkAcc": f3(park["accuracy"]),
        "medmnistRfiftyAuroc": f3(r50["auroc"]),
        "medmnistRfiftyAcc": f3(r50["accuracy"]),
        "medmnistReighteenAuroc": f"{r18['auroc']:.2f}",
        "medmnistReighteenAcc": f3(r18["accuracy"]),
        "marginPark": f3(test["auroc"]["value"] - park["auroc"]),
        "marginParkLo": f3(test["auroc"]["lower"] - park["auroc"]),
        "commit": run["git"]["commit"][:8],
        "gpu": run["environment"]["gpu"].replace("NVIDIA GeForce ", ""),
        "torchVersion": run["environment"]["packages"]["torch"].split("+")[0],
        "lightningVersion": run["environment"]["packages"]["lightning"],
        "monaiVersion": run["environment"]["packages"]["monai"],
    }
    for name in macros:
        assert name.isalpha(), name
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    OUT.write_text("% Written by scripts/make_numbers.py. Do not edit.\n" + "\n".join(lines) + "\n")
    print(
        f"recomputed val and test metrics match metrics.json; wrote {len(macros)} macros to {OUT}"
    )
    for k, v in macros.items():
        print(f"{k}\t{v}")


if __name__ == "__main__":
    main()
