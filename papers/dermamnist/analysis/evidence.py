"""Recompute every number the DermaMNIST paper uses from the committed run files.

    uv run python papers/dermamnist/analysis/evidence.py            # print and check
    uv run python papers/dermamnist/analysis/evidence.py --write    # also write numbers.tex

Reads predictions_test.csv, predictions_val.csv, metrics.json, metrics.csv, and run.json
from the run, and baselines.yaml from the model project. Recomputes the metrics with the
pipeline's own metric and bootstrap code, checks each against metrics.json and the AUROC and
accuracy against the official MedMNIST evaluator, and fails if any differs by more than 1e-9.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from medmnist.evaluator import getACC, getAUC
from sklearn.metrics import confusion_matrix

from opendiagnostics.metrics import classification as cm
from opendiagnostics.metrics.bootstrap import bootstrap
from opendiagnostics.metrics.compare import read_predictions

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parents[1] / "models/dermamnist-convnext"
RUN = PROJECT / "runs/20261126-141911-ab39eb97"
SHORT = {
    "actinic keratoses and intraepithelial carcinoma": "akiec",
    "basal cell carcinoma": "bcc",
    "benign keratosis-like lesions": "bkl",
    "dermatofibroma": "df",
    "melanoma": "mel",
    "melanocytic nevi": "nv",
    "vascular lesions": "vasc",
}


def statistic_for(target: np.ndarray, prob: np.ndarray, names: list[str]):
    def statistic(index: np.ndarray) -> dict[str, float]:
        t, p = target[index], prob[index]
        roc = cm.per_class(t, p, cm.auroc)
        prc = cm.per_class(t, p, cm.auprc)
        predicted = p.argmax(1)
        out = {
            "auroc": cm.nanmean(roc),
            "auprc": cm.nanmean(prc),
            "accuracy": float(np.mean(predicted == t)),
            "balanced_accuracy": cm.balanced_accuracy(t, predicted, len(names)),
        }
        for name, a, b in zip(names, roc, prc, strict=True):
            out[f"auroc/{name}"] = float(a)
            out[f"auprc/{name}"] = float(b)
        return out

    return statistic


def rescore(split: str, recorded: dict) -> dict:
    pred = read_predictions(RUN / f"predictions_{split}.csv")
    estimates = bootstrap(
        statistic_for(pred.target, pred.prob, pred.classes),
        len(pred.target),
        replicates=recorded["bootstrap"]["replicates"],
        confidence=recorded["bootstrap"]["confidence"],
        seed=recorded["bootstrap"]["seed"],
    )
    for key, estimate in estimates.items():
        stored = recorded[split][key]
        for field in ("value", "lower", "upper"):
            got, want = getattr(estimate, field), stored[field]
            if not math.isclose(got, want, abs_tol=1e-9):
                raise SystemExit(f"{split} {key} {field}: recomputed {got}, metrics.json {want}")
    official = {
        "auroc": getAUC(pred.target, pred.prob, "multi-class"),
        "accuracy": getACC(pred.target, pred.prob, "multi-class"),
    }
    for key, value in official.items():
        if not math.isclose(value, estimates[key].value, abs_tol=1e-12):
            raise SystemExit(
                f"{split} {key}: MedMNIST evaluator {value}, pipeline {estimates[key].value}"
            )
    return {"pred": pred, "est": estimates}


def epoch_series(column: str) -> dict[int, float]:
    with (RUN / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {int(r["epoch"]): float(r[column]) for r in rows if r["epoch"] and r[column]}


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def numbers() -> dict[str, str]:
    recorded = json.loads((RUN / "metrics.json").read_text())
    run = json.loads((RUN / "run.json").read_text())
    baselines = yaml.safe_load((PROJECT / "baselines.yaml").read_text())["baselines"]
    test = rescore("test", recorded)
    val = rescore("val", recorded)
    pred, est = test["pred"], test["est"]
    names = pred.classes
    predicted = pred.prob.argmax(1)
    counts = confusion_matrix(pred.target, predicted, labels=range(len(names)))
    index = {SHORT[n]: i for i, n in enumerate(names)}

    best = max(baselines, key=lambda b: b["metrics"]["auroc"])
    best_acc = max(b["metrics"]["accuracy"] for b in baselines)
    nv = int(np.sum(pred.target == index["nv"]))
    val_auroc = epoch_series("val/auroc")
    val_ba = epoch_series("val/balanced_accuracy")
    late = [v for e, v in val_auroc.items() if e >= 7]
    started = datetime.fromisoformat(run["started"])
    finished = datetime.fromisoformat(run["finished"])
    minutes, seconds = divmod(int((finished - started).total_seconds()), 60)
    epochs = max(val_auroc) + 1

    out = {
        "testAuroc": fmt(est["auroc"].value),
        "testAurocLo": fmt(est["auroc"].lower),
        "testAurocHi": fmt(est["auroc"].upper),
        "testAcc": fmt(est["accuracy"].value),
        "testAccLo": fmt(est["accuracy"].lower),
        "testAccHi": fmt(est["accuracy"].upper),
        "testBacc": fmt(est["balanced_accuracy"].value),
        "testBaccLo": fmt(est["balanced_accuracy"].lower),
        "testBaccHi": fmt(est["balanced_accuracy"].upper),
        "testAuprc": fmt(est["auprc"].value),
        "valAuroc": fmt(val["est"]["auroc"].value),
        "valAurocLo": fmt(val["est"]["auroc"].lower),
        "valAurocHi": fmt(val["est"]["auroc"].upper),
        "valBacc": fmt(val["est"]["balanced_accuracy"].value),
        "valBaccLo": fmt(val["est"]["balanced_accuracy"].lower),
        "valBaccHi": fmt(val["est"]["balanced_accuracy"].upper),
        "bestPublishedAuroc": fmt(best["metrics"]["auroc"]),
        "bestPublishedAcc": fmt(best_acc),
        "gapToLower": fmt(est["auroc"].lower - best["metrics"]["auroc"]),
        "gapPoint": fmt(est["auroc"].value - best["metrics"]["auroc"], 3),
        "nTest": f"{len(pred.target):,}",
        "nVal": f"{len(val['pred'].target):,}",
        "nNevusTest": f"{nv:,}",
        "constantAcc": fmt(nv / len(pred.target), 3),
        "melAsNv": str(counts[index["mel"], index["nv"]]),
        "nvAsMel": str(counts[index["nv"], index["mel"]]),
        "akiecHit": str(counts[index["akiec"], index["akiec"]]),
        "akiecAsBkl": str(counts[index["akiec"], index["bkl"]]),
        "selectedEpoch": str(recorded["selection"]["best_epoch"]),
        "selectedValAurocLogged": fmt(val_auroc[recorded["selection"]["best_epoch"]]),
        "valAurocEpochSeven": fmt(val_auroc[7], 3),
        "valAurocLateMin": fmt(min(late), 3),
        "valAurocLateMax": fmt(max(late), 3),
        "valBaccEpochSeven": fmt(val_ba[7], 3),
        "epochs": str(epochs),
        "trainMinutes": str(minutes),
        "trainSeconds": str(seconds),
        "minutesPerEpoch": fmt((finished - started).total_seconds() / 60 / epochs, 2),
        "bootstrapReplicates": f"{recorded['bootstrap']['replicates']:,}",
        "parametersMillions": fmt(run["parameters"] / 1e6, 1),
    }
    for short, i in index.items():
        key = short.capitalize()
        name = names[i]
        n = int(np.sum(pred.target == i))
        out[f"n{key}"] = f"{n:,}"
        out[f"auroc{key}"] = fmt(est[f"auroc/{name}"].value)
        out[f"auroc{key}Lo"] = fmt(est[f"auroc/{name}"].lower)
        out[f"auroc{key}Hi"] = fmt(est[f"auroc/{name}"].upper)
        out[f"auprc{key}"] = fmt(est[f"auprc/{name}"].value)
        out[f"recall{key}"] = fmt(counts[i, i] / n, 3)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="write generated/numbers.tex")
    args = parser.parse_args()
    values = numbers()
    for key, value in values.items():
        print(f"{key}\t{value}")
    if args.write:
        lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in values.items()]
        out = HERE / "generated/numbers.tex"
        out.parent.mkdir(exist_ok=True)
        out.write_text(
            "% Written by analysis/evidence.py. Do not edit.\n" + "\n".join(lines) + "\n"
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
