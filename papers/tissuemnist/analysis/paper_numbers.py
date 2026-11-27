"""Recompute every number the TissueMNIST paper uses from the committed run files.

    uv run python papers/tissuemnist/analysis/paper_numbers.py [--out papers/tissuemnist/generated/numbers.tex]

Prints each number with its name. With --out, also writes one LaTeX macro per number.
Fails if a recomputed value disagrees with the run's metrics.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from medmnist.evaluator import getACC, getAUC

from opendiagnostics.metrics import classification as cm
from opendiagnostics.metrics.bootstrap import bootstrap
from opendiagnostics.metrics.compare import read_predictions

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "models/tissuemnist-convnext"
RUN = PROJECT / "runs/20261126-200655-ab39eb97"
SHORT = {
    "Collecting Duct, Connecting Tubule": "CDCNT",
    "Distal Convoluted Tubule": "DCT",
    "Glomerular endothelial cells": "GEC",
    "Interstitial endothelial cells": "IEC",
    "Leukocytes": "Leuk",
    "Podocytes": "Podo",
    "Proximal Tubule Segments": "PT",
    "Thick Ascending Limb": "TAL",
}


def statistic(target: np.ndarray, prob: np.ndarray, classes: int):
    def run(index: np.ndarray) -> dict[str, float]:
        t, p = target[index], prob[index]
        predicted = p.argmax(1)
        return {
            "auroc": cm.nanmean(cm.per_class(t, p, cm.auroc)),
            "auprc": cm.nanmean(cm.per_class(t, p, cm.auprc)),
            "accuracy": float(np.mean(predicted == t)),
            "balanced_accuracy": cm.balanced_accuracy(t, predicted, classes),
        }

    return run


def check(name: str, ours: float, recorded: float) -> None:
    if abs(ours - recorded) > 1e-9:
        raise SystemExit(f"{name}: recomputed {ours!r} but metrics.json has {recorded!r}")


def epoch_history() -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    with (RUN / "metrics.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["epoch"]:
                continue
            values = out.setdefault(int(row["epoch"]), {})
            for key in ("train/loss", "val/auroc", "val/accuracy"):
                if row[key]:
                    values[key] = float(row[key])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    recorded = json.loads((RUN / "metrics.json").read_text())
    run = json.loads((RUN / "run.json").read_text())
    baselines = yaml.safe_load((PROJECT / "baselines.yaml").read_text())["baselines"]
    numbers: dict[str, str] = {}

    def put(name: str, value: float | int | str, fmt: str = "{:.4f}") -> None:
        numbers[name] = value if isinstance(value, str) else fmt.format(value)

    for split in ("test", "val"):
        pred = read_predictions(RUN / f"predictions_{split}.csv")
        classes = len(pred.classes)
        estimates = bootstrap(
            statistic(pred.target, pred.prob, classes),
            len(pred.target),
            replicates=recorded["bootstrap"]["replicates"],
            seed=recorded["bootstrap"]["seed"],
        )
        for key, estimate in estimates.items():
            for part in ("value", "lower", "upper"):
                check(f"{split}/{key}/{part}", getattr(estimate, part), recorded[split][key][part])
        check(
            f"{split}/medmnist AUC",
            getAUC(pred.target, pred.prob, "multi-class"),
            estimates["auroc"].value,
        )
        check(
            f"{split}/medmnist ACC",
            getACC(pred.target, pred.prob, "multi-class"),
            estimates["accuracy"].value,
        )
        prefix = "test" if split == "test" else "val"
        for key, macro in (
            ("auroc", "Auroc"),
            ("auprc", "Auprc"),
            ("accuracy", "Acc"),
            ("balanced_accuracy", "BalAcc"),
        ):
            put(f"{prefix}{macro}", estimates[key].value)
            put(f"{prefix}{macro}Lo", estimates[key].lower)
            put(f"{prefix}{macro}Hi", estimates[key].upper)
        put(f"{prefix}N", len(pred.target), "{:,}")

        if split == "test":
            predicted = pred.prob.argmax(1)
            counts = np.bincount(pred.target, minlength=classes)
            put("testErrorPct", 100 * (1 - estimates["accuracy"].value), "{:.0f}")
            per_class = cm.per_class(pred.target, pred.prob, cm.auroc)
            for i, name in enumerate(pred.classes):
                short = SHORT[name]
                check(
                    f"test/auroc/{name}",
                    float(per_class[i]),
                    recorded["test"][f"auroc/{name}"]["value"],
                )
                put(f"auroc{short}", recorded["test"][f"auroc/{name}"]["value"])
                put(f"auroc{short}Lo", recorded["test"][f"auroc/{name}"]["lower"])
                put(f"auroc{short}Hi", recorded["test"][f"auroc/{name}"]["upper"])
                put(f"recall{short}", float(np.mean(predicted[pred.target == i] == i)), "{:.3f}")
                put(f"count{short}", int(counts[i]), "{:,}")
            index = {SHORT[n]: i for i, n in enumerate(pred.classes)}
            for name, true, said in [
                ("confDCTasCDCNT", "DCT", "CDCNT"),
                ("confDCTasTAL", "DCT", "TAL"),
                ("confPodoasGEC", "Podo", "GEC"),
                ("confGECasPodo", "GEC", "Podo"),
            ]:
                mask = pred.target == index[true]
                put(name, 100 * float(np.mean(predicted[mask] == index[said])), "{:.0f}")

    best = max(baselines, key=lambda b: b["metrics"]["auroc"])
    native = next(b for b in baselines if "native 224" in b["method"])
    put("baseBestAuroc", best["metrics"]["auroc"])
    put("baseBestAcc", best["metrics"]["accuracy"])
    put("baseNativeAuroc", native["metrics"]["auroc"])
    put("baseNativeAcc", native["metrics"]["accuracy"])
    put("marginLowerAuroc", recorded["test"]["auroc"]["lower"] - best["metrics"]["auroc"])
    put("gainPointAuroc", recorded["test"]["auroc"]["value"] - best["metrics"]["auroc"])
    put("marginLowerAcc", recorded["test"]["accuracy"]["lower"] - best["metrics"]["accuracy"])

    history = epoch_history()
    selected = recorded["selection"]["best_epoch"]
    put("selectedEpoch", selected + 1, "{}")
    put("epochs", len(history), "{}")
    put("emaValAurocFirst", history[0]["val/auroc"])
    put("emaValAurocSelected", history[selected]["val/auroc"])
    put("emaValAurocLast", history[max(history)]["val/auroc"])
    put("emaValAccSelected", history[selected]["val/accuracy"])
    put("trainLossFirst", history[0]["train/loss"], "{:.3f}")
    put("trainLossLast", history[max(history)]["train/loss"], "{:.3f}")
    put("emaMinusRawAuroc", history[selected]["val/auroc"] - recorded["val"]["auroc"]["value"])
    late = [history[e]["val/auroc"] for e in sorted(history)[-3:]]
    put("lastThreeSpread", max(late) - min(late))
    started = datetime.fromisoformat(run["started"])
    finished = datetime.fromisoformat(run["finished"])
    put("runMinutes", (finished - started).total_seconds() / 60, "{:.0f}")
    put("parameters", run["parameters"], "{:,}")
    put("bootReplicates", recorded["bootstrap"]["replicates"], "{:,}")

    for name, value in numbers.items():
        print(f"{name}\t{value}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"\\newcommand{{\\{name}}}{{{value.replace(',', '{,}')}}}"
            for name, value in numbers.items()
        ]
        args.out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
