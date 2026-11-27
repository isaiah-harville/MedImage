"""Stage 1 checks on the ChestMNIST run: recompute the paper's numbers from the run files.

uv run python papers/chestmnist/vet.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from medmnist.evaluator import getACC, getAUC
from sklearn.metrics import roc_auc_score

from opendiagnostics.metrics.bootstrap import bootstrap

RUN = (
    Path(__file__).resolve().parents[2] / "models/chestmnist-convnext/runs/20261126-165951-ab39eb97"
)


def load(split: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(RUN / f"predictions_{split}.csv")
    target = frame.filter(like="target/").to_numpy()
    prob = frame.filter(like="prob/").to_numpy()
    return target, prob


def macro_auroc(target: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean([roc_auc_score(target[:, j], prob[:, j]) for j in range(target.shape[1])]))


def main() -> None:
    metrics = json.loads((RUN / "metrics.json").read_text())
    target, prob = load("test")
    n = len(target)
    print("test images", n, "labels", target.shape[1])
    print(
        "sklearn macro AUROC", macro_auroc(target, prob), "metrics.json", metrics["test"]["auroc"]
    )
    print("medmnist getAUC", getAUC(target, prob, "multi-label, binary-class"))
    print("medmnist getACC", getACC(target, prob, "multi-label, binary-class"))
    print("all-negative ACC", 1 - target.mean())
    print(
        "predicted positive rate at 0.5", (prob > 0.5).mean(), "label positive rate", target.mean()
    )
    prevalence = target.mean(axis=0)
    print("prevalence min max", prevalence.min(), prevalence.max())

    def stat(index: np.ndarray) -> dict[str, float]:
        return {"auroc": macro_auroc(target[index], prob[index])}

    print("image bootstrap", bootstrap(stat, n, replicates=1000, seed=0)["auroc"])

    adjacent = [np.corrcoef(prob[:-1, j], prob[1:, j])[0, 1] for j in range(prob.shape[1])]
    rng = np.random.default_rng(0)
    shuffled = prob[rng.permutation(n)]
    random_pairs = [
        np.corrcoef(shuffled[:-1, j], shuffled[1:, j])[0, 1] for j in range(prob.shape[1])
    ]
    print(
        "mean adjacent-row correlation of scores",
        np.mean(adjacent),
        "shuffled",
        np.mean(random_pairs),
    )
    same_labels = (target[:-1] == target[1:]).all(axis=1) & (target[:-1].sum(axis=1) > 0)
    print("adjacent rows with identical nonempty label sets", same_labels.mean())

    for block in (8, 64, 512):
        groups = [str(i // block) for i in range(n)]
        print(
            f"block bootstrap, {block} consecutive images",
            bootstrap(stat, n, groups=groups, replicates=1000, seed=0)["auroc"],
        )


if __name__ == "__main__":
    main()
