"""SVG figures for run pages, drawn with matplotlib (in the ``docs`` dependency group)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import metrics as skm

from opendiagnostics.metrics import classification as cm
from opendiagnostics.metrics.compare import read_predictions


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.hashsalt": "opendiagnostics", "svg.fonttype": "none", "font.size": 9})
    return plt


def _save(fig: Any, path: Path) -> None:
    # A fixed date and hash salt keep the SVG byte-identical across rebuilds.
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None, "Creator": None})


def training_curves(history: list[dict[str, Any]], monitor: str, path: Path) -> None:
    plt = _pyplot()
    epochs = [h["epoch"] for h in history]
    fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3))
    left.plot(epochs, [h.get("train_loss", np.nan) for h in history], marker="o")
    left.set(xlabel="epoch", ylabel="training loss", title="Training loss")
    values = [h["val"].get(monitor, np.nan) for h in history]
    right.plot(epochs, values, marker="o", color="C1")
    best = int(np.nanargmax(values)) if np.isfinite(values).any() else None
    if best is not None:
        right.axvline(epochs[best], color="0.6", linestyle="--", linewidth=1)
    right.set(xlabel="epoch", ylabel=monitor, title=f"Validation {monitor}")
    for ax in (left, right):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def roc_and_pr(predictions_csv: Path, path: Path) -> bool:
    """ROC and precision-recall curves for a binary run. Returns False for other tasks."""
    with predictions_csv.open(newline="") as handle:
        if "target" not in next(csv.reader(handle), []):
            return False
    predictions = read_predictions(predictions_csv)
    if predictions.prob.shape[1] != 1:
        return False
    plt = _pyplot()
    y, score = predictions.target.astype(bool), predictions.prob[:, 0]
    fpr, tpr, _ = skm.roc_curve(y, score)
    precision, recall, _ = skm.precision_recall_curve(y, score)
    fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3.4))
    left.plot(fpr, tpr, label=f"AUROC {cm.auroc(y, score):.4f}")
    left.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    left.set(xlabel="1 - specificity", ylabel="sensitivity", title="ROC (test)")
    right.step(recall, precision, where="post", label=f"AUPRC {cm.auprc(y, score):.4f}")
    right.axhline(
        y.mean(), color="0.6", linestyle="--", linewidth=1, label=f"prevalence {y.mean():.3f}"
    )
    right.set(xlabel="recall", ylabel="precision", title="Precision-recall (test)", ylim=(0, 1.02))
    for ax in (left, right):
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right" if ax is left else "lower left", frameon=False)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)
    return True
