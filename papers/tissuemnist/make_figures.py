"""Draw the paper's figures from the committed TissueMNIST run files.

    uv run python papers/tissuemnist/make_figures.py

Writes training_curves.pdf and per_class_auroc.pdf to papers/tissuemnist/figures/, and
prints the test numbers it recomputed (AUROC, accuracy, per-class recall and the main
confusions) so they can be checked against metrics.json and the text.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "models/tissuemnist-convnext/runs/20261126-200655-ab39eb97"
OUT = Path(__file__).resolve().parent / "figures"

SERIES = "#2a78d6"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dcdbd6"
SHORT = {
    "Collecting Duct, Connecting Tubule": "CD/CNT",
    "Distal Convoluted Tubule": "DCT",
    "Glomerular endothelial cells": "GEC",
    "Interstitial endothelial cells": "IEC",
    "Leukocytes": "Leuk.",
    "Podocytes": "Podo.",
    "Proximal Tubule Segments": "PT",
    "Thick Ascending Limb": "TAL",
}

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
    }
)


def history() -> dict[str, dict[int, float]]:
    """Per-epoch training loss and validation metrics from Lightning's CSVLogger file."""
    columns = ("train/loss", "val/auroc", "val/accuracy")
    out: dict[str, dict[int, float]] = {c: {} for c in columns}
    with (RUN / "metrics.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["epoch"]:
                continue
            for column in columns:
                if row[column]:
                    out[column][int(row["epoch"])] = float(row[column])
    return out


def predictions() -> tuple[list[str], np.ndarray, np.ndarray]:
    path = RUN / "predictions_test.csv"
    with path.open(newline="") as handle:
        header = next(csv.reader(handle))
    names = [h.removeprefix("prob/") for h in header[3:]]
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(2, len(header)))
    return names, data[:, 0].astype(np.int64), data[:, 1:]


def training_curves(selected: int, saved: dict[str, float]) -> None:
    """Logged validation used the EMA weights; ``saved`` holds the selected checkpoint's raw
    weights rescored on validation, drawn as open squares at the selected epoch."""
    curves = history()
    panels = [
        ("train/loss", "Training loss"),
        ("val/auroc", "Validation AUROC"),
        ("val/accuracy", "Validation accuracy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.9), constrained_layout=True)
    for ax, (column, label) in zip(axes, panels, strict=True):
        epochs = sorted(curves[column])
        values = [curves[column][e] for e in epochs]
        shown = [e + 1 for e in epochs]
        ax.plot(shown, values, color=SERIES, linewidth=1.5, marker="o", markersize=3.5)
        ax.axvline(selected + 1, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_title(label, fontsize=8, color=TEXT, loc="left")
        ax.set_xticks(shown)
        if column in saved:
            ax.plot(
                selected + 1,
                saved[column],
                marker="s",
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor=TEXT,
                linestyle="none",
            )
    axes[1].annotate(
        "selected",
        xy=(selected + 1, curves["val/auroc"][selected]),
        xytext=(-38, -22),
        textcoords="offset points",
        color=MUTED,
        fontsize=7,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6},
    )
    fig.savefig(OUT / "training_curves.pdf")
    plt.close(fig)


def per_class_auroc(names: list[str], target: np.ndarray, prob: np.ndarray, test: dict) -> None:
    values = np.array([roc_auc_score(target == k, prob[:, k]) for k in range(len(names))])
    lower = np.array([test[f"auroc/{n}"]["lower"] for n in names])
    upper = np.array([test[f"auroc/{n}"]["upper"] for n in names])
    counts = np.bincount(target, minlength=len(names))
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(3.5, 2.3), constrained_layout=True)
    rows = np.arange(len(names))
    ax.errorbar(
        values[order],
        rows,
        xerr=[values[order] - lower[order], upper[order] - values[order]],
        fmt="o",
        color=SERIES,
        markersize=4,
        elinewidth=1.2,
        capsize=0,
    )
    macro = float(values.mean())
    ax.axvline(macro, color=MUTED, linewidth=0.8, linestyle="--")
    ax.text(macro, len(names) - 0.4, f" macro {macro:.4f}", color=MUTED, fontsize=7, va="bottom")
    ax.set_yticks(rows, [f"{SHORT[names[i]]} (n={counts[i]:,})" for i in order])
    ax.set_xlabel("One-vs-rest test AUROC")
    ax.set_ylim(-0.6, len(names) - 0.1)
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "per_class_auroc.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    metrics = json.loads((RUN / "metrics.json").read_text())
    names, target, prob = predictions()
    saved = {
        "val/auroc": metrics["val"]["auroc"]["value"],
        "val/accuracy": metrics["val"]["accuracy"]["value"],
    }
    training_curves(metrics["selection"]["best_epoch"], saved)
    per_class_auroc(names, target, prob, metrics["test"])
    macro = roc_auc_score(target, prob, multi_class="ovr", average="macro")
    accuracy = accuracy_score(target, prob.argmax(1))
    print(f"test images {len(target)}, macro AUROC {macro:.6f}, accuracy {accuracy:.6f}")
    matrix = confusion_matrix(target, prob.argmax(1), labels=range(len(names)))
    share = matrix / matrix.sum(1, keepdims=True)
    for k, name in enumerate(names):
        top = np.argsort(share[k])[::-1][:3]
        others = ", ".join(f"{SHORT[names[j]]} {share[k, j]:.3f}" for j in top if j != k)
        print(f"{SHORT[name]}: n={matrix[k].sum()}, recall {share[k, k]:.3f}; to {others}")
    print(
        f"metrics.json: AUROC {metrics['test']['auroc']['value']:.6f}, "
        f"accuracy {metrics['test']['accuracy']['value']:.6f}"
    )


if __name__ == "__main__":
    main()
