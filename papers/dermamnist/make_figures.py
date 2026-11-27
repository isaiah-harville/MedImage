"""Draw the paper's figures from the committed DermaMNIST run files.

    uv run python papers/dermamnist/make_figures.py

Reads metrics.csv, metrics.json, and predictions_test.csv from the run directory and writes
PDFs to papers/dermamnist/figures/.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1] / "models/dermamnist-convnext/runs/20261126-141911-ab39eb97"
OUT = HERE / "figures"
SHORT = {
    "actinic keratoses and intraepithelial carcinoma": "akiec",
    "basal cell carcinoma": "bcc",
    "benign keratosis-like lesions": "bkl",
    "dermatofibroma": "df",
    "melanoma": "mel",
    "melanocytic nevi": "nv",
    "vascular lesions": "vasc",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def series(rows: list[dict[str, str]], column: str) -> tuple[np.ndarray, np.ndarray]:
    points = [(float(r["epoch"]), float(r[column])) for r in rows if r[column] and r["epoch"]]
    epochs, values = zip(*points, strict=True)
    return np.array(epochs), np.array(values)


def training_curves() -> None:
    rows = read_rows(RUN / "metrics.csv")
    best = json.loads((RUN / "metrics.json").read_text())["selection"]["best_epoch"]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.4, 3.2), sharex=True)
    epochs, loss = series(rows, "train/loss")
    top.plot(epochs, loss, color="#1f4e79", lw=1.2)
    top.set_ylabel("Training loss")
    for column, label, color in [
        ("val/auroc", "Macro AUROC", "#1f4e79"),
        ("val/accuracy", "Accuracy", "#7f7f7f"),
        ("val/balanced_accuracy", "Balanced accuracy", "#c55a11"),
    ]:
        epochs, values = series(rows, column)
        bottom.plot(epochs, values, label=label, color=color, lw=1.2)
    for ax in (top, bottom):
        ax.axvline(best, color="black", lw=0.7, ls="--")
    bottom.text(best + 0.6, 0.05, f"selected\nepoch {best}", fontsize=7)
    bottom.set_ylabel("Validation")
    bottom.set_xlabel("Epoch")
    bottom.set_ylim(0, 1.0)
    bottom.legend(loc="lower center", bbox_to_anchor=(0.42, 0.0), frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "training_curves.pdf")
    plt.close(fig)


def test_confusion() -> None:
    rows = read_rows(RUN / "predictions_test.csv")
    columns = [c for c in rows[0] if c.startswith("prob/")]
    names = [SHORT[c.removeprefix("prob/")] for c in columns]
    target = np.array([int(r["target"]) for r in rows])
    prob = np.array([[float(r[c]) for c in columns] for r in rows])
    counts = confusion_matrix(target, prob.argmax(1), labels=range(len(names)))
    share = counts / counts.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if counts[i, j]:
                color = "white" if share[i, j] > 0.6 else "black"
                ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=7, color=color)
    ticks = range(len(names))
    ax.set_xticks(ticks, names)
    ax.set_yticks(ticks, [f"{n} ({counts[i].sum()})" for i, n in enumerate(names)])
    ax.set_xlabel("Predicted class (argmax)")
    ax.set_ylabel("True class (test count)")
    ax.spines[:].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "confusion_test.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    training_curves()
    test_confusion()


if __name__ == "__main__":
    main()
