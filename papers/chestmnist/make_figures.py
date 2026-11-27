"""Draw the paper's figures from the committed ChestMNIST run files.

    uv run python papers/chestmnist/make_figures.py

Reads metrics.csv and metrics.json from the run directory and writes PDFs to
papers/chestmnist/figures/.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / "models/chestmnist-convnext"
RUN = PROJECT / "runs/20261126-165951-ab39eb97"
OUT = HERE / "figures"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#52514e"

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
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.5,
    }
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def series(rows: list[dict[str, str]], column: str) -> tuple[np.ndarray, np.ndarray]:
    """Values of ``column`` by epoch, numbered from 1."""
    points = [(int(r["epoch"]) + 1, float(r[column])) for r in rows if r[column] and r["epoch"]]
    epochs, values = zip(*points, strict=True)
    return np.array(epochs), np.array(values)


def training_curves(metrics: dict) -> None:
    rows = read_rows(RUN / "metrics.csv")
    best = metrics["selection"]["best_epoch"] + 1
    saved = metrics["val"]["auroc"]["value"]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.4, 3.0), sharex=True)
    epochs, loss = series(rows, "train/loss")
    top.plot(epochs, loss, color=BLUE, lw=1.5, marker="o", ms=3)
    top.set_ylabel("Training loss (BCE)")
    epochs, auroc = series(rows, "val/auroc")
    bottom.plot(epochs, auroc, color=BLUE, lw=1.5, marker="o", ms=3, label="EMA weights, logged")
    bottom.plot([best], [saved], ls="none", marker="D", ms=4, color=ORANGE, label="Saved weights")
    for ax in (top, bottom):
        ax.axvline(best, color=GRAY, lw=0.7, ls="--")
    bottom.set_ylabel("Validation macro AUROC")
    bottom.set_xlabel("Epoch")
    bottom.set_xticks(range(1, len(epochs) + 1))
    bottom.legend(loc="lower right", frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "training_curves.pdf")
    plt.close(fig)


def lr_schedules() -> None:
    """Our logged learning rate against the MedMNIST+ schedule, recomputed from its code.

    MedMNIST+ (train.py at commit b67b772 of sdoerrich97/rethinking-model-prototyping-
    MedMNISTPlus) uses timm's CosineLRScheduler with t_initial=100, lr_min=0, no warmup, and
    calls scheduler.step(epoch=e) after epoch e, so epoch e+1 trains at lr(e).
    """
    rows = read_rows(RUN / "metrics.csv")
    ours = [float(r["lr-AdamW"]) for r in rows if r["lr-AdamW"]]
    base, cycle, shown = 1e-4, 100, 30

    def medmnist_plus(epoch: int) -> float:
        t = max(epoch - 1, 0)
        return 0.5 * base * (1 + np.cos(np.pi * t / cycle))

    theirs = [medmnist_plus(e) for e in range(shown)]
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ax.plot(range(1, shown + 1), theirs, color=ORANGE, lw=1.5, label="MedMNIST+ (first 30 of 100)")
    ax.plot(
        range(1, len(ours) + 1), ours, color=BLUE, lw=1.5, marker="o", ms=3, label="Ours (logged)"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_ylim(0, 1.1e-4)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="center right", frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "lr_schedules.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    metrics = json.loads((RUN / "metrics.json").read_text())
    training_curves(metrics)
    lr_schedules()


if __name__ == "__main__":
    main()
