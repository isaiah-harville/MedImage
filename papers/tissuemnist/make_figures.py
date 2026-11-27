"""Draw the paper's figures from the committed TissueMNIST run files and baselines.yaml.

    uv run --with matplotlib python -I papers/tissuemnist/make_figures.py

Writes comparison.pdf and training_curves.pdf to papers/tissuemnist/figures/. The values
come from each run's metrics.json and metrics.csv, which analysis/paper_numbers.py checks
against the committed predictions.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "models/tissuemnist-convnext"
RUNS = ["20261126-200655-ab39eb97", "20261127-113245-f5bc0be9", "20261127-125653-a35c5dae"]
OUT = Path(__file__).resolve().parent / "figures"
PUBLISHED = [
    ("ViT-B/16, end-to-end", "ViT-B/16 [Wu]"),
    ("EfficientNet-B4, end", "EffNet-B4 [Wu]"),
    ("DenseNet-121 from", "DenseNet-121 [Arman]"),
    ("CytoECA-Net", "CytoECA-Net [Arman]"),
    ("DenseNet-121, end-to-end, native 128", "DenseNet-121 128 [M+]"),
    ("DenseNet-121, end-to-end, native 224", "DenseNet-121 224 [M+]"),
]
OURS = "#0072b2"
THEIRS = "#d55e00"
STYLES = [("-", "o"), ("--", "s"), (":", "^")]

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "legend.frameon": False,
    }
)


def runs() -> list[dict]:
    out = []
    for run_id in RUNS:
        path = PROJECT / "runs" / run_id
        history: dict[int, dict[str, float]] = {}
        with (path / "metrics.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["epoch"]:
                    values = history.setdefault(int(row["epoch"]), {})
                    for key in ("train/loss", "val/auroc"):
                        if row[key]:
                            values[key] = float(row[key])
        out.append(
            {
                "seed": json.loads((path / "run.json").read_text())["seed"],
                "metrics": json.loads((path / "metrics.json").read_text()),
                "history": history,
            }
        )
    return out


def comparison(scored: list[dict]) -> None:
    baselines = yaml.safe_load((PROJECT / "baselines.yaml").read_text())["baselines"]
    rows = []
    for prefix, label in PUBLISHED:
        row = next(b for b in baselines if b["method"].startswith(prefix))
        rows.append((label, row["metrics"]))
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.5), sharey=True)
    for ax, key, title in (
        (axes[0], "auroc", "Test AUROC"),
        (axes[1], "accuracy", "Test accuracy"),
    ):
        labels = []
        y = 0
        for run in scored:
            estimate = run["metrics"]["test"][key]
            ax.errorbar(
                estimate["value"],
                y,
                xerr=[
                    [estimate["value"] - estimate["lower"]],
                    [estimate["upper"] - estimate["value"]],
                ],
                fmt="o",
                color=OURS,
                ms=3.5,
                capsize=2,
                lw=0.9,
            )
            labels.append(f"ConvNeXt-S seed {run['seed']}")
            y += 1
        values = [run["metrics"]["test"][key]["value"] for run in scored]
        mean = sum(values) / len(values)
        sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        ax.errorbar(mean, y, xerr=sd, fmt="D", color=OURS, ms=3.5, capsize=2, lw=0.9)
        labels.append("ConvNeXt-S mean")
        y += 1
        for label, metrics in rows:
            ax.errorbar(
                metrics[key],
                y,
                xerr=metrics.get(f"{key}_sd", 0),
                fmt="s",
                color=THEIRS,
                mfc="white",
                ms=3.5,
                capsize=2,
                lw=0.9,
            )
            labels.append(label)
            y += 1
        ax.axvline(rows[0][1][key], color=THEIRS, lw=0.6, ls="--")
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_yticks(range(len(labels)), labels)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.951, 0.966)
    axes[1].set_xlim(0.735, 0.79)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(OUT / "comparison.pdf")
    plt.close(fig)


def curves(scored: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.7))
    for run, (line, marker) in zip(scored, STYLES, strict=True):
        epochs = sorted(run["history"])
        x = [e + 1 for e in epochs]
        label = f"seed {run['seed']}"
        axes[0].plot(
            x,
            [run["history"][e]["train/loss"] for e in epochs],
            line,
            marker=marker,
            ms=2.5,
            lw=0.9,
            color=OURS,
            label=label,
        )
        axes[1].plot(
            x,
            [run["history"][e]["val/auroc"] for e in epochs],
            line,
            marker=marker,
            ms=2.5,
            lw=0.9,
            color=OURS,
            label=label,
        )
        selected = run["metrics"]["selection"]["best_epoch"]
        axes[1].plot(
            selected + 1.15,
            run["metrics"]["val"]["auroc"]["value"],
            marker,
            ms=3.5,
            mfc="white",
            color=THEIRS,
        )
    axes[0].set_ylabel("Training loss")
    axes[1].set_ylabel("Validation AUROC")
    axes[1].set_xlim(2.7, 8.3)
    axes[1].set_ylim(0.955, 0.966)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(1, 9))
    axes[1].set_xticks(range(3, 9))
    axes[0].legend(fontsize=7, handlelength=2.2)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(OUT / "training_curves.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    scored = runs()
    comparison(scored)
    curves(scored)


if __name__ == "__main__":
    main()
