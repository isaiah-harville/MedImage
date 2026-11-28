"""Draw the paper's figures from the committed ChestMNIST run files and baselines.yaml.

    uv run --with matplotlib python -I papers/chestmnist/make_figures.py

Run analysis/paper_numbers.py first: it checks every run against its predictions and writes
generated/seeds.json, which holds the image- and patient-level intervals drawn here. Writes
comparison.pdf and training_curves.pdf to papers/chestmnist/figures/.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1].parent
PAPER = Path(__file__).resolve().parent
PROJECT = ROOT / "models/chestmnist-convnext"
RUNS = ["20261126-165951-ab39eb97", "20261127-142008-a35c5dae", "20261127-150930-405f7138"]
OUT = PAPER / "figures"
PUBLISHED = [
    ("BioFuse", "BioFuse [Hos]"),
    ("DINOv2 ViT-B/14", "DINOv2 ViT-B/14 [Wu]"),
    ("MambaVision-B", "MambaVision-B [Far]"),
    ("DenseNet-121, end-to-end, native 224", "DenseNet-121 [M+]"),
    ("ConvNeXt-T", "ConvNeXt-T [Far]"),
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


def comparison() -> None:
    seeds = json.loads((PAPER / "generated/seeds.json").read_text())
    baselines = yaml.safe_load((PROJECT / "baselines.yaml").read_text())["baselines"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    labels = []
    y = 0
    for seed in seeds.values():
        for offset, (key, marker, fill) in enumerate(
            (("image", "o", OURS), ("patient", "o", "white"))
        ):
            value, lower, upper = seed[key]
            ax.errorbar(
                value,
                y - 0.17 + 0.34 * offset,
                xerr=[[value - lower], [upper - value]],
                fmt=marker,
                color=OURS,
                mfc=fill,
                ms=3.5,
                capsize=2,
                lw=0.9,
            )
        labels.append(f"ConvNeXt-S seed {seed['seed']}")
        y += 1
    values = [s["image"][0] for s in seeds.values()]
    mean = sum(values) / len(values)
    sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
    ax.errorbar(mean, y, xerr=sd, fmt="D", color=OURS, ms=3.5, capsize=2, lw=0.9)
    labels.append("ConvNeXt-S mean")
    y += 1
    best = max(b["metrics"]["auroc"] for b in baselines)
    for prefix, label in PUBLISHED:
        metrics = next(b for b in baselines if b["method"].startswith(prefix))["metrics"]
        ax.errorbar(
            metrics["auroc"],
            y,
            xerr=metrics.get("auroc_sd", 0),
            fmt="s",
            color=THEIRS,
            mfc="white",
            ms=3.5,
            capsize=2,
            lw=0.9,
        )
        labels.append(label)
        y += 1
    ax.axvline(best, color=THEIRS, lw=0.6, ls="--")
    ax.set_xlabel("Test macro AUROC")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    ax.set_xlim(0.795, 0.855)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "comparison.pdf")
    plt.close(fig)


def history(run_id: str) -> tuple[int, dict, dict[int, dict[str, float]]]:
    path = PROJECT / "runs" / run_id
    out: dict[int, dict[str, float]] = {}
    with (path / "metrics.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["epoch"]:
                values = out.setdefault(int(row["epoch"]), {})
                for key in ("train/loss", "val/auroc"):
                    if row[key]:
                        values[key] = float(row[key])
    seed = json.loads((path / "run.json").read_text())["seed"]
    return seed, json.loads((path / "metrics.json").read_text()), out


def curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.7))
    for run_id, (line, marker) in zip(RUNS, STYLES, strict=True):
        seed, metrics, epochs = history(run_id)
        x = [e + 1 for e in sorted(epochs)]
        for ax, key in ((axes[0], "train/loss"), (axes[1], "val/auroc")):
            ax.plot(
                x,
                [epochs[e][key] for e in sorted(epochs)],
                line,
                marker=marker,
                ms=2.5,
                lw=0.9,
                color=OURS,
                label=f"seed {seed}",
            )
        selected = metrics["selection"]["best_epoch"]
        axes[1].plot(
            selected + 1.2,
            metrics["val"]["auroc"]["value"],
            marker,
            ms=4,
            mfc="white",
            color="black",
        )
    axes[0].set_ylabel("Training loss")
    axes[1].set_ylabel("Validation AUROC")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(1, 11, 3))
    axes[0].legend(fontsize=7, handlelength=2.2)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(OUT / "training_curves.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    comparison()
    curves()


if __name__ == "__main__":
    main()
