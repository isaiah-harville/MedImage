"""Draw the paper's figures from generated/seeds.json.

    uv run --with matplotlib python -I papers/synapsemnist3d/make_figures.py

Run analysis/paper_numbers.py first: it checks every run against its predictions and writes
generated/seeds.json. Writes seeds.pdf and training_curves.pdf to papers/synapsemnist3d/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PAPER = Path(__file__).resolve().parent
OUT = PAPER / "figures"
ARMS = (
    ("Scratch", "ResNet-18, scratch", "#0072b2", "o", "-"),
    ("Kinetics", "R3D-18, Kinetics", "#d55e00", "s", "--"),
)
PUBLISHED = (
    ("R3D-18, Kinetics-400 pretrained, fine-tuned (64³)", "TopoCAM's R3D-18", (0, (4, 2))),
    ("DINOv3 ViT-L/16, fully fine-tuned, 3D baseline (mean of 5 runs)", "PlaneCycle 3D ViT-L", ":"),
)

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


def seeds(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for offset, (arm, label, color, marker, _) in enumerate(ARMS):
        runs = [r for r in data["runs"].values() if r["arm"] == arm]
        for run in runs:
            auroc = run["test"]["auroc"]
            ax.errorbar(
                run["seed"] - 0.12 + 0.24 * offset,
                auroc["value"],
                yerr=[[auroc["value"] - auroc["lower"]], [auroc["upper"] - auroc["value"]]],
                fmt=marker,
                color=color,
                ms=3.5,
                capsize=2,
                lw=0.9,
                label=label if run["seed"] == 0 else None,
            )
        pooled = data["pooled"][arm]
        ax.errorbar(
            5.0 + 0.24 * offset - 0.12,
            pooled["value"],
            yerr=[[pooled["value"] - pooled["lower"]], [pooled["upper"] - pooled["value"]]],
            fmt="D",
            color=color,
            mfc=color if offset == 0 else "white",
            ms=4,
            capsize=2,
            lw=0.9,
            label=f"{label.split(',')[0]} mean",
        )
    for method, label, style in PUBLISHED:
        ax.axhline(
            data["baselines"][method]["auroc"], color="#444444", lw=0.8, ls=style, label=label
        )
    ax.set_xticks(range(6), ["0", "1", "2", "3", "4", "mean"])
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Test AUROC (352 volumes)")
    ax.set_ylim(0.94, 1.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "seeds.pdf")
    plt.close(fig)


def training_curves(data: dict) -> None:
    fig, grid = plt.subplots(
        2, 2, figsize=(7.16, 3.6), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
    )
    styles = ("-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)))
    markers = ("o", "s", "^", "v", "D")
    for column, (arm, label, color, _, _) in enumerate(ARMS):
        runs = sorted(
            (r for r in data["runs"].values() if r["arm"] == arm), key=lambda r: r["seed"]
        )
        for ax, low in zip(grid[:, column], (0.4, 0.93), strict=True):
            for run, style, marker in zip(runs, styles, markers, strict=True):
                epochs = [h["epoch"] for h in run["history"]]
                values = [h["val_auroc"] for h in run["history"]]
                ax.plot(epochs, values, color=color, lw=0.8, ls=style, label=f"seed {run['seed']}")
                raw = run["val"]["auroc"]["value"]
                ax.plot([run["epoch"]] * 2, [run["val_auroc_ema"], raw], color="black", lw=0.6)
                ax.plot(run["epoch"], run["val_auroc_ema"], marker=marker, color="black", ms=3.5)
                ax.plot(run["epoch"], raw, marker=marker, mfc="white", color="black", ms=3.5)
            ax.set_ylim(low, 1.0)
        grid[0, column].set_title(label, fontsize=8)
        grid[1, column].set_xlabel("Epoch")
    handles = [
        Line2D([], [], color="#444444", lw=0.8, ls=style, marker=marker, mfc="black", ms=3.5)
        for style, marker in zip(styles, markers, strict=True)
    ]
    handles += [
        Line2D([], [], color="black", marker="o", ls="none", ms=3.5),
        Line2D([], [], color="black", marker="o", mfc="white", ls="none", ms=3.5),
    ]
    names = [f"seed {s}" for s in range(len(styles))] + ["EMA, selected", "raw, tested"]
    fig.legend(handles, names, loc="lower center", fontsize=7, ncol=7)
    for ax in grid[:, 0]:
        ax.set_ylabel("Validation AUROC")
    fig.tight_layout(pad=0.3, rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "training_curves.pdf")
    plt.close(fig)


def main() -> None:
    data = json.loads((PAPER / "generated/seeds.json").read_text())
    OUT.mkdir(exist_ok=True)
    seeds(data)
    training_curves(data)


if __name__ == "__main__":
    main()
