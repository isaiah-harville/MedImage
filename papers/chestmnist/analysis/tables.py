"""The LaTeX tables of the ChestMNIST paper, from values paper_numbers.py has checked."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

PAPER = Path(__file__).resolve().parents[1]


def baseline_cell(metrics: dict[str, float], key: str) -> str:
    if key not in metrics:
        return "n.r."
    value = metrics[key]
    digits = 4 if len(str(value)) > 5 else 3
    sd = metrics.get(f"{key}_sd")
    text = f"{value:.{digits}f}"
    return f"{text} $\\pm$ {sd:.{digits}f}" if sd is not None else text


def results_table(
    seeds: dict[str, dict[str, Any]], numbers: dict[str, str], baselines: list[dict]
) -> str:
    rows = []
    for seed in seeds.values():
        image, patient = seed["image"], seed["patient"]
        rows.append(
            f"ConvNeXt-S, seed {seed['seed']} (ours) & native 224 & 1 & {image[0]:.4f} & "
            f"[{image[1]:.4f}, {image[2]:.4f}] & [{patient[1]:.4f}, {patient[2]:.4f}] & "
            f"{seed['accuracy']:.4f} & this work \\\\"
        )
    rows.append(
        f"ConvNeXt-S, mean (ours) & native 224 & 3 & \\textbf{{{numbers['testAurocMean']}}} $\\pm$ "
        f"{numbers['testAurocSD']} & & & {numbers['testAccMean']} & this work \\\\"
    )
    rows.append("\\midrule")
    with (PAPER / "analysis/baseline_rows.tsv").open(newline="") as handle:
        labels = list(csv.DictReader(handle, delimiter="\t"))
    for b in baselines:
        row = next((x for x in labels if b["method"].startswith(x["method_prefix"])), None)
        if row is None:
            raise SystemExit(f"baselines.yaml row {b['method']!r} has no line in baseline_rows.tsv")
        rows.append(
            f"{row['label']} & {row['input']} & {row['runs']} & {baseline_cell(b['metrics'], 'auroc')} & & & "
            f"{baseline_cell(b['metrics'], 'accuracy')} & \\cite{{{row['cite']}}}, {row['where']} \\\\"
        )
    rows.append("\\midrule")
    rows.append(
        f"Every label negative & -- & -- & 0.5 & & & {numbers['allNegAcc']} & computed \\\\"
    )
    body = "\n".join(rows)
    return (
        "\\begin{table*}[t]\n"
        "\\caption{ChestMNIST test results on the official split of 22,433 images. AUROC is the macro mean over "
        "14 labels; ACC is label accuracy at a 0.5 threshold. For our seeds, Image CI and Patient CI are 95\\% "
        "percentile bootstrap intervals that resample test images or test patients; the mean row gives the mean "
        "$\\pm$ SD over the three seeds. Published rows give the mean, with $\\pm$ SD over runs where the source "
        "reports it. Runs: seeds or trials averaged (n.s.: not stated; n.r.: not reported). Input: native 224 is the "
        "MedMNIST+ 224-pixel release; 28 $\\rightarrow$ 224 is a 28-pixel image enlarged.}\n"
        "\\label{tab:main}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{@{}llllllll@{}}\n\\toprule\n"
        "Method & Input & Runs & AUROC & Image CI & Patient CI & ACC & Source \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table*}}\n"
    )


def per_label_table(rows: list[str]) -> str:
    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n"
        "\\caption{Which findings does the model rank worst? Test prevalence and one-vs-rest AUROC for each "
        "label, as the mean and the range over the three seeds, sorted by AUROC.}\n"
        "\\label{tab:perlabel}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}lrll@{}}\n\\toprule\n"
        "Label & Prev.\\ (\\%) & AUROC & range \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def subsets_table(rows: list[str], commit: str) -> str:
    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n"
        "\\caption{How does the same recipe do on the other MedMNIST 2D subsets? Test AUROC of the first run "
        f"(seed 0, all at commit \\textnormal{{\\texttt{{{commit}}}}}) on each subset at 224 pixels, with its 95\\% bootstrap interval over "
        "test images, against the best published value in that project's \\textnormal{\\texttt{baselines.yaml}}. Only the "
        "ChestMNIST, DermaMNIST, and TissueMNIST lists have had a literature review. Our CI: whether our interval "
        "lies above or below that value, or contains it.}\n"
        "\\label{tab:subsets}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{@{}llll@{}}\n\\toprule\n"
        "Subset & Ours [95\\% CI] & Published & Our CI \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )
