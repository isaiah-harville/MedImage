"""Recompute every number the TissueMNIST paper uses from the committed run files.

    uv run python -I papers/tissuemnist/analysis/paper_numbers.py

Scores each of the three seeds from its committed predictions, checks every value against
the run's metrics.json and the medmnist evaluator, and fails on any mismatch. Prints each
number with its name, and writes generated/numbers.tex (one LaTeX macro per number) and the
results and per-class tables in generated/tables/.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from medmnist.evaluator import getACC, getAUC
from scipy import stats

from opendiagnostics.metrics import classification as cm
from opendiagnostics.metrics.bootstrap import Estimate, bootstrap
from opendiagnostics.metrics.compare import read_predictions

ROOT = Path(__file__).resolve().parents[3]
PAPER = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "models/tissuemnist-convnext"
RUNS = {
    "Zero": "20261126-200655-ab39eb97",
    "One": "20261127-113245-f5bc0be9",
    "Two": "20261127-125653-a35c5dae",
}
SHORT = {
    "Collecting Duct, Connecting Tubule": ("CDCNT", "CD/CNT"),
    "Distal Convoluted Tubule": ("DCT", "DCT"),
    "Glomerular endothelial cells": ("GEC", "GEC"),
    "Interstitial endothelial cells": ("IEC", "IEC"),
    "Leukocytes": ("Leuk", "Leukocytes"),
    "Podocytes": ("Podo", "Podocytes"),
    "Proximal Tubule Segments": ("PT", "PT"),
    "Thick Ascending Limb": ("TAL", "TAL"),
}
CONFUSIONS = [
    ("confDCTasCDCNT", "DCT", "CDCNT"),
    ("confDCTasTAL", "DCT", "TAL"),
    ("confPodoasGEC", "Podo", "GEC"),
    ("confGECasPodo", "GEC", "Podo"),
]
DEFAULTED_KEYS = {
    "model.init_args.class_weight": None,
    "data.init_args.variant": None,
    "data.init_args.folds": 5,
    "data.init_args.fold": 0,
}
METRICS = (
    ("auroc", "Auroc"),
    ("auprc", "Auprc"),
    ("accuracy", "Acc"),
    ("balanced_accuracy", "BalAcc"),
)


@dataclass
class Scored:
    name: str
    path: Path
    record: dict[str, Any]
    run: dict[str, Any]
    config: dict[str, Any]
    estimates: dict[str, dict[str, Estimate]]
    per_class_auroc: dict[str, tuple[float, float, float]]
    recall: dict[str, float]
    confusion: dict[str, float]
    counts: dict[str, int]
    sizes: dict[str, int]
    test_target: np.ndarray
    history: dict[int, dict[str, float]]


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
        raise SystemExit(f"{name}: recomputed {ours!r} but the run records {recorded!r}")


def epoch_history(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["epoch"]:
                continue
            values = out.setdefault(int(row["epoch"]), {})
            for key in ("train/loss", "val/auroc", "val/accuracy"):
                if row[key]:
                    values[key] = float(row[key])
    return out


def flatten(tree: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(tree, dict):
        return {prefix: tree}
    out: dict[str, Any] = {}
    for key, value in tree.items():
        out |= flatten(value, f"{prefix}.{key}" if prefix else key)
    return out


def score(name: str, run_id: str) -> Scored:
    path = PROJECT / "runs" / run_id
    record = json.loads((path / "metrics.json").read_text())
    run = json.loads((path / "run.json").read_text())
    if run["git"]["dirty"]:
        raise SystemExit(f"{run_id} trained from a dirty tree")
    estimates: dict[str, dict[str, Estimate]] = {}
    per_class: dict[str, tuple[float, float, float]] = {}
    recall: dict[str, float] = {}
    confusion: dict[str, float] = {}
    counts: dict[str, int] = {}
    sizes: dict[str, int] = {}
    target = np.empty(0)
    for split in ("test", "val"):
        pred = read_predictions(path / f"predictions_{split}.csv")
        classes = len(pred.classes)
        found = bootstrap(
            statistic(pred.target, pred.prob, classes),
            len(pred.target),
            replicates=record["bootstrap"]["replicates"],
            seed=record["bootstrap"]["seed"],
        )
        for key, estimate in found.items():
            for part in ("value", "lower", "upper"):
                check(
                    f"{run_id} {split}/{key}/{part}",
                    getattr(estimate, part),
                    record[split][key][part],
                )
        check(
            f"{run_id} {split} medmnist AUC",
            getAUC(pred.target, pred.prob, "multi-class"),
            found["auroc"].value,
        )
        check(
            f"{run_id} {split} medmnist ACC",
            getACC(pred.target, pred.prob, "multi-class"),
            found["accuracy"].value,
        )
        estimates[split] = found
        sizes[split] = len(pred.target)
        if split != "test":
            continue
        target = pred.target
        predicted = pred.prob.argmax(1)
        aurocs = cm.per_class(pred.target, pred.prob, cm.auroc)
        index = {}
        for i, label in enumerate(pred.classes):
            short = SHORT[label][0]
            index[short] = i
            entry = record["test"][f"auroc/{label}"]
            check(f"{run_id} test/auroc/{label}", float(aurocs[i]), entry["value"])
            per_class[short] = (entry["value"], entry["lower"], entry["upper"])
            recall[short] = float(np.mean(predicted[pred.target == i] == i))
            counts[short] = int(np.sum(pred.target == i))
        for macro, true, said in CONFUSIONS:
            mask = pred.target == index[true]
            confusion[macro] = float(np.mean(predicted[mask] == index[said]))
        wrong = pred.target[predicted != pred.target]
        for macro, group in (
            ("errShareDCTGlom", ("DCT", "GEC", "Podo")),
            ("errShareCDCNTPT", ("CDCNT", "PT")),
        ):
            confusion[macro] = float(np.isin(wrong, [index[g] for g in group]).mean())
    config = yaml.safe_load((path / "config.yaml").read_text())
    history = epoch_history(path / "metrics.csv")
    return Scored(
        name,
        path,
        record,
        run,
        config,
        estimates,
        per_class,
        recall,
        confusion,
        counts,
        sizes,
        target,
        history,
    )


def check_same_protocol(runs: list[Scored]) -> None:
    """The seeds share data, split, and configuration; only the seed differs."""
    first = runs[0]
    base = flatten(first.config)
    for other in runs[1:]:
        if not np.array_equal(first.test_target, other.test_target):
            raise SystemExit(f"{other.name}: test targets differ from seed {first.name}")
        if other.run["dataset"]["fingerprint"] != first.run["dataset"]["fingerprint"]:
            raise SystemExit(f"{other.name}: data fingerprint differs")
        flat = flatten(other.config)
        for key in sorted(set(base) | set(flat)):
            if key == "seed_everything" or base.get(key) == flat.get(key):
                continue
            if key not in base and key in DEFAULTED_KEYS and flat[key] == DEFAULTED_KEYS[key]:
                continue
            raise SystemExit(
                f"{other.name}: config key {key} is {flat.get(key)!r}, was {base.get(key)!r}"
            )
        for package, version in first.run["environment"]["packages"].items():
            if (
                package != "opendiagnostics"
                and other.run["environment"]["packages"][package] != version
            ):
                raise SystemExit(
                    f"{other.name}: {package} {other.run['environment']['packages'][package]} != {version}"
                )


def mean_sd(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def baseline_row(method: str) -> dict[str, str]:
    """How the results table labels a baselines.yaml row: the first prefix that matches."""
    with (Path(__file__).parent / "baseline_rows.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if method.startswith(row["method_prefix"]):
                return row
    raise SystemExit(f"baselines.yaml row {method!r} has no line in baseline_rows.tsv")


def plus_minus(metrics: dict[str, float], key: str) -> str:
    sd = metrics.get(f"{key}_sd")
    digits = 4 if sd is not None or len(f"{metrics[key]}") > 5 else 3
    value = f"{metrics[key]:.{digits}f}"
    return f"{value} $\\pm$ {sd:.4f}" if sd is not None else value


def results_table(runs: list[Scored], numbers: dict[str, str], baselines: list[dict]) -> str:
    rows = []
    for run in runs:
        test = run.estimates["test"]
        rows.append(
            f"ConvNeXt-S, seed {run.run['seed']} (ours) & 32 $\\rightarrow$ 224 & 1 & "
            f"{test['auroc'].value:.4f} [{test['auroc'].lower:.4f}, {test['auroc'].upper:.4f}] & "
            f"{test['accuracy'].value:.4f} [{test['accuracy'].lower:.4f}, {test['accuracy'].upper:.4f}] & this work \\\\"
        )
    rows.append(
        f"ConvNeXt-S, mean (ours) & 32 $\\rightarrow$ 224 & 3 & \\textbf{{{numbers['testAurocMean']}}} $\\pm$ "
        f"{numbers['testAurocSD']} & \\textbf{{{numbers['testAccMean']}}} $\\pm$ {numbers['testAccSD']} & this work \\\\"
    )
    rows.append("\\midrule")
    for baseline in baselines:
        row = baseline_row(baseline["method"])
        rows.append(
            f"{row['label']} & {row['input']} & {row['runs']} & "
            f"{plus_minus(baseline['metrics'], 'auroc')} & "
            f"{plus_minus(baseline['metrics'], 'accuracy')} & \\cite{{{row['cite']}}}, {row['where']} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table*}[t]\n"
        "\\caption{TissueMNIST test results on the official split of 47,280 images. Our seed rows give 95\\% "
        "percentile bootstrap intervals over test images; the mean row gives the mean $\\pm$ SD over the three "
        "seeds. Published rows give the mean, and $\\pm$ SD over runs where the source reports it. Runs: seeds or "
        "trials averaged (n.s.: not stated). Input: the image the network sees, from the $32\\times32$ projection "
        "or the 28-pixel MedMNIST image; an arrow marks a resize.}\n"
        "\\label{tab:main}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}llllll@{}}\n\\toprule\n"
        "Method & Input & Runs & AUROC & Accuracy & Source \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table*}}\n"
    )


def per_class_table(runs: list[Scored], numbers: dict[str, str]) -> str:
    rows = []
    order = sorted(SHORT.values(), key=lambda s: float(numbers[f"auroc{s[0]}Mean"]))
    for short, label in order:
        rows.append(
            f"{label} & {numbers[f'count{short}']} & {numbers[f'auroc{short}Mean']} & "
            f"{numbers[f'auroc{short}Min']}--{numbers[f'auroc{short}Max']} & {numbers[f'recall{short}Mean']} & "
            f"{numbers[f'recall{short}Min']}--{numbers[f'recall{short}Max']} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n"
        "\\caption{Which classes does the model confuse? One-vs-rest test AUROC and recall at the argmax "
        "decision for each class, as the mean and the range over the three seeds, sorted by AUROC. "
        "$n$: test images.}\n"
        "\\label{tab:perclass}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}lrllll@{}}\n\\toprule\n"
        "Class & $n$ & AUROC & range & Recall & range \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def main() -> None:
    runs = [score(name, run_id) for name, run_id in RUNS.items()]
    check_same_protocol(runs)
    baselines = yaml.safe_load((PROJECT / "baselines.yaml").read_text())["baselines"]
    numbers: dict[str, str] = {}

    def put(name: str, value: float | int | str, fmt: str = "{:.4f}") -> None:
        numbers[name] = value if isinstance(value, str) else fmt.format(value)

    for run in runs:
        for split, prefix in (("test", "test"), ("val", "val")):
            for key, macro in METRICS:
                estimate = run.estimates[split][key]
                put(f"{prefix}{macro}Seed{run.name}", estimate.value)
                put(f"{prefix}{macro}LoSeed{run.name}", estimate.lower)
                put(f"{prefix}{macro}HiSeed{run.name}", estimate.upper)
        minutes = (
            datetime.fromisoformat(run.run["finished"]) - datetime.fromisoformat(run.run["started"])
        ).total_seconds() / 60
        put(f"runMinutesSeed{run.name}", minutes, "{:.0f}")
        put(f"commitSeed{run.name}", run.run["git"]["commit"][:7])
        selected = run.record["selection"]["best_epoch"]
        put(f"selectedEpochSeed{run.name}", selected + 1, "{}")
        put(
            f"emaMinusRawAurocSeed{run.name}",
            run.history[selected]["val/auroc"] - run.record["val"]["auroc"]["value"],
        )

    for split, prefix in (("test", "test"), ("val", "val")):
        for key, macro in METRICS:
            values = [r.estimates[split][key].value for r in runs]
            mean, sd = mean_sd(values)
            put(f"{prefix}{macro}Mean", mean)
            put(f"{prefix}{macro}SD", sd)
            put(f"{prefix}{macro}Min", min(values))
            put(f"{prefix}{macro}Max", max(values))
            put(f"{prefix}{macro}LoMin", min(r.estimates[split][key].lower for r in runs))
            half = float(
                np.mean(
                    [
                        (r.estimates[split][key].upper - r.estimates[split][key].lower) / 2
                        for r in runs
                    ]
                )
            )
            put(f"{prefix}{macro}HalfWidth", half)
            put(f"{prefix}{macro}BootSE", half / 1.96)
    put("testN", len(runs[0].test_target), "{:,}")
    put("valN", runs[0].sizes["val"], "{:,}")
    put("testErrorPct", 100 * (1 - float(numbers["testAccMean"])), "{:.0f}")
    put("seeds", len(runs), "{}")

    for short, _ in SHORT.values():
        aurocs = [r.per_class_auroc[short][0] for r in runs]
        recalls = [r.recall[short] for r in runs]
        put(f"count{short}", runs[0].counts[short], "{:,}")
        put(f"auroc{short}Mean", float(np.mean(aurocs)))
        put(f"auroc{short}Min", min(aurocs))
        put(f"auroc{short}Max", max(aurocs))
        put(f"recall{short}Mean", float(np.mean(recalls)), "{:.3f}")
        put(f"recall{short}Min", min(recalls), "{:.3f}")
        put(f"recall{short}Max", max(recalls), "{:.3f}")
    for macro, _, _ in CONFUSIONS:
        shares = [r.confusion[macro] for r in runs]
        put(macro, 100 * float(np.mean(shares)), "{:.0f}")
        put(f"{macro}Min", 100 * min(shares), "{:.0f}")
        put(f"{macro}Max", 100 * max(shares), "{:.0f}")

    best = max(baselines, key=lambda b: b["metrics"]["auroc"])
    best_acc = max(baselines, key=lambda b: b["metrics"]["accuracy"])
    if best is not best_acc:
        raise SystemExit(
            f"best AUROC ({best['method']}) and best accuracy ({best_acc['method']}) differ"
        )
    native = next(
        b for b in baselines if b["method"].startswith("DenseNet-121, end-to-end, native 224")
    )
    second = sorted(baselines, key=lambda b: -b["metrics"]["auroc"])[1]
    for prefix, row in (("baseBest", best), ("baseNative", native), ("baseSecond", second)):
        put(f"{prefix}Auroc", row["metrics"]["auroc"])
        put(f"{prefix}Acc", row["metrics"]["accuracy"])
    put("baseBestAurocSD", best["metrics"]["auroc_sd"])
    put("baseBestAccSD", best["metrics"]["accuracy_sd"])
    for key, macro in (("auroc", "Auroc"), ("accuracy", "Acc")):
        ours = [r.estimates["test"][key].value for r in runs]
        mean, sd = mean_sd(ours)
        theirs, their_sd = best["metrics"][key], best["metrics"][f"{key}_sd"]
        pooled = math.sqrt((sd**2 + their_sd**2) / 2)
        welch = stats.ttest_ind_from_stats(
            mean, sd, len(ours), theirs, their_sd, 3, equal_var=False
        )
        put(f"gainMean{macro}", mean - theirs)
        put(f"marginLower{macro}", min(r.estimates["test"][key].lower for r in runs) - theirs)
        put(f"pooledSD{macro}", pooled)
        put(f"gainPooledSD{macro}", (mean - theirs) / pooled, "{:.1f}")
        put(f"welchT{macro}", float(welch.statistic), "{:.1f}")
        put(f"welchP{macro}", float(welch.pvalue), "{:.2g}")
        put(f"gainMinSeed{macro}", min(ours) - theirs)
        put(f"gainMinSeedInSD{macro}", (min(ours) - theirs) / their_sd, "{:.0f}")
        a, b = sd**2 / len(ours), their_sd**2 / 3
        put(f"welchDf{macro}", (a + b) ** 2 / (a**2 / (len(ours) - 1) + b**2 / 2), "{:.1f}")
        by_n = stats.ttest_ind_from_stats(
            mean, sd, len(ours), theirs, their_sd * math.sqrt(3 / 2), 3, equal_var=False
        )
        put(f"welchPByN{macro}", float(by_n.pvalue), "{:.2g}")
    put("gainNativeAurocMean", float(numbers["testAurocMean"]) - native["metrics"]["auroc"])

    first = runs[0]
    selected = first.record["selection"]["best_epoch"]
    if any(r.record["selection"]["best_epoch"] != selected for r in runs):
        raise SystemExit("the seeds selected different epochs; the text assumes one")
    put("selectedEpoch", selected + 1, "{}")
    for macro in ("errShareDCTGlom", "errShareCDCNTPT"):
        shares = [r.confusion[macro] for r in runs]
        put(f"{macro}Min", 100 * min(shares), "{:.0f}")
        put(f"{macro}Max", 100 * max(shares), "{:.0f}")
    put("epochs", len(first.history), "{}")
    put("emaValAurocFirst", float(np.mean([r.history[0]["val/auroc"] for r in runs])))
    put(
        "emaMinusRawAurocMean",
        float(
            np.mean(
                [
                    r.history[r.record["selection"]["best_epoch"]]["val/auroc"]
                    - r.record["val"]["auroc"]["value"]
                    for r in runs
                ]
            )
        ),
    )
    put(
        "lastThreeSpreadMax",
        max(
            max(v) - min(v)
            for v in ([r.history[e]["val/auroc"] for e in sorted(r.history)[-3:]] for r in runs)
        ),
    )
    put("trainLossFirst", float(np.mean([r.history[0]["train/loss"] for r in runs])), "{:.3f}")
    put(
        "trainLossLast",
        float(np.mean([r.history[max(r.history)]["train/loss"] for r in runs])),
        "{:.3f}",
    )
    put("emaValAurocSelected", first.history[selected]["val/auroc"])
    put("parameters", first.run["parameters"], "{:,}")
    put("bootReplicates", first.record["bootstrap"]["replicates"], "{:,}")
    put("commitRescore", first.record["rescored"]["git"]["commit"][:7])

    for name, value in numbers.items():
        print(f"{name}\t{value}")
    out = PAPER / "generated"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "numbers.tex").write_text(
        "\n".join(f"\\newcommand{{\\{n}}}{{{v.replace(',', '{,}')}}}" for n, v in numbers.items())
        + "\n"
    )
    (out / "tables/results.tex").write_text(results_table(runs, numbers, baselines))
    (out / "tables/perclass.tex").write_text(per_class_table(runs, numbers))


if __name__ == "__main__":
    main()
