"""Recompute every number the ChestMNIST paper uses from the committed run files.

    uv run python -I papers/chestmnist/analysis/paper_numbers.py

Scores each of the three seeds from its committed predictions, checks every value against
the run's metrics.json, the medmnist evaluator, and generated/patient_ids.json, and fails on
any mismatch. Prints each number with its name, and writes generated/numbers.tex (one LaTeX
macro per number), generated/seeds.json for the figures, and the results, per-label, and
subset tables in generated/tables/.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml
from medmnist.evaluator import getACC, getAUC
from scipy import stats

from opendiagnostics.metrics import classification as cm
from opendiagnostics.metrics.bootstrap import Estimate, bootstrap

ROOT = Path(__file__).resolve().parents[3]
PAPER = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "models/chestmnist-convnext"
RUNS = {
    "Zero": "20261126-165951-ab39eb97",
    "One": "20261127-142008-a35c5dae",
    "Two": "20261127-150930-405f7138",
}
TISSUE = ROOT / "papers/tissuemnist/analysis"
BEST_END_TO_END = "DINOv2 ViT-B/14"
BEST = "BioFuse"
MEDMNIST_PLUS = "DenseNet-121, end-to-end, native 224"
PATIENT_REPLICATES = 10_000
LABEL_NAMES = {"pleural": "Pleural thickening"}
SUBSETS_SCREENED = 12


def load_module(path: Path) -> Any:
    """The TissueMNIST paper's analysis code, reused so the two papers apply one rule."""
    spec = importlib.util.spec_from_file_location(f"tissue_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Scored:
    name: str
    record: dict[str, Any]
    run: dict[str, Any]
    config: dict[str, Any]
    ids: list[str]
    test_target: np.ndarray
    test_prob: np.ndarray
    image: dict[str, dict[str, Estimate]]
    history: dict[int, dict[str, float]]


def predictions(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    frame = pd.read_csv(path)
    labels = [c.removeprefix("prob/") for c in frame.columns if c.startswith("prob/")]
    target = frame[[f"target/{n}" for n in labels]].to_numpy(np.int64)
    prob = frame[[f"prob/{n}" for n in labels]].to_numpy(np.float64)
    return frame["id"].astype(str).tolist(), target, prob, labels


def statistic(target: np.ndarray, prob: np.ndarray):
    """The pipeline's multilabel statistic: macro AUROC, and accuracy at score >= 0.5."""

    def run(index: np.ndarray) -> dict[str, float]:
        t, p = target[index], prob[index]
        return {
            "auroc": cm.nanmean(cm.per_class(t, p, cm.auroc)),
            "accuracy": float(np.mean((p >= 0.5) == t)),
        }

    return run


def check(name: str, ours: float, recorded: float, tolerance: float = 1e-9) -> None:
    if abs(ours - recorded) > tolerance:
        raise SystemExit(f"{name}: recomputed {ours!r} but the record says {recorded!r}")


def learning_rates(path: Path) -> list[float]:
    """The learning rate of each epoch, in order, as LearningRateMonitor logged it."""
    with path.open(newline="") as handle:
        return [float(row["lr-AdamW"]) for row in csv.DictReader(handle) if row["lr-AdamW"]]


def epoch_history(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["epoch"]:
                continue
            values = out.setdefault(int(row["epoch"]), {})
            for key in ("train/loss", "val/auroc"):
                if row[key]:
                    values[key] = float(row[key])
    return out


def score(name: str, run_id: str) -> Scored:
    path = PROJECT / "runs" / run_id
    record = json.loads((path / "metrics.json").read_text())
    run = json.loads((path / "run.json").read_text())
    if run["git"]["dirty"]:
        raise SystemExit(f"{run_id} trained from a dirty tree")
    image: dict[str, dict[str, Estimate]] = {}
    ids: list[str] = []
    test_target = test_prob = np.empty(0)
    for split in ("test", "val"):
        split_ids, target, prob, labels = predictions(path / f"predictions_{split}.csv")
        found = bootstrap(
            statistic(target, prob),
            len(target),
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
            f"{run_id} {split} getAUC",
            getAUC(target, prob, "multi-label, binary-class"),
            found["auroc"].value,
        )
        acc = getACC(target, prob, "multi-label, binary-class")
        check(f"{run_id} {split} getACC", acc, found["accuracy"].value)
        check(
            f"{run_id} {split} auprc",
            cm.nanmean(cm.per_class(target, prob, cm.auprc)),
            record[split]["auprc"]["value"],
        )
        for label, value in zip(labels, cm.per_class(target, prob, cm.auroc), strict=True):
            check(
                f"{run_id} {split}/auroc/{label}",
                float(value),
                record[split][f"auroc/{label}"]["value"],
            )
        image[split] = found
        if split == "test":
            ids, test_target, test_prob = split_ids, target, prob
    return Scored(
        name,
        record,
        run,
        yaml.safe_load((path / "config.yaml").read_text()),
        ids,
        test_target,
        test_prob,
        image,
        epoch_history(path / "metrics.csv"),
    )


def patient_table() -> tuple[pd.DataFrame, pd.Series]:
    """The NIH match for every ChestMNIST row, and which matches are confident, by the rule
    in scripts/patient_ids.py."""
    table = pd.read_csv(PAPER / "generated/patient_ids.csv.gz", dtype={"patient_id": "Int64"})
    confident = (table["correlation"] >= 0.95) & (
        table["runner_up"].isna() | (table["correlation"] - table["runner_up"] >= 0.02)
    )
    return table, confident


def baseline(baselines: list[dict], prefix: str) -> dict[str, float]:
    return next(b for b in baselines if b["method"].startswith(prefix))["metrics"]


def welch(
    ours: list[float], theirs: float, their_sd: float, n: int = 3
) -> tuple[float, float, float]:
    mean, sd = float(np.mean(ours)), float(np.std(ours, ddof=1))
    result = stats.ttest_ind_from_stats(mean, sd, len(ours), theirs, their_sd, n, equal_var=False)
    a, b = sd**2 / len(ours), their_sd**2 / n
    df = (a + b) ** 2 / (a**2 / (len(ours) - 1) + b**2 / (n - 1))
    return float(result.statistic), df, float(result.pvalue)


def main() -> None:
    tissue = load_module(TISSUE / "paper_numbers.py")
    runs = [score(name, run_id) for name, run_id in RUNS.items()]
    tissue.check_same_protocol(
        [
            SimpleNamespace(name=r.name, test_target=r.test_target, run=r.run, config=r.config)
            for r in runs
        ]
    )
    labels = predictions(PROJECT / "runs" / RUNS["Zero"] / "predictions_test.csv")[3]
    numbers: dict[str, str] = {}

    def put(name: str, value: float | int | str, fmt: str = "{:.4f}") -> None:
        numbers[name] = value if isinstance(value, str) else fmt.format(value)

    table, confident = patient_table()
    recorded = json.loads((PAPER / "generated/patient_ids.json").read_text())
    splits = ("train", "val", "test")
    for split in splits:
        rows = table["split"] == split
        check(f"{split} rows", int(rows.sum()), recorded["matched"][split], 0)
        check(
            f"{split} confident",
            int((rows & confident).sum()),
            recorded["confident_matches"][split],
            0,
        )
        put(f"rows{split.capitalize()}", int(rows.sum()), "{:,}")
        put(f"confident{split.capitalize()}", int((rows & confident).sum()), "{:,}")
        put(f"unmatched{split.capitalize()}", int((rows & ~confident).sum()), "{:,}")
    sure = {s: set(table.loc[(table["split"] == s) & confident, "patient_id"]) for s in splits}
    loose = {s: set(table.loc[table["split"] == s, "patient_id"].dropna()) for s in splits}
    for split in splits:
        check(f"{split} patients", len(sure[split]), recorded["patients"][split], 0)
        put(f"patients{split.capitalize()}", len(sure[split]), "{:,}")
    check(
        "train and test",
        len(sure["train"] & sure["test"]),
        recorded["patients_in_train_and_test"],
        0,
    )
    check("val and test", len(sure["val"] & sure["test"]), recorded["patients_in_val_and_test"], 0)
    put("sharedTrainTest", len(sure["train"] & sure["test"]), "{}")
    put("sharedValTest", len(sure["val"] & sure["test"]), "{}")
    put("sharedTrainVal", len(sure["train"] & sure["val"]), "{}")
    put("looseSharedTrainTest", len(loose["train"] & loose["test"]), "{}")
    test = table[table["split"] == "test"].set_index("id")
    nih_test = int((test["nih_split"] == "test").sum())
    check("test rows from NIH test", nih_test, recorded["test_rows_from_nih_test"], 0)
    put("testFromNihTest", nih_test, "{:,}")
    put("testFromNihTrain", int((test["nih_split"] == "train").sum()), "{:,}")
    for name, value, fmt in load_module(PAPER / "analysis/matching.py").matching_checks(
        table, confident
    ):
        put(name, value, fmt)
    per_patient = test.loc[
        confident[table["split"] == "test"].to_numpy(), "patient_id"
    ].value_counts()
    put("imagesPerPatientMean", float(per_patient.mean()), "{:.1f}")
    put("imagesPerPatientMax", int(per_patient.max()), "{}")
    put("testPatientsSingle", 100 * float((per_patient == 1).mean()), "{:.0f}")
    put("corrQuantileOne", float(table["correlation"].quantile(0.001)))

    seeds: dict[str, dict[str, Any]] = {}
    for r in runs:
        groups = [
            str(test["patient_id"].get(i)) if pd.notna(test["patient_id"].get(i)) else i
            for i in r.ids
        ]
        stat = statistic(r.test_target, r.test_prob)
        seed = r.record["bootstrap"]["seed"]
        patient = bootstrap(stat, len(r.ids), groups, replicates=PATIENT_REPLICATES, seed=seed)
        as_logged = bootstrap(stat, len(r.ids), groups)["auroc"]
        logged = recorded["runs"][RUNS[r.name]]["patient_level"]
        for value, part in zip(logged, ("value", "lower", "upper"), strict=True):
            check(f"{r.name} patient_ids.json {part}", getattr(as_logged, part), value)
        unseen = ~test.loc[r.ids, "patient_id"].isin(loose["train"]).to_numpy()
        test_auroc = r.image["test"]["auroc"]
        seeds[r.name] = {
            "seed": r.run["seed"],
            "image": [test_auroc.value, test_auroc.lower, test_auroc.upper],
            "patient": [patient["auroc"].value, patient["auroc"].lower, patient["auroc"].upper],
            "accuracy": r.image["test"]["accuracy"].value,
        }
        for split in ("test", "val"):
            for key, macro in (("auroc", "Auroc"), ("accuracy", "Acc")):
                est = r.image[split][key]
                put(f"{split}{macro}Seed{r.name}", est.value)
                put(f"{split}{macro}LoSeed{r.name}", est.lower)
                put(f"{split}{macro}HiSeed{r.name}", est.upper)
        put(f"patientAurocLoSeed{r.name}", patient["auroc"].lower)
        put(f"patientAurocHiSeed{r.name}", patient["auroc"].upper)
        put(f"patientAccLoSeed{r.name}", patient["accuracy"].lower)
        put(f"patientAccHiSeed{r.name}", patient["accuracy"].upper)
        put(f"loggedPatientLoSeed{r.name}", as_logged.lower)
        put(
            f"widthRatioSeed{r.name}",
            (patient["auroc"].upper - patient["auroc"].lower)
            / (test_auroc.upper - test_auroc.lower),
            "{:.2f}",
        )
        put(
            f"unseenShiftSeed{r.name}",
            cm.nanmean(cm.per_class(r.test_target[unseen], r.test_prob[unseen], cm.auroc))
            - test_auroc.value,
            "{:+.4f}",
        )
        put(f"unseenRowsSeed{r.name}", int((~unseen).sum()), "{}")
        minutes = (
            datetime.fromisoformat(r.run["finished"]) - datetime.fromisoformat(r.run["started"])
        ).total_seconds() / 60
        put(f"runMinutesSeed{r.name}", minutes, "{:.0f}")
        put(f"commitSeed{r.name}", r.run["git"]["commit"][:7])
        selected = r.record["selection"]["best_epoch"]
        put(f"selectedEpochSeed{r.name}", selected + 1, "{}")
        put(f"emaValAurocSeed{r.name}", r.history[selected]["val/auroc"])
        rates = learning_rates(PROJECT / "runs" / RUNS[r.name] / "metrics.csv")
        put(f"selectedLrSeed{r.name}", rates[selected] * 1e5, "{:.1f}")
        put(
            f"selectionGapSeed{r.name}",
            r.history[selected]["val/auroc"]
            - max(v["val/auroc"] for e, v in r.history.items() if e != selected),
            "{:.5f}",
        )
        put(
            f"emaMinusRawSeed{r.name}",
            r.history[selected]["val/auroc"] - r.record["val"]["auroc"]["value"],
        )

    aurocs = [r.image["test"]["auroc"].value for r in runs]
    accs = [r.image["test"]["accuracy"].value for r in runs]
    for values, macro in ((aurocs, "Auroc"), (accs, "Acc")):
        put(f"test{macro}Mean", float(np.mean(values)))
        put(f"test{macro}SD", float(np.std(values, ddof=1)))
        put(f"test{macro}Min", min(values))
        put(f"test{macro}Max", max(values))
    put("valAurocMean", float(np.mean([r.image["val"]["auroc"].value for r in runs])))
    put("patientLoMin", min(s["patient"][1] for s in seeds.values()))
    put(
        "imageHalfWidth",
        float(np.mean([(s["image"][2] - s["image"][1]) / 2 for s in seeds.values()])),
    )
    put(
        "patientHalfWidth",
        float(np.mean([(s["patient"][2] - s["patient"][1]) / 2 for s in seeds.values()])),
    )
    put("patientBootSE", float(numbers["patientHalfWidth"]) / 1.96)

    baselines = yaml.safe_load((PROJECT / "baselines.yaml").read_text())["baselines"]
    best = max(baselines, key=lambda b: b["metrics"]["auroc"])
    if not best["method"].startswith(BEST):
        raise SystemExit(f"the best published AUROC is now {best['method']}; update the text")
    biofuse = best["metrics"]
    wu = baseline(baselines, BEST_END_TO_END)
    plus = baseline(baselines, MEDMNIST_PLUS)
    mean = float(np.mean(aurocs))
    for prefix, row in (("baseBest", biofuse), ("baseWu", wu), ("basePlus", plus)):
        put(f"{prefix}Auroc", row["auroc"], "{:.4f}" if len(str(row["auroc"])) > 5 else "{:.3f}")
        put(
            f"{prefix}Acc", row["accuracy"], "{:.4f}" if len(str(row["accuracy"])) > 5 else "{:.3f}"
        )
        put(f"gainMean{prefix[4:]}", mean - row["auroc"])
        put(f"gainMinSeed{prefix[4:]}", min(aurocs) - row["auroc"])
    put("baseWuSD", wu["auroc_sd"])
    put("basePlusSD", plus["auroc_sd"])
    put("gainMinSeedInWuSD", (min(aurocs) - wu["auroc"]) / wu["auroc_sd"], "{:.1f}")
    put("gainMeanInWuSD", (mean - wu["auroc"]) / wu["auroc_sd"], "{:.1f}")
    put("marginPatientBest", min(s["patient"][1] for s in seeds.values()) - biofuse["auroc"])
    for name, theirs, sd in (
        ("Wu", wu["auroc"], wu["auroc_sd"]),
        ("WuByN", wu["auroc"], wu["auroc_sd"] * math.sqrt(3 / 2)),
        ("BestAssumed", biofuse["auroc"], wu["auroc_sd"]),
        ("Plus", plus["auroc"], plus["auroc_sd"]),
    ):
        t, df, p = welch(aurocs, theirs, sd)
        put(f"welchT{name}", t, "{:.1f}")
        put(f"welchDf{name}", df, "{:.1f}")
        put(f"welchP{name}", p, "{:.2g}")
    fixed = stats.ttest_1samp(aurocs, biofuse["auroc"])
    put("oneSampleTBest", float(fixed.statistic), "{:.1f}")
    put("oneSamplePBest", float(fixed.pvalue), "{:.2g}")
    put(
        "oneSamplePBestHigh",
        float(stats.ttest_1samp(aurocs, biofuse["auroc"] + 0.0005).pvalue),
        "{:.3f}",
    )
    put("bonferroniWu", min(1.0, SUBSETS_SCREENED * float(numbers["welchPWu"])), "{:.2f}")
    put("bonferroniBest", min(1.0, SUBSETS_SCREENED * float(fixed.pvalue)), "{:.2f}")
    put("subsetsScreened", SUBSETS_SCREENED, "{}")
    later = aurocs[1:]
    put("laterWelchPWu", welch(later, wu["auroc"], wu["auroc_sd"])[2], "{:.2g}")
    put("laterOneSamplePBest", float(stats.ttest_1samp(later, biofuse["auroc"]).pvalue), "{:.2g}")
    t, df, _ = welch(aurocs, wu["auroc"], wu["auroc_sd"])
    half = float(stats.t.ppf(0.975, df)) * (mean - wu["auroc"]) / t
    put("diffWuLo", mean - wu["auroc"] - half)
    put("diffWuHi", mean - wu["auroc"] + half)

    target = runs[0].test_target
    if any(not np.array_equal(target, r.test_target) for r in runs):
        raise SystemExit("the seeds' test labels differ")
    prevalence = target.mean(axis=0)
    put("testN", len(target), "{:,}")
    put("labelDecisions", target.size, "{:,}")
    put("allNegAcc", 1 - float(target.mean()))
    put("labelPosPct", 100 * float(target.mean()), "{:.2f}")
    put("predPosPct", 100 * float(np.mean([np.mean(r.test_prob >= 0.5) for r in runs])), "{:.2f}")
    put("noFindingPct", 100 * float(np.mean(target.sum(axis=1) == 0)), "{:.1f}")
    per_label = np.array([cm.per_class(r.test_target, r.test_prob, cm.auroc) for r in runs])
    rows = []
    for k in np.argsort(per_label.mean(axis=0)):
        rows.append(
            f"{LABEL_NAMES.get(labels[k], labels[k].capitalize())} & {100 * prevalence[k]:.2f} & {per_label[:, k].mean():.4f} & "
            f"{per_label[:, k].min():.4f}--{per_label[:, k].max():.4f} \\\\"
        )
    low, high = int(np.argmin(per_label.mean(axis=0))), int(np.argmax(per_label.mean(axis=0)))
    if (labels[low], labels[high]) != ("infiltration", "hernia"):
        raise SystemExit(f"worst and best labels are now {labels[low]} and {labels[high]}")
    put("lowLabelAuroc", float(per_label[:, low].mean()))
    put("highLabelAuroc", float(per_label[:, high].mean()))
    put("prevMinPct", 100 * float(prevalence.min()), "{:.2f}")
    put("prevMaxPct", 100 * float(prevalence.max()), "{:.2f}")
    put("perLabelSpreadMax", float((per_label.max(axis=0) - per_label.min(axis=0)).max()))

    first = runs[0]
    put("epochs", len(first.history), "{}")
    put("parameters", first.run["parameters"], "{:,}")
    put("bootReplicates", first.record["bootstrap"]["replicates"], "{:,}")
    put("patientReplicates", PATIENT_REPLICATES, "{:,}")
    put("commitRescore", first.record["rescored"]["git"]["commit"][:7])
    put(
        "emaMinusRawMean",
        float(np.mean([float(numbers[f"emaMinusRawSeed{r.name}"]) for r in runs])),
    )
    put("trainLossFirst", float(np.mean([r.history[0]["train/loss"] for r in runs])), "{:.3f}")
    put(
        "trainLossLast",
        float(np.mean([r.history[max(r.history)]["train/loss"] for r in runs])),
        "{:.3f}",
    )

    subsets = load_module(TISSUE / "recipe_subsets.py")
    found = subsets.rows()
    if len(found) != 12:
        raise SystemExit(f"expected the 12 MedMNIST 2D subsets, found {len(found)}")
    subset_rows = []
    for row in found:
        verdict = (
            "above"
            if row["lower"] > row["best"]
            else "below"
            if row["upper"] < row["best"]
            else "inside"
        )
        row["verdict"] = verdict
        subset_rows.append(
            f"{subsets.display(row['subset'])} & {row['value']:.4f} [{row['lower']:.4f}, {row['upper']:.4f}] & "
            f"{row['best']:.4f} & {verdict} \\\\"
        )
    put("subsetsAbove", sum(r["verdict"] == "above" for r in found), "{}")
    put("subsetsPointAbove", sum(r["value"] > r["best"] for r in found), "{}")
    put("subsetsTotal", len(found), "{}")

    for name, value in numbers.items():
        print(f"{name}\t{value}")
    out = PAPER / "generated"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "numbers.tex").write_text(
        "\n".join(f"\\newcommand{{\\{n}}}{{{v.replace(',', '{,}')}}}" for n, v in numbers.items())
        + "\n"
    )
    (out / "seeds.json").write_text(json.dumps(seeds, indent=2) + "\n")
    tables = load_module(PAPER / "analysis/tables.py")
    (out / "tables/results.tex").write_text(tables.results_table(seeds, numbers, baselines))
    (out / "tables/perlabel.tex").write_text(tables.per_label_table(rows))
    (out / "tables/subsets.tex").write_text(
        tables.subsets_table(subset_rows, first.run["git"]["commit"][:7])
    )


if __name__ == "__main__":
    main()
