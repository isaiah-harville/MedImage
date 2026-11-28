"""Recompute every number the SynapseMNIST3D paper uses from the committed run files.

    uv run python -I papers/synapsemnist3d/analysis/paper_numbers.py

Scores the five seeds of each network from their committed predictions and checks every
value against the run's metrics.json and the medmnist evaluator. Checks that the runs share
one split and one test set, and that the two projects' configs differ only in the network
and the seed. Runs the paired comparisons, the seed-level tests, and the comparisons with
published values. Fails on any mismatch. Writes generated/numbers.tex (one LaTeX macro per
number) and generated/seeds.json for tables.py and make_figures.py.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from medmnist.evaluator import getACC, getAUC
from scipy import stats

from opendiagnostics.metrics import classification as cm
from opendiagnostics.metrics.bootstrap import bootstrap_draws
from opendiagnostics.metrics.compare import compare, read_predictions

ROOT = Path(__file__).resolve().parents[3]
PAPER = Path(__file__).resolve().parents[1]
ARMS = {"Scratch": "synapsemnist3d-resnet18", "Kinetics": "synapsemnist3d-r3d18"}
SEEDS = ("Zero", "One", "Two", "Three", "Four")
POSITIVE = "prob/excitatory synapse"
REPLICATES = 2000
TOPOCAM = "R3D-18, Kinetics-400 pretrained, fine-tuned (64³)"
PLANECYCLE = "DINOv3 ViT-L/16, fully fine-tuned, 3D baseline (mean of 5 runs)"
PARK = "Axial-centric cross-plane attention (64³)"
MEDMNIST = "ResNet-50 + 3D (28³), mean of at least 3 trials"
MTDL = "MTDL, Hodge-decomposition channels (64³), mean of 3 seeds"
CNMTDL = "CNMTDL, combinatorial-complex MTDL (64³), mean of 5 seeds"
TOPOCAM_TABLE9 = {
    "RThreeD": (0.973, 0.810, 0.929),
    "MCThree": (0.969, 0.874, 0.943),
    "RTwoPlusOne": (0.945, 0.842, 0.918),
}


@dataclass(frozen=True)
class Run:
    arm: str
    seed: str
    path: Path
    ids: list[str]
    target: np.ndarray
    prob: np.ndarray
    metrics: dict[str, Any]
    info: dict[str, Any]


def load(arm: str, seed: str, path: Path) -> Run:
    with (path / "predictions_test.csv").open(newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: r["id"])
    return Run(
        arm,
        seed,
        path,
        [r["id"] for r in rows],
        np.array([int(r["target"]) for r in rows]),
        np.array([float(r[POSITIVE]) for r in rows]),
        json.loads((path / "metrics.json").read_text()),
        json.loads((path / "run.json").read_text()),
    )


def statistic_for(target: np.ndarray, prob: np.ndarray, threshold: float):
    def statistic(index: np.ndarray) -> dict[str, float]:
        t, p = target[index], prob[index]
        return {
            "accuracy_at_0.5": cm.threshold_metrics(t, p, 0.5)["accuracy"],
            "auroc": cm.auroc(t, p),
            "auprc": cm.auprc(t, p),
            "brier": cm.brier(t, p),
            "ece": cm.expected_calibration_error(t, p, 15),
        } | cm.threshold_metrics(t, p, threshold)

    return statistic


def interval(draws: np.ndarray) -> tuple[float, float]:
    values = draws[~np.isnan(draws)]
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def check(name: str, ours: float, recorded: float) -> None:
    if not math.isclose(ours, recorded, abs_tol=1e-9):
        sys.exit(f"{name}: recomputed {ours}, recorded {recorded}")


def rescore(run: Run, split: str) -> dict[str, dict[str, float]]:
    """Every metric of one split with its interval, checked against metrics.json."""
    with (run.path / f"predictions_{split}.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [r["id"] for r in rows]
    target = np.array([int(r["target"]) for r in rows])
    prob = np.array([float(r[POSITIVE]) for r in rows])
    stored = run.metrics[split]
    assert len(set(ids)) == len(ids) == stored["n_samples"], (run.path, split)
    statistic = statistic_for(target, prob, run.metrics["calibration"]["threshold"])
    boot = run.metrics["bootstrap"]
    point, draws = bootstrap_draws(statistic, len(ids), ids, boot["replicates"], boot["seed"])
    out = {}
    for key, value in point.items():
        lower, upper = interval(draws[key])
        for part, mine in (("value", value), ("lower", lower), ("upper", upper)):
            check(f"{run.path.name} {split} {key} {part}", mine, stored[key][part])
        out[key] = {"value": value, "lower": lower, "upper": upper}
    if split == "test":
        check(
            f"{run.path.name} medmnist AUC",
            getAUC(target, prob[:, None], "binary-class"),
            point["auroc"],
        )
        acc = getACC(target, prob[:, None], "binary-class", 0.5)
        check(f"{run.path.name} medmnist ACC", acc, point["accuracy_at_0.5"])
    return out


def network_free(config: dict[str, Any]) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    for key in ("seed_everything", "name", "notes"):
        config.pop(key, None)
    config["model"]["init_args"].pop("network")
    config["model"]["init_args"].pop("class_weight", None)
    config["data"]["init_args"].pop("variant", None)
    return config


def logged_epochs(path: Path) -> list[dict[str, float]]:
    with (path / "metrics.csv").open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["val/auroc"]]
    return [{"epoch": int(r["epoch"]), "val_auroc": float(r["val/auroc"])} for r in rows]


def flops(arm: str) -> float:
    """Forward GFLOPs for one 64-voxel volume, counted by torch on CPU with random weights."""
    import torch
    from monai.networks.nets import ResNet
    from torch.utils.flop_counter import FlopCounterMode

    from opendiagnostics.networks import KineticsVideoResNet

    if arm == "Scratch":
        network = ResNet(
            "basic",
            [2, 2, 2, 2],
            [64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1,
        )
    else:
        network = KineticsVideoResNet("r3d_18", 1, pretrained=False)
    network.eval()
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        network(torch.zeros(1, 1, 64, 64, 64))
    return counter.get_total_flops() / 1e9


def permutation_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided permutation test of the difference in means over every relabeling."""
    pooled = np.concatenate([a, b])
    observed = abs(b.mean() - a.mean())
    hits = total = 0
    for chosen in itertools.combinations(range(len(pooled)), len(b)):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(chosen)] = True
        hits += abs(pooled[mask].mean() - pooled[~mask].mean()) >= observed - 1e-12
        total += 1
    return hits / total


def main() -> None:
    runs: dict[str, list[Run]] = {}
    for arm, project in ARMS.items():
        paths = sorted((ROOT / "models" / project / "runs").iterdir())
        assert len(paths) == len(SEEDS), (project, paths)
        runs[arm] = [load(arm, s, p) for s, p in zip(SEEDS, paths, strict=True)]
    every = [r for arm in runs.values() for r in arm]

    reference = every[0]
    split_ids = json.loads((reference.path / "splits.json").read_text())["ids"]
    sizes = {k: len(set(v)) for k, v in split_ids.items()}
    assert sizes == {"train": 1230, "val": 177, "test": 352}, sizes
    for a, b in itertools.combinations(("train", "val", "test"), 2):
        assert not set(split_ids[a]) & set(split_ids[b]), (a, b)
    base_config = network_free(yaml.safe_load((every[1].path / "config.yaml").read_text()))
    for run in every:
        assert json.loads((run.path / "splits.json").read_text())["ids"] == split_ids, run.path
        assert run.ids == reference.ids, run.path
        assert np.array_equal(run.target, reference.target), run.path
        config = yaml.safe_load((run.path / "config.yaml").read_text())
        assert config["seed_everything"] == SEEDS.index(run.seed), run.path
        assert network_free(config) == base_config, f"{run.path} differs beyond the network"
        assert run.info["git"]["dirty"] is False, run.path
    packages = {json.dumps(r.info["environment"]["packages"], sort_keys=True) for r in every}
    gpus = {r.info["environment"]["gpu"] for r in every}
    assert len(packages) == 1, packages
    assert len(gpus) == 1, gpus

    scored: dict[str, Any] = {}
    for run in every:
        test = rescore(run, "test")
        val = rescore(run, "val")
        history = logged_epochs(run.path)
        started = datetime.fromisoformat(run.info["started"])
        finished = datetime.fromisoformat(run.info["finished"])
        inhibitory = run.prob[run.target == 0]
        scored[f"{run.arm}{run.seed}"] = {
            "arm": run.arm,
            "seed": SEEDS.index(run.seed),
            "run": run.path.name,
            "commit": run.info["git"]["commit"][:8],
            "test": test,
            "val": val,
            "threshold": run.metrics["calibration"]["threshold"],
            "epoch": run.metrics["checkpoint_epoch"],
            "epochs": len(history),
            "history": history,
            "minutes": (finished - started).total_seconds() / 60,
            "parameters": run.info["parameters"],
            "mean_prob_inhibitory": float(inhibitory.mean()),
        }

    macros: dict[str, str] = {}

    def put(name: str, value: float | int | str, fmt: str = "{:.4f}") -> None:
        assert name.isalpha(), name
        macros[name] = value if isinstance(value, str) else fmt.format(value)

    auroc = {
        arm: np.array([scored[f"{arm}{s}"]["test"]["auroc"]["value"] for s in SEEDS])
        for arm in ARMS
    }
    acc = {
        arm: np.array([scored[f"{arm}{s}"]["test"]["accuracy_at_0.5"]["value"] for s in SEEDS])
        for arm in ARMS
    }
    for arm in ARMS:
        for seed in SEEDS:
            entry = scored[f"{arm}{seed}"]
            for metric, short in (("auroc", "Auroc"), ("accuracy_at_0.5", "Acc")):
                for part, suffix in (("value", ""), ("lower", "Lo"), ("upper", "Hi")):
                    digits = "{:.3f}" if short == "Acc" and not suffix else "{:.4f}"
                    put(f"{short}{arm}{seed}{suffix}", entry["test"][metric][part], digits)
            put(f"Epoch{arm}{seed}", entry["epoch"], "{}")
            put(f"Thr{arm}{seed}", entry["threshold"], "{:.3f}")
            put(f"AccThr{arm}{seed}", entry["test"]["accuracy"]["value"], "{:.3f}")
            put(f"PInh{arm}{seed}", entry["mean_prob_inhibitory"], "{:.2f}")
        put(f"meanAuroc{arm}", auroc[arm].mean())
        put(f"sdAuroc{arm}", auroc[arm].std(ddof=1))
        put(f"minAuroc{arm}", auroc[arm].min())
        put(f"maxAuroc{arm}", auroc[arm].max())
        put(f"rangeAuroc{arm}", auroc[arm].max() - auroc[arm].min(), "{:.3f}")
        put(f"meanAcc{arm}", acc[arm].mean())
        put(f"minAcc{arm}", acc[arm].min(), "{:.3f}")
        put(f"maxAcc{arm}", acc[arm].max(), "{:.3f}")
        half = [
            (e["test"]["auroc"]["upper"] - e["test"]["auroc"]["lower"]) / 2
            for k, e in scored.items()
            if e["arm"] == arm
        ]
        put(f"halfWidth{arm}", float(np.mean(half)), "{:.3f}")
        put(
            f"minutes{arm}",
            float(np.mean([e["minutes"] for e in scored.values() if e["arm"] == arm])),
            "{:.0f}",
        )
        put(f"params{arm}", scored[f"{arm}Zero"]["parameters"] / 1e6, "{:.2f}")
    put("laterMeanAurocScratch", auroc["Scratch"][1:].mean())
    ema_val, raw_val, tests = [], [], []
    for key, entry in scored.items():
        ema = next(h["val_auroc"] for h in entry["history"] if h["epoch"] == entry["epoch"])
        entry["val_auroc_ema"] = ema
        put(f"ValEma{key}", ema)
        put(f"ValRaw{key}", entry["val"]["auroc"]["value"])
        ema_val.append(ema)
        raw_val.append(entry["val"]["auroc"]["value"])
        tests.append(entry["test"]["auroc"]["value"])
    raw_rho = stats.spearmanr(raw_val, tests)
    ema_rho = stats.spearmanr(ema_val, tests)
    put("rhoRawTest", float(raw_rho.statistic), "{:.2f}")
    put("rhoRawTestP", float(raw_rho.pvalue), "{:.3f}")
    put("rhoEmaTest", float(ema_rho.statistic), "{:.2f}")
    margins = []
    for entry in scored.values():
        values = sorted((h["val_auroc"] for h in entry["history"]), reverse=True)
        margins.append(values[0] - values[1])
    put("maxSelectionMargin", max(margins), "{:.4f}")
    for arm in ARMS:
        keys = [k for k, e in scored.items() if e["arm"] == arm]
        raw = stats.spearmanr([scored[k]["val"]["auroc"]["value"] for k in keys], auroc[arm])
        ema = stats.spearmanr([scored[k]["val_auroc_ema"] for k in keys], auroc[arm])
        put(f"rhoRaw{arm}", float(raw.statistic), "{:.2f}")
        put(f"rhoEma{arm}", float(ema.statistic), "{:.2f}")
        mean_ci = stats.ttest_1samp(auroc[arm], 0).confidence_interval(0.95)
        put(f"tLo{arm}", float(mean_ci.low))
        put(f"tHi{arm}", float(mean_ci.high))
    put("leveneP", float(stats.levene(auroc["Scratch"], auroc["Kinetics"]).pvalue), "{:.2f}")
    levene_mean = stats.levene(auroc["Scratch"], auroc["Kinetics"], center="mean")
    put("leveneMeanP", float(levene_mean.pvalue), "{:.3f}")
    variance_ratio = auroc["Scratch"].var(ddof=1) / auroc["Kinetics"].var(ddof=1)
    df = len(SEEDS) - 1
    put("fTestP", float(min(1.0, 2 * stats.f.sf(variance_ratio, df, df))), "{:.3f}")
    wins = [float(k > a) for k in auroc["Kinetics"] for a in auroc["Scratch"]]
    put("probImprovement", float(np.mean(wins)), "{:.2f}")
    put("sdRatio", auroc["Scratch"].std(ddof=1) / auroc["Kinetics"].std(ddof=1), "{:.1f}")

    seed_diff = auroc["Kinetics"].mean() - auroc["Scratch"].mean()
    welch = stats.ttest_ind(auroc["Kinetics"], auroc["Scratch"], equal_var=False)
    welch_ci = welch.confidence_interval(0.95)
    put("seedDiff", seed_diff)
    put("permP", permutation_p(auroc["Scratch"], auroc["Kinetics"]), "{:.2f}")
    put("permCount", math.comb(2 * len(SEEDS), len(SEEDS)), "{}")
    put("welchP", float(welch.pvalue), "{:.2f}")
    put("welchDf", float(welch.df), "{:.1f}")
    put("welchLo", float(welch_ci.low), "{:.3f}")
    put("welchHi", float(welch_ci.high), "{:.3f}")
    put(
        "mannWhitneyP",
        float(stats.mannwhitneyu(auroc["Kinetics"], auroc["Scratch"]).pvalue),
        "{:.2f}",
    )

    paired = {}
    for seed, a, b in zip(SEEDS, runs["Scratch"], runs["Kinetics"], strict=True):
        result = compare(
            read_predictions(a.path / "predictions_test.csv"),
            read_predictions(b.path / "predictions_test.csv"),
            replicates=REPLICATES,
        )
        paired[seed] = result.to_dict()
        put(f"pairDiff{seed}", result.difference)
        put(f"pairLo{seed}", result.lower)
        put(f"pairHi{seed}", result.upper)
        put(f"pairP{seed}", result.p_value, "{:.3f}" if result.p_value >= 0.001 else "<0.001")
    order = sorted(paired, key=lambda k: paired[k]["p_value"])
    running = 0.0
    for rank, seed in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * paired[seed]["p_value"]))
        paired[seed]["holm"] = running
        put(f"pairHolm{seed}", running, "{:.3f}" if running >= 0.001 else "<0.001")
    put("holmSurvive", sum(p["holm"] < 0.05 for p in paired.values()), "{}")

    target = reference.target
    probs = {arm: np.stack([r.prob for r in runs[arm]]) for arm in ARMS}

    def mean_auroc(index: np.ndarray) -> dict[str, float]:
        t = target[index]
        means = {arm: float(np.mean([cm.auroc(t, p[index]) for p in probs[arm]])) for arm in ARMS}
        return means | {"difference": means["Kinetics"] - means["Scratch"]}

    pooled: dict[str, dict[str, float]] = {}
    point, draws = bootstrap_draws(mean_auroc, len(target), reference.ids, REPLICATES, 0)
    diff = draws["difference"]
    put("pooledDiff", point["difference"])
    for arm in ARMS:
        put(f"ensemble{arm}", cm.auroc(target, probs[arm].mean(0)))

    def ensemble_difference(index: np.ndarray) -> dict[str, float]:
        t = target[index]
        scores = {arm: cm.auroc(t, probs[arm].mean(0)[index]) for arm in ARMS}
        return {"difference": scores["Kinetics"] - scores["Scratch"]}

    ens_point, ens_draws = bootstrap_draws(
        ensemble_difference, len(target), reference.ids, REPLICATES, 0
    )
    put("ensembleDiff", ens_point["difference"])
    put("ensembleLo", interval(ens_draws["difference"])[0])
    put("ensembleHi", interval(ens_draws["difference"])[1])
    put("pooledLo", interval(diff)[0])
    put("pooledHi", interval(diff)[1])
    put("pooledP", min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))), "{:.3f}")
    for arm in ARMS:
        lower, upper = interval(draws[arm])
        put(f"pooledLo{arm}", lower)
        pooled[arm] = {"value": point[arm], "lower": lower, "upper": upper}
        put(f"pooledHi{arm}", upper)

    baselines = {
        b["method"]: b["metrics"]
        for b in yaml.safe_load(
            (ROOT / "models" / ARMS["Kinetics"] / "baselines.yaml").read_text()
        )["baselines"]
    }
    lists = [(ROOT / "models" / p / "baselines.yaml").read_text() for p in ARMS.values()]
    assert yaml.safe_load(lists[0]) == yaml.safe_load(lists[1]), "baselines.yaml lists differ"
    for name, method in (
        ("Topo", TOPOCAM),
        ("Plane", PLANECYCLE),
        ("Park", PARK),
        ("Medmnist", MEDMNIST),
        ("Mtdl", MTDL),
        ("Cnmtdl", CNMTDL),
    ):
        value = baselines[method]["auroc"]
        put(f"base{name}Auroc", value, "{:.3f}")
        if "accuracy" in baselines[method]:
            put(f"base{name}Acc", baselines[method]["accuracy"], "{:.3f}")
        for arm in ARMS:
            entries = [scored[f"{arm}{s}"]["test"]["auroc"] for s in SEEDS]
            put(f"above{name}{arm}", sum(e["value"] > value for e in entries), "{}")
            put(f"ciAbove{name}{arm}", sum(e["lower"] > value for e in entries), "{}")
            put(f"ciBelow{name}{arm}", sum(e["upper"] < value for e in entries), "{}")
            put(
                f"accAbove{name}{arm}",
                int(sum(acc[arm] > baselines[method].get("accuracy", 2))),
                "{}",
            )
    for arm in ARMS:
        one = stats.ttest_1samp(auroc[arm], baselines[PLANECYCLE]["auroc"])
        put(f"oneSampleP{arm}", float(one.pvalue), "{:.3f}")
        put(
            f"oneSampleTopoP{arm}",
            float(stats.ttest_1samp(auroc[arm], baselines[TOPOCAM]["auroc"]).pvalue),
            "{:.2f}",
        )

    duplicates = json.loads((PAPER / "generated/near_duplicates.json").read_text())
    put("dupMax", duplicates["similarity_quantiles"]["1.0"], "{:.2f}")
    put("dupMedian", duplicates["similarity_quantiles"]["0.5"], "{:.2f}")
    put("dupCount", duplicates["thresholds"]["0.8"]["near_duplicates"], "{}")
    put("dupRef", duplicates["reference_volumes"], "{}")

    with (reference.path / "predictions_val.csv").open(newline="") as handle:
        val_target = np.array([int(r["target"]) for r in csv.DictReader(handle)])
    put("nValExc", int(val_target.sum()), "{}")
    put("nValInh", int(len(val_target) - val_target.sum()), "{}")
    put("valStep", 1 / (len(val_target) - val_target.sum()), "{:.3f}")
    for name, (sens, spec, reported) in TOPOCAM_TABLE9.items():
        rebuilt = sens * target.mean() + spec * (1 - target.mean())
        assert abs(rebuilt - reported) < 0.002, (name, rebuilt, reported)
        put(f"topoRebuilt{name}", rebuilt, "{:.3f}")
    gflops = {arm: flops(arm) for arm in ARMS}
    for arm, value in gflops.items():
        put(f"gflops{arm}", value, "{:.1f}")
    put("gflopsRatio", gflops["Kinetics"] / gflops["Scratch"], "{:.1f}")
    put("nTrain", sizes["train"], "{}")
    put("nVal", sizes["val"], "{}")
    put("nTest", sizes["test"], "{}")
    put("nTestExc", int(target.sum()), "{}")
    put("nTestInh", int(len(target) - target.sum()), "{}")
    put("prevalence", float(target.mean()), "{:.3f}")
    put("nSeeds", len(SEEDS), "{}")
    put("replicates", REPLICATES, "{}")
    put("bootReps", reference.metrics["bootstrap"]["replicates"], "{}")
    env = reference.info["environment"]
    put("gpu", env["gpu"].replace("NVIDIA GeForce ", ""))
    put("torchVersion", env["packages"]["torch"].split("+")[0])
    put("torchvisionVersion", env["packages"]["torchvision"].split("+")[0])
    put("lightningVersion", env["packages"]["lightning"])
    put("monaiVersion", env["packages"]["monai"])
    put("medmnistVersion", env["packages"]["medmnist"])
    put("cudaVersion", env["cuda_version"])

    (PAPER / "generated").mkdir(exist_ok=True)
    (PAPER / "generated/numbers.tex").write_text(
        "% Written by analysis/paper_numbers.py. Do not edit.\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items())
        + "\n"
    )
    (PAPER / "generated/seeds.json").write_text(
        json.dumps(
            {"runs": scored, "paired": paired, "pooled": pooled, "baselines": baselines}, indent=1
        )
        + "\n"
    )
    for name, value in macros.items():
        print(f"{name}\t{value}")
    print(
        f"all {len(every)} runs match metrics.json and the medmnist evaluator; {len(macros)} macros"
    )


if __name__ == "__main__":
    main()
