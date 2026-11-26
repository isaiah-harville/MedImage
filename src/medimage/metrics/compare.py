"""Paired comparison of two classifiers scored on the same test samples.

A claim that model B beats model A needs the uncertainty of the *difference*, not two
overlapping intervals. Both models' predictions are aligned on sample ID, and whole
patients are resampled together, so each replicate scores both models on the same
resample. The two-sided p-value is twice the smaller tail of the bootstrap distribution of
the difference at zero: a percentile-bootstrap test of "no difference".
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from medimage.metrics import classification as cm
from medimage.metrics.bootstrap import bootstrap_draws

METRICS = {"auroc": cm.auroc, "auprc": cm.auprc}


@dataclass(frozen=True)
class Predictions:
    ids: list[str]
    patients: list[str]
    target: np.ndarray
    prob: np.ndarray
    classes: list[str]


@dataclass(frozen=True)
class Comparison:
    metric: str
    a: float
    b: float
    difference: float
    lower: float
    upper: float
    p_value: float
    confidence: float
    replicates: int
    valid_replicates: int
    samples: int
    patients: int

    def to_dict(self) -> dict:
        return asdict(self)


def read_predictions(path: Path) -> Predictions:
    """Read a ``predictions_<split>.csv`` written by a binary or multiclass run."""
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "target" not in rows[0]:
        raise ValueError(f"{path} is not a binary or multiclass predictions file")
    columns = [c for c in rows[0] if c.startswith("prob/")]
    return Predictions(
        ids=[r["id"] for r in rows],
        patients=[r["patient_id"] for r in rows],
        target=np.array([int(r["target"]) for r in rows]),
        prob=np.array([[float(r[c]) for c in columns] for r in rows]),
        classes=[c.removeprefix("prob/") for c in columns],
    )


def score(metric: str, target: np.ndarray, prob: np.ndarray) -> float:
    """Binary: the metric on the positive-class column. Multiclass: one-vs-rest macro mean."""
    fn = METRICS[metric]
    if prob.shape[1] == 1:
        return fn(target, prob[:, 0])
    per_class = cm.multiclass_ovr(target, prob, fn)
    return float(np.nanmean(per_class)) if np.isfinite(per_class).any() else float("nan")


def compare(
    a: Predictions,
    b: Predictions,
    metric: str = "auroc",
    replicates: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Comparison:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}, got {metric!r}")
    if a.classes != b.classes:
        raise ValueError(f"the runs predict different classes: {a.classes} and {b.classes}")
    position = {sample: i for i, sample in enumerate(b.ids)}
    shared = [i for i, sample in enumerate(a.ids) if sample in position]
    if len(shared) != len(a.ids) or len(shared) != len(b.ids):
        raise ValueError(
            f"the runs scored different samples ({len(a.ids)} and {len(b.ids)}, "
            f"{len(shared)} shared); compare runs on the same split"
        )
    ia = np.array(shared)
    ib = np.array([position[a.ids[i]] for i in shared])
    if not np.array_equal(a.target[ia], b.target[ib]):
        raise ValueError("the runs disagree on ground-truth labels for the same sample IDs")
    target, prob_a, prob_b = a.target[ia], a.prob[ia], b.prob[ib]

    def statistic(index: np.ndarray) -> dict[str, float]:
        sa = score(metric, target[index], prob_a[index])
        sb = score(metric, target[index], prob_b[index])
        return {"a": sa, "b": sb, "difference": sb - sa}

    patients = [a.patients[i] for i in shared]
    point, draws = bootstrap_draws(statistic, len(shared), patients, replicates, seed)
    diff = draws["difference"][~np.isnan(draws["difference"])]
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(diff, [alpha, 1 - alpha]) if len(diff) else (np.nan, np.nan)
    p_value = min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))) if len(diff) else np.nan
    return Comparison(
        metric=metric,
        a=point["a"],
        b=point["b"],
        difference=point["difference"],
        lower=float(lower),
        upper=float(upper),
        p_value=float(p_value),
        confidence=confidence,
        replicates=replicates,
        valid_replicates=len(diff),
        samples=len(shared),
        patients=len(set(patients)),
    )
