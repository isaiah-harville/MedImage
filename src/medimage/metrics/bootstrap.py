"""Percentile bootstrap confidence intervals, optionally resampling whole patients.

With several images per patient, resampling images treats correlated samples as
independent and gives intervals that are too narrow. The cluster bootstrap resamples
patients with replacement and keeps all of a patient's images together (Field and Welsh,
"Bootstrapping clustered data", JRSS-B 69(3), 2007, https://doi.org/10.1111/j.1467-9868.2007.00593.x).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Estimate:
    """A point estimate with a percentile bootstrap interval."""

    value: float
    lower: float
    upper: float
    confidence: float
    replicates: int
    valid_replicates: int
    unit: str

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        pct = round(self.confidence * 100)
        return f"{self.value:.4f} [{pct}% CI {self.lower:.4f}, {self.upper:.4f}]"


def cluster_indices(groups: Sequence[str]) -> list[np.ndarray]:
    """Sample indices for each distinct group, in first-seen order."""
    order: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        order.setdefault(group, []).append(index)
    return [np.asarray(v, dtype=np.int64) for v in order.values()]


def bootstrap(
    statistic: Callable[[np.ndarray], dict[str, float]],
    n: int,
    groups: Sequence[str] | None = None,
    replicates: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Estimate]:
    """Bootstrap every metric returned by ``statistic``.

    ``statistic`` takes an index array into the data and returns named metric values.
    Replicates where a metric is NaN (for example AUROC on a resample with no positives) are
    left out of that metric's interval, and ``valid_replicates`` reports how many remained.
    """
    point = statistic(np.arange(n))
    unit = "sample" if groups is None else "patient"
    clusters = cluster_indices(groups) if groups is not None else None
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(replicates):
        if clusters is None:
            index = rng.integers(0, n, size=n)
        else:
            chosen = rng.integers(0, len(clusters), size=len(clusters))
            index = np.concatenate([clusters[c] for c in chosen])
        for key, value in statistic(index).items():
            draws[key].append(value)
    alpha = (1 - confidence) / 2
    result = {}
    for key, value in point.items():
        values = np.asarray(draws[key], dtype=np.float64)
        values = values[~np.isnan(values)]
        if len(values):
            lower, upper = np.quantile(values, [alpha, 1 - alpha])
        else:
            lower = upper = float("nan")
        result[key] = Estimate(
            value=float(value),
            lower=float(lower),
            upper=float(upper),
            confidence=confidence,
            replicates=replicates,
            valid_replicates=len(values),
            unit=unit,
        )
    return result
