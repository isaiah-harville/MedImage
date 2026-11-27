"""A binary decision threshold fitted on out-of-fold predictions.

A threshold fitted on a small validation split is noisy. Training ``k`` runs with
``split: kfold`` (each holding out one fold of the pooled official train and validation
splits) gives every pooled sample a prediction from a model that never trained on it. The
Youden threshold on those out-of-fold predictions uses all of them, and the official test
split stays untouched.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from opendiagnostics.metrics import classification as cm
from opendiagnostics.metrics.compare import read_predictions


class CrossValidationError(ValueError):
    """Raised when fold runs cannot give a valid out-of-fold threshold."""


def resolve_run(path: Path) -> Path:
    """``path`` if it is a run, else the newest run inside it (``runs/<name>/``)."""
    if (path / "splits.json").exists():
        return path
    runs = sorted(p for p in path.glob("*") if (p / "splits.json").exists())
    if not runs:
        raise CrossValidationError(f"{path} is not a run and holds no runs")
    return runs[-1]


def _ids(run: Path) -> dict[str, list[str]]:
    return json.loads((run / "splits.json").read_text())["ids"]


def out_of_fold_threshold(run: Path, folds: Sequence[Path]) -> dict[str, Any]:
    """The Youden threshold on the fold runs' validation predictions, as a calibration record.

    The folds' held-out sets must not overlap, must not contain any of ``run``'s test samples,
    and must together cover ``run``'s train and validation samples.
    """
    folds = [resolve_run(fold) for fold in folds]
    if len(folds) < 2:
        raise CrossValidationError("give at least two fold runs")
    test = set(_ids(run)["test"])
    pool = set(_ids(run)["train"]) | set(_ids(run)["val"])
    held: set[str] = set()
    for fold in folds:
        val = set(_ids(fold)["val"])
        if val & held:
            raise CrossValidationError(f"{fold.name} holds out samples another fold also holds out")
        if val & test:
            raise CrossValidationError(
                f"{fold.name} holds out samples from {run.name}'s test split"
            )
        held |= val
    if held != pool:
        raise CrossValidationError(
            f"the folds hold out {len(held)} samples, but {run.name} trained and validated on "
            f"{len(pool)}; every pooled sample needs exactly one out-of-fold prediction"
        )
    predictions = [read_predictions(fold / "predictions_val.csv") for fold in folds]
    if any(p.prob.shape[1] != 1 for p in predictions):
        raise CrossValidationError("an out-of-fold threshold is for binary runs only")
    target = np.concatenate([p.target for p in predictions])
    score = np.concatenate([p.prob[:, 0] for p in predictions])
    return {
        "threshold": cm.youden_threshold(target, score),
        "threshold_source": f"youden on out-of-fold predictions from {len(folds)} folds",
        "folds": [fold.name for fold in folds],
        "samples": len(target),
    }
