"""Classification metrics on NumPy arrays.

AUROC, average precision, and the ROC curve come from scikit-learn in float64, as in the
official MedMNIST evaluator; torchmetrics computes them in float32, which drifted from
scikit-learn by up to 1e-4 on the Windows CUDA build. Calibration and the threshold counts
come from ``torchmetrics.functional``.

Each function returns NaN where the metric is undefined (AUROC on one class, for example),
so the bootstrap can leave that replicate out instead of averaging a misleading zero.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import ArrayLike
from sklearn import metrics as skm
from torchmetrics.functional import classification as tmc


def _t(values: ArrayLike, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=dtype)


def _both_classes(target: ArrayLike) -> bool:
    values = np.asarray(target)
    return bool(np.any(values == 1) and np.any(values == 0))


def auroc(target: ArrayLike, score: ArrayLike) -> float:
    if not _both_classes(target):
        return float("nan")
    return float(skm.roc_auc_score(np.asarray(target), np.asarray(score, np.float64)))


def auprc(target: ArrayLike, score: ArrayLike) -> float:
    if not np.any(np.asarray(target) == 1):
        return float("nan")
    return float(skm.average_precision_score(np.asarray(target), np.asarray(score, np.float64)))


def brier(target: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(prob, np.float64) - np.asarray(target, np.float64)) ** 2))


def expected_calibration_error(target: np.ndarray, prob: np.ndarray, bins: int = 15) -> float:
    return float(tmc.binary_calibration_error(_t(prob), _t(target, torch.long), n_bins=bins))


def youden_threshold(target: ArrayLike, score: ArrayLike) -> float:
    """The threshold that maximizes sensitivity + specificity - 1 on these scores."""
    if not _both_classes(target):
        return 0.5
    fpr, tpr, thresholds = skm.roc_curve(np.asarray(target), np.asarray(score, np.float64))
    return float(min(thresholds[int(np.argmax(tpr - fpr))], 1.0))


def threshold_metrics(target: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    """Counts at ``score >= threshold``, the convention ROC thresholds use. torchmetrics
    thresholds with ``>``, so the scores are binarized here first."""
    predicted = _t(np.asarray(score) >= threshold, torch.long)
    tp, fp, tn, fn, _ = (int(v) for v in tmc.binary_stat_scores(predicted, _t(target, torch.long)))

    def ratio(a: int, b: int) -> float:
        return a / b if b else float("nan")

    return {
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "accuracy": ratio(tp + tn, tp + fp + tn + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def per_class(target: np.ndarray, prob: np.ndarray, metric) -> np.ndarray:
    """One-vs-rest ``metric`` for each column of ``prob``. ``target`` is class indices or a
    multilabel indicator matrix."""
    indicator = target if target.ndim == 2 else np.eye(prob.shape[1], dtype=np.int64)[target]
    return np.array([metric(indicator[:, k], prob[:, k]) for k in range(prob.shape[1])])


def nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, np.float64)
    return float(np.mean(values[~np.isnan(values)])) if np.any(~np.isnan(values)) else float("nan")


def balanced_accuracy(target: np.ndarray, predicted: np.ndarray, num_classes: int) -> float:
    present = np.unique(target)
    recall = tmc.multiclass_recall(
        _t(predicted, torch.long), _t(target, torch.long), num_classes=num_classes, average="none"
    ).numpy()
    return float(np.mean(recall[present]))
