"""Classification metrics in plain NumPy.

The ranking metrics follow scikit-learn's definitions exactly (the tests check this):

* AUROC is the Mann-Whitney U statistic with ties counted as one half.
* AUPRC is average precision, ``sum_n (R_n - R_{n-1}) P_n`` over distinct score thresholds,
  without the trapezoidal interpolation that overstates the area.

A metric that is undefined for the given data (AUROC with one class present) returns NaN.
"""

from __future__ import annotations

import numpy as np


def _midranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    boundaries = np.flatnonzero(np.diff(sorted_scores)) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(scores)]
    for start, stop in zip(starts, stops, strict=True):
        ranks[order[start:stop]] = (start + stop + 1) / 2  # mean of 1-based ranks start+1..stop
    return ranks


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _midranks(y_score)
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y_score, y_true = y_score[order], y_true[order]
    last_of_tie = np.r_[np.flatnonzero(np.diff(y_score)), len(y_true) - 1]
    tps = np.cumsum(y_true)[last_of_tie]
    fps = last_of_tie + 1 - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """The score threshold that maximizes sensitivity + specificity - 1.

    Choose it on validation data and apply it unchanged to the test set.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.all() or not y_true.any():
        return 0.5
    order = np.argsort(-y_score, kind="mergesort")
    sorted_scores, sorted_true = y_score[order], y_true[order]
    last_of_tie = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_true) - 1]
    tpr = np.cumsum(sorted_true)[last_of_tie] / sorted_true.sum()
    fpr = np.cumsum(~sorted_true)[last_of_tie] / (~sorted_true).sum()
    # Predicting positive for score >= this threshold gives the best J; ties keep the highest.
    return float(sorted_scores[last_of_tie[np.argmax(tpr - fpr)]])


def threshold_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(bool)
    predicted = np.asarray(y_score) >= threshold
    tp = float(np.sum(predicted & y_true))
    tn = float(np.sum(~predicted & ~y_true))
    fp = float(np.sum(predicted & ~y_true))
    fn = float(np.sum(~predicted & y_true))

    def ratio(a: float, b: float) -> float:
        return a / b if b else float("nan")

    return {
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "accuracy": ratio(tp + tn, len(y_true)),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_prob, float) - np.asarray(y_true, float)) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    """Equal-width-bin ECE: the bin-size-weighted gap between confidence and accuracy."""
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(y_prob, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            error += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(error)


def multiclass_ovr(y_true: np.ndarray, y_prob: np.ndarray, metric) -> np.ndarray:
    """Apply a binary metric one-vs-rest to each class column."""
    y_true = np.asarray(y_true)
    return np.array([metric(y_true == k, y_prob[:, k]) for k in range(y_prob.shape[1])])


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    recalls = [np.mean(y_pred[y_true == k] == k) for k in range(num_classes) if np.any(y_true == k)]
    return float(np.mean(recalls)) if recalls else float("nan")
