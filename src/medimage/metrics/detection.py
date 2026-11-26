"""Detection metrics: COCO-style mean average precision and FROC.

mAP follows the COCO evaluation protocol as implemented in ``pycocotools`` (the tests check
agreement): greedy matching per image and class in descending score order, at most
``max_dets`` detections per image and class, 101-point interpolated precision, and the mean
over classes that have ground truth. Area ranges and crowd regions are not used.

FROC (free-response ROC) reports lesion-level sensitivity at fixed numbers of false
positives per image, the standard for lesion detection (e.g. LUNA16, Setio et al., Medical
Image Analysis 42, 2017, https://doi.org/10.1016/j.media.2017.06.015). A detection that hits
a lesion another detection already found is ignored rather than counted as a false positive,
as in LUNA16. The competition performance metric (CPM) is the mean sensitivity at 1/8, 1/4,
1/2, 1, 2, 4, and 8 false positives per image. Sensitivity at a rate is the highest reached
without exceeding that rate (a step function, with no interpolation).

Every metric is computed from per-image match results, so the bootstrap only re-aggregates
and never re-matches.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

COCO_IOUS = tuple(float(t) for t in np.linspace(0.5, 0.95, 10))
RECALL_POINTS = np.linspace(0.0, 1.0, 101)
FROC_RATES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of ``(N, 4)`` and ``(M, 4)`` boxes in ``(x1, y1, x2, y2)`` order."""
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


@dataclass
class ImageResult:
    """Per-image match results, reused by every bootstrap replicate."""

    # Per class: detection scores (sorted, descending), TP flags (T, D), and ground-truth count.
    class_scores: list[np.ndarray]
    class_tp: list[np.ndarray]
    class_gt: np.ndarray
    # FROC, class-agnostic: scores and outcomes (1 TP, 0 FP, -1 ignored) and lesion count.
    froc_scores: np.ndarray
    froc_outcome: np.ndarray
    froc_gt: int


def _greedy_match(iou: np.ndarray, threshold: float) -> np.ndarray:
    """COCO matching: each detection, in order, takes the unmatched GT with the highest IoU."""
    matched = np.zeros(iou.shape[1], dtype=bool)
    tp = np.zeros(iou.shape[0], dtype=bool)
    for d in range(iou.shape[0]):
        best, best_iou = -1, min(threshold, 1 - 1e-10)
        for g in range(iou.shape[1]):
            if matched[g] or iou[d, g] < best_iou:
                continue
            best, best_iou = g, iou[d, g]
        if best >= 0:
            matched[best] = True
            tp[d] = True
    return tp


class DetectionEvaluator:
    def __init__(
        self,
        num_classes: int,
        iou_thresholds: Sequence[float] = COCO_IOUS,
        max_dets: int = 100,
        froc_iou: float = 0.5,
        froc_rates: Sequence[float] = FROC_RATES,
    ) -> None:
        self.num_classes = num_classes
        self.iou_thresholds = np.asarray(iou_thresholds, np.float64)
        self.max_dets = max_dets
        self.froc_iou = froc_iou
        self.froc_rates = tuple(froc_rates)

    def match_image(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
    ) -> ImageResult:
        pred_boxes = np.asarray(pred_boxes, np.float64).reshape(-1, 4)
        pred_scores = np.asarray(pred_scores, np.float64).reshape(-1)
        pred_labels = np.asarray(pred_labels, np.int64).reshape(-1)
        gt_boxes = np.asarray(gt_boxes, np.float64).reshape(-1, 4)
        gt_labels = np.asarray(gt_labels, np.int64).reshape(-1)

        class_scores, class_tp = [], []
        class_gt = np.zeros(self.num_classes, dtype=np.int64)
        for k in range(self.num_classes):
            det = np.flatnonzero(pred_labels == k)
            det = det[np.argsort(-pred_scores[det], kind="mergesort")][: self.max_dets]
            gt = np.flatnonzero(gt_labels == k)
            class_gt[k] = len(gt)
            iou = box_iou(pred_boxes[det], gt_boxes[gt])
            tp = (
                np.stack([_greedy_match(iou, t) for t in self.iou_thresholds])
                if len(det)
                else np.zeros((len(self.iou_thresholds), 0), bool)
            )
            class_scores.append(pred_scores[det])
            class_tp.append(tp)

        order = np.argsort(-pred_scores, kind="mergesort")
        iou = box_iou(pred_boxes[order], gt_boxes)
        hit = np.zeros(len(gt_boxes), dtype=bool)
        outcome = np.zeros(len(order), dtype=np.int8)
        for d in range(len(order)):
            candidates = np.flatnonzero(iou[d] >= self.froc_iou)
            if len(candidates) == 0:
                continue  # false positive
            free = candidates[~hit[candidates]]
            if len(free):
                hit[free[np.argmax(iou[d, free])]] = True
                outcome[d] = 1
            else:
                outcome[d] = -1  # a second hit on a lesion already found
        return ImageResult(
            class_scores, class_tp, class_gt, pred_scores[order], outcome, len(gt_boxes)
        )

    def compute(
        self, results: Sequence[ImageResult], index: np.ndarray | None = None
    ) -> dict[str, float]:
        chosen = [results[i] for i in (range(len(results)) if index is None else index)]
        metrics: dict[str, float] = {}

        ap = np.full((len(self.iou_thresholds), self.num_classes), np.nan)
        for k in range(self.num_classes):
            n_gt = sum(int(r.class_gt[k]) for r in chosen)
            if n_gt == 0:
                continue
            scores = np.concatenate([r.class_scores[k] for r in chosen])
            tp = np.concatenate([r.class_tp[k] for r in chosen], axis=1)
            order = np.argsort(-scores, kind="mergesort")
            tp = tp[:, order]
            tps = np.cumsum(tp, axis=1, dtype=np.float64)
            fps = np.cumsum(~tp, axis=1, dtype=np.float64)
            for t in range(len(self.iou_thresholds)):
                recall = tps[t] / n_gt
                precision = tps[t] / np.maximum(tps[t] + fps[t], np.finfo(np.float64).eps)
                precision = np.maximum.accumulate(precision[::-1])[::-1]
                positions = np.searchsorted(recall, RECALL_POINTS, side="left")
                sampled = (
                    np.where(
                        positions < len(precision),
                        precision[np.minimum(positions, len(precision) - 1)],
                        0.0,
                    )
                    if len(precision)
                    else np.zeros(len(RECALL_POINTS))
                )
                ap[t, k] = sampled.mean()

        with np.errstate(invalid="ignore"):
            per_threshold = (
                np.nanmean(ap, axis=1)
                if np.isfinite(ap).any()
                else np.full(len(self.iou_thresholds), np.nan)
            )
        metrics["map"] = (
            float(np.nanmean(per_threshold)) if np.isfinite(per_threshold).any() else float("nan")
        )
        for label, value in (("map50", 0.5), ("map75", 0.75)):
            match = np.flatnonzero(np.isclose(self.iou_thresholds, value))
            metrics[label] = float(per_threshold[match[0]]) if len(match) else float("nan")
        first = (
            0
            if not np.isclose(self.iou_thresholds, 0.5).any()
            else int(np.flatnonzero(np.isclose(self.iou_thresholds, 0.5))[0])
        )
        for k in range(self.num_classes):
            metrics[f"ap50/{k}"] = float(ap[first, k])

        total_gt = sum(r.froc_gt for r in chosen)
        scores = np.concatenate([r.froc_scores for r in chosen]) if chosen else np.zeros(0)
        outcome = (
            np.concatenate([r.froc_outcome for r in chosen]) if chosen else np.zeros(0, np.int8)
        )
        keep = outcome >= 0
        scores, outcome = scores[keep], outcome[keep]
        order = np.argsort(-scores, kind="mergesort")
        outcome = outcome[order]
        if total_gt == 0:
            sensitivities = [float("nan")] * len(self.froc_rates)
        else:
            sensitivity = np.r_[0.0, np.cumsum(outcome == 1) / total_gt]
            fp_rate = np.r_[0.0, np.cumsum(outcome == 0) / max(len(chosen), 1)]
            sensitivities = [float(sensitivity[fp_rate <= rate].max()) for rate in self.froc_rates]
        for rate, value in zip(self.froc_rates, sensitivities, strict=True):
            metrics[f"froc@{rate:g}"] = value
        metrics["cpm"] = float(np.mean(sensitivities))
        return metrics
