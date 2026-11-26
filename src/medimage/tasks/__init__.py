"""Tasks connect a dataset, a model, and the metrics.

A task knows how to size the model for a dataset, compute the training loss, turn model
output into per-sample predictions, and score those predictions. The trainer in
:mod:`medimage.engine` is the same for every task; only the task object changes.

Adding a task means subclassing :class:`Task` and registering it::

    @TASKS.register("segmentation")
    class Segmentation(Task): ...
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn

from medimage.data.base import DatasetInfo, collate_classification, collate_detection
from medimage.metrics import classification as cm
from medimage.metrics.bootstrap import bootstrap as run_bootstrap
from medimage.metrics.detection import DetectionEvaluator
from medimage.registry import TASKS

Prediction = dict[str, Any]


class Task:
    """Base class. Subclasses set ``task_type`` and ``default_monitor``."""

    task_type: ClassVar[str]
    default_monitor: ClassVar[str]

    def __init__(self, info: DatasetInfo) -> None:
        if info.task_type != self.task_type:
            raise ValueError(
                f"{type(self).__name__} needs a {self.task_type} dataset, got {info.task_type}"
            )
        self.info = info

    def model_kwargs(self) -> dict[str, Any]:
        raise NotImplementedError

    def collate(self, samples: list[dict]) -> dict:
        raise NotImplementedError

    def loss(self, model: nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, model: nn.Module, batch: dict, device: torch.device) -> list[Prediction]:
        raise NotImplementedError

    def calibrate(self, predictions: list[Prediction]) -> dict[str, Any]:
        """Settings chosen on validation predictions and reused on test, e.g. a threshold."""
        return {}

    def metrics(
        self,
        predictions: list[Prediction],
        calibration: dict[str, Any] | None = None,
        bootstrap: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score predictions. With ``bootstrap`` settings, each metric becomes an estimate with a CI."""
        raise NotImplementedError

    def prediction_rows(self, predictions: list[Prediction]) -> list[dict[str, Any]]:
        raise NotImplementedError

    @staticmethod
    def _score(
        statistic: Callable[[np.ndarray], dict[str, float]],
        n: int,
        groups: Sequence[str],
        bootstrap: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not bootstrap:
            return statistic(np.arange(n))
        estimates = run_bootstrap(
            statistic,
            n,
            groups=groups if bootstrap.get("unit", "patient") == "patient" else None,
            replicates=bootstrap.get("replicates", 1000),
            confidence=bootstrap.get("confidence", 0.95),
            seed=bootstrap.get("seed", 0),
        )
        return {k: v.to_dict() for k, v in estimates.items()}


@TASKS.register("classification")
class Classification(Task):
    """Binary, multiclass, or multilabel image classification.

    Binary models emit one logit trained with BCE; multiclass emit one logit per class trained
    with cross-entropy; multilabel emit one logit per label trained with BCE. ``pos_weight``
    (binary or multilabel) reweights positives for class imbalance.
    """

    task_type = "classification"
    default_monitor = "auroc"

    def __init__(
        self,
        info: DatasetInfo,
        pos_weight: float | list[float] | None = None,
        label_smoothing: float = 0.0,
        threshold: str | float = "youden",
        calibration_bins: int = 15,
    ) -> None:
        super().__init__(info)
        if info.label_mode is None:
            raise ValueError("classification datasets must set label_mode")
        self.mode = info.label_mode
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.threshold = threshold
        self.calibration_bins = calibration_bins

    @property
    def num_outputs(self) -> int:
        return 1 if self.mode == "binary" else self.info.num_classes

    def model_kwargs(self) -> dict[str, Any]:
        return {
            "in_channels": self.info.in_channels,
            "spatial_dims": self.info.spatial_dims,
            "num_classes": self.num_outputs,
        }

    def collate(self, samples: list[dict]) -> dict:
        return collate_classification(samples)

    def loss(self, model: nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
        logits = model(batch["image"].to(device, non_blocking=True))
        target = batch["label"].to(device, non_blocking=True)
        if self.mode == "multiclass":
            return nn.functional.cross_entropy(logits, target, label_smoothing=self.label_smoothing)
        if self.mode == "binary":
            logits, target = logits.reshape(-1), target.float().reshape(-1)
        pos_weight = (
            None
            if self.pos_weight is None
            else torch.as_tensor(self.pos_weight, dtype=logits.dtype, device=device)
        )
        return nn.functional.binary_cross_entropy_with_logits(
            logits, target.float(), pos_weight=pos_weight
        )

    @torch.no_grad()
    def predict(self, model: nn.Module, batch: dict, device: torch.device) -> list[Prediction]:
        logits = model(batch["image"].to(device, non_blocking=True)).float()
        if self.mode == "multiclass":
            probs = logits.softmax(dim=1)
        else:
            probs = logits.sigmoid()
            if self.mode == "binary":
                probs = probs.reshape(-1, 1)
        probs_np = probs.cpu().numpy()
        labels = batch["label"].cpu().numpy()
        return [
            {"id": i, "patient_id": p, "target": labels[n], "prob": probs_np[n]}
            for n, (i, p) in enumerate(zip(batch["id"], batch["patient_id"], strict=True))
        ]

    def _arrays(self, predictions: list[Prediction]) -> tuple[np.ndarray, np.ndarray]:
        target = np.stack([np.asarray(p["target"]) for p in predictions])
        prob = np.stack([np.asarray(p["prob"]) for p in predictions])
        return target, prob

    def calibrate(self, predictions: list[Prediction]) -> dict[str, Any]:
        if self.mode != "binary":
            return {}
        if isinstance(self.threshold, int | float):
            return {"threshold": float(self.threshold), "threshold_source": "config"}
        target, prob = self._arrays(predictions)
        return {
            "threshold": cm.youden_threshold(target, prob[:, 0]),
            "threshold_source": "youden on val",
        }

    def metrics(
        self,
        predictions: list[Prediction],
        calibration: dict[str, Any] | None = None,
        bootstrap: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target, prob = self._arrays(predictions)
        names = self.info.class_names
        threshold = (calibration or {}).get("threshold", 0.5)

        def statistic(index: np.ndarray) -> dict[str, float]:
            t, p = target[index], prob[index]
            if self.mode == "binary":
                out = {
                    "auroc": cm.auroc(t, p[:, 0]),
                    "auprc": cm.auprc(t, p[:, 0]),
                    "brier": cm.brier(t, p[:, 0]),
                    "ece": cm.expected_calibration_error(t, p[:, 0], self.calibration_bins),
                }
                out |= cm.threshold_metrics(t, p[:, 0], threshold)
                return out
            if self.mode == "multiclass":
                per_class = cm.multiclass_ovr(t, p, cm.auroc)
                prc = cm.multiclass_ovr(t, p, cm.auprc)
                predicted = p.argmax(axis=1)
                out = {
                    "auroc": _nanmean(per_class),
                    "auprc": _nanmean(prc),
                    "accuracy": float(np.mean(predicted == t)),
                    "balanced_accuracy": cm.balanced_accuracy(t, predicted, len(names)),
                }
            else:
                per_class = np.array([cm.auroc(t[:, k], p[:, k]) for k in range(p.shape[1])])
                prc = np.array([cm.auprc(t[:, k], p[:, k]) for k in range(p.shape[1])])
                out = {"auroc": _nanmean(per_class), "auprc": _nanmean(prc)}
            for name, a, b in zip(names, per_class, prc, strict=True):
                out[f"auroc/{name}"] = float(a)
                out[f"auprc/{name}"] = float(b)
            return out

        result = self._score(
            statistic, len(predictions), [p["patient_id"] for p in predictions], bootstrap
        )
        result["n_samples"] = len(predictions)
        result["n_patients"] = len({p["patient_id"] for p in predictions})
        if self.mode == "binary":
            result["prevalence"] = float(np.mean(target))
            result["threshold"] = threshold
        return result

    def prediction_rows(self, predictions: list[Prediction]) -> list[dict[str, Any]]:
        rows = []
        for p in predictions:
            row = {"id": p["id"], "patient_id": p["patient_id"]}
            target = np.asarray(p["target"]).reshape(-1)
            if self.mode == "multilabel":
                row |= {
                    f"target/{n}": float(v)
                    for n, v in zip(self.info.class_names, target, strict=True)
                }
            else:
                row["target"] = int(target[0])
            if self.mode == "binary":
                row[f"prob/{self.info.class_names[1]}"] = float(p["prob"][0])
            else:
                row |= {
                    f"prob/{n}": float(v)
                    for n, v in zip(self.info.class_names, p["prob"], strict=True)
                }
            rows.append(row)
        return rows


@TASKS.register("detection")
class Detection(Task):
    """2D object detection with torchvision-style detectors.

    Dataset labels start at 0 for the first foreground class; the model sees them shifted by
    one because torchvision reserves 0 for background.
    """

    task_type = "detection"
    default_monitor = "map50"

    def __init__(
        self,
        info: DatasetInfo,
        iou_thresholds: list[float] | None = None,
        max_dets: int = 100,
        froc_iou: float = 0.5,
        froc_rates: list[float] | None = None,
    ) -> None:
        super().__init__(info)
        if info.spatial_dims != 2:
            raise ValueError("detection supports 2D images only")
        kwargs: dict[str, Any] = {"max_dets": max_dets, "froc_iou": froc_iou}
        if iou_thresholds is not None:
            kwargs["iou_thresholds"] = iou_thresholds
        if froc_rates is not None:
            kwargs["froc_rates"] = froc_rates
        self.evaluator = DetectionEvaluator(info.num_classes, **kwargs)

    def model_kwargs(self) -> dict[str, Any]:
        return {
            "in_channels": self.info.in_channels,
            "spatial_dims": 2,
            "num_classes": self.info.num_classes + 1,
        }

    def collate(self, samples: list[dict]) -> dict:
        return collate_detection(samples)

    def loss(self, model: nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
        images = [image.to(device) for image in batch["image"]]
        targets = [
            {"boxes": t["boxes"].to(device), "labels": t["labels"].to(device) + 1}
            for t in batch["target"]
        ]
        losses = model(images, targets)
        return sum(losses.values())  # type: ignore[return-value]

    @torch.no_grad()
    def predict(self, model: nn.Module, batch: dict, device: torch.device) -> list[Prediction]:
        outputs = model([image.to(device) for image in batch["image"]])
        predictions = []
        for out, target, sample_id, patient in zip(
            outputs, batch["target"], batch["id"], batch["patient_id"], strict=True
        ):
            predictions.append(
                {
                    "id": sample_id,
                    "patient_id": patient,
                    "boxes": out["boxes"].float().cpu().numpy(),
                    "scores": out["scores"].float().cpu().numpy(),
                    "labels": out["labels"].cpu().numpy() - 1,
                    "gt_boxes": target["boxes"].numpy(),
                    "gt_labels": target["labels"].numpy(),
                }
            )
        return predictions

    def metrics(
        self,
        predictions: list[Prediction],
        calibration: dict[str, Any] | None = None,
        bootstrap: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        matched = [
            self.evaluator.match_image(
                p["boxes"], p["scores"], p["labels"], p["gt_boxes"], p["gt_labels"]
            )
            for p in predictions
        ]
        result = self._score(
            lambda index: self.evaluator.compute(matched, index),
            len(predictions),
            [p["patient_id"] for p in predictions],
            bootstrap,
        )
        result["n_images"] = len(predictions)
        result["n_patients"] = len({p["patient_id"] for p in predictions})
        result["n_lesions"] = int(sum(len(p["gt_boxes"]) for p in predictions))
        return result

    def prediction_rows(self, predictions: list[Prediction]) -> list[dict[str, Any]]:
        rows = []
        for p in predictions:
            for box, score, label in zip(p["boxes"], p["scores"], p["labels"], strict=True):
                rows.append(
                    {
                        "id": p["id"],
                        "patient_id": p["patient_id"],
                        "label": self.info.class_names[int(label)],
                        "score": float(score),
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                    }
                )
        return rows


def _nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values[~np.isnan(values)])) if np.any(~np.isnan(values)) else float("nan")


__all__ = ["Classification", "Detection", "Task"]
