"""2D object detection with torchvision detectors, scored by torchmetrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torchmetrics.detection import MeanAveragePrecision

from opendiagnostics.data import DatasetInfo
from opendiagnostics.modules.base import DiagnosticModule, Prediction

METRICS = ("map", "map_50", "map_75", "mar_100")


class ObjectDetector(DiagnosticModule):
    """A torchvision detector such as ``fasterrcnn_mobilenet_v3_large_fpn``.

    Its ``num_classes`` counts background, so it is the dataset's class count plus one.
    Dataset labels start at 0; the network sees them shifted by one.
    """

    default_monitor = "map_50"

    def __init__(self, network: nn.Module, info: DatasetInfo | None = None) -> None:
        super().__init__(network, info)
        info = self.info
        if info.label_mode != "detection":
            raise ValueError("ObjectDetector needs a detection data module")

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        targets = [{"boxes": t["boxes"], "labels": t["labels"] + 1} for t in batch["target"]]
        loss = torch.stack(list(self.network(batch["image"], targets).values())).sum()
        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=len(targets))
        return loss

    def predict_batch(self, batch: dict) -> list[Prediction]:
        outputs = self.network(batch["image"])
        return [
            {
                "id": sample_id,
                "patient_id": patient,
                "boxes": out["boxes"].float().cpu(),
                "scores": out["scores"].float().cpu(),
                "labels": out["labels"].cpu() - 1,
                "gt_boxes": target["boxes"].cpu(),
                "gt_labels": target["labels"].cpu(),
            }
            for out, target, sample_id, patient in zip(
                outputs, batch["target"], batch["id"], batch["patient_id"], strict=True
            )
        ]

    def score(
        self,
        predictions: list[Prediction],
        calibration: dict[str, Any] | None = None,
        bootstrap: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        def statistic(index: np.ndarray) -> dict[str, float]:
            metric = MeanAveragePrecision()
            chosen = [predictions[i] for i in index]
            metric.update(
                [
                    {"boxes": p["boxes"], "scores": p["scores"], "labels": p["labels"]}
                    for p in chosen
                ],
                [{"boxes": p["gt_boxes"], "labels": p["gt_labels"]} for p in chosen],
            )
            values = metric.compute()
            return {k: float(values[k]) for k in METRICS}

        patients = [p["patient_id"] for p in predictions]
        result = self.bootstrapped(statistic, len(predictions), patients, bootstrap)
        result["n_images"] = len(predictions)
        result["n_patients"] = len(set(patients))
        result["n_lesions"] = int(sum(len(p["gt_boxes"]) for p in predictions))
        return result

    def prediction_rows(self, predictions: list[Prediction]) -> list[dict[str, Any]]:
        return [
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
            for p in predictions
            for box, score, label in zip(p["boxes"], p["scores"], p["labels"], strict=True)
        ]
