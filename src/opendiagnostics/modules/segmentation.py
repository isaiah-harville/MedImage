"""Semantic segmentation of 2D or 3D images with MONAI networks, losses and inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from monai.data import MetaTensor
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.transforms import ResampleToMatch
from torch import nn

from opendiagnostics.data import DatasetInfo
from opendiagnostics.metrics import classification as cm
from opendiagnostics.metrics.segmentation import case_metrics
from opendiagnostics.modules.base import DiagnosticModule, Prediction


class SegmentationModule(DiagnosticModule):
    """A network that maps an image to one logit per class and voxel, such as MONAI's
    ``DynUNet``, ``UNet`` or ``SwinUNETR``, with ``out_channels`` equal to the number of
    classes, background included.

    Training uses ``loss`` (MONAI's ``DiceCELoss`` on softmax outputs by default) on the
    patches the data module crops. Evaluation runs MONAI's sliding-window inference with
    ``patch_size`` windows overlapping by ``overlap``, blended with Gaussian weights. When
    the label is on another voxel grid than the image (the label was left out of
    ``Spacingd``), the logits are resampled onto the label's grid one class at a time before
    the argmax, so cases are scored in their original resolution.

    Each case gets Dice and HD95 per region. ``regions`` maps a name to the label values it
    unions, for benchmarks that score overlapping regions (KiTS23's hierarchical evaluation
    classes, the BraTS tumour regions); by default each foreground class is a region. The
    headline ``dice`` is the mean over regions of each region's mean over cases.
    ``ignore_empty`` sets Dice to NaN where the reference lacks the region instead of
    scoring it 1 or 0; see :mod:`opendiagnostics.metrics.segmentation`.
    """

    default_monitor = "dice"

    def __init__(
        self,
        network: nn.Module,
        patch_size: list[int],
        info: DatasetInfo | None = None,
        loss: nn.Module | None = None,
        regions: dict[str, list[int]] | None = None,
        sw_batch_size: int = 2,
        overlap: float = 0.5,
        ignore_empty: bool = False,
    ) -> None:
        super().__init__(network, info)
        info = self.info
        if info.label_mode != "segmentation":
            raise ValueError("SegmentationModule needs a segmentation data module")
        if len(patch_size) != info.spatial_dims:
            raise ValueError(
                f"patch_size {patch_size} needs {info.spatial_dims} values for this dataset"
            )
        self.loss = loss or DiceCELoss(to_onehot_y=True, softmax=True)
        names = info.class_names
        self.regions = regions or {name: [k] for k, name in enumerate(names) if k > 0}
        unknown = {v for values in self.regions.values() for v in values} - set(range(len(names)))
        if unknown:
            raise ValueError(
                f"regions name label values {sorted(unknown)} outside 0..{len(names) - 1}"
            )
        self.ignore_empty = ignore_empty
        self.inferer = SlidingWindowInferer(
            roi_size=patch_size, sw_batch_size=sw_batch_size, overlap=overlap, mode="gaussian"
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.network(images)
        if logits.ndim != images.ndim or logits.shape[1] != self.info.num_classes:
            raise ValueError(
                f"the network returned logits shaped {tuple(logits.shape)}; this dataset needs "
                f"(batch, {self.info.num_classes}, *spatial). Set the network's out_channels to "
                f"{self.info.num_classes}, background included."
            )
        return logits

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["image"])
        loss = self.loss(logits, batch["target"].long())
        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=len(batch["id"]))
        return loss

    @torch.no_grad()
    def segment(
        self,
        image: torch.Tensor,
        image_affine: torch.Tensor,
        target_affine: torch.Tensor,
        target_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Class indices for one ``(C, *spatial)`` image, on the target's voxel grid."""
        logits = self.inferer(image[None], self)
        assert isinstance(logits, torch.Tensor)
        logits = logits[0].float()
        if tuple(logits.shape[1:]) == tuple(target_shape) and torch.allclose(
            image_affine.double(), target_affine.double()
        ):
            return logits.argmax(0)
        resample = ResampleToMatch(mode="bilinear", padding_mode="border")
        grid = MetaTensor(
            torch.zeros((1, *target_shape), device=logits.device), affine=target_affine
        )
        best = labels = None
        for k in range(logits.shape[0]):
            channel = MetaTensor(logits[k : k + 1], affine=image_affine)
            values = torch.as_tensor(resample(channel, grid))[0]
            if best is None or labels is None:
                best, labels = values, torch.zeros_like(values, dtype=torch.long)
            else:
                better = values > best
                best = torch.where(better, values, best)
                labels[better] = k
        assert labels is not None
        return labels

    def predict_batch(self, batch: dict, distances: bool = True) -> list[Prediction]:
        predictions = []
        for n, (sample, patient) in enumerate(zip(batch["id"], batch["patient_id"], strict=True)):
            target = batch["target"][n, 0]
            labels = self.segment(
                batch["image"][n],
                batch["image_affine"][n],
                batch["target_affine"][n],
                tuple(target.shape),
            )
            metrics = case_metrics(
                labels,
                target.long(),
                self.regions,
                batch["spacing"][n],
                distances=distances,
                ignore_empty=self.ignore_empty,
            )
            predictions.append({"id": sample, "patient_id": patient} | metrics)
        return predictions

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._val_predictions.extend(self.predict_batch(batch, distances=False))

    def score(
        self,
        predictions: list[Prediction],
        calibration: dict[str, Any] | None = None,
        bootstrap: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        dice = np.stack([p["dice"] for p in predictions])
        hd95 = np.stack([p["hd95"] for p in predictions])
        with_distances = bool(np.any(~np.isnan(hd95)))
        names = list(self.regions)

        def statistic(index: np.ndarray) -> dict[str, float]:
            per_region = [cm.nanmean(dice[index, r]) for r in range(len(names))]
            out = {"dice": cm.nanmean(np.asarray(per_region))}
            out |= {f"dice/{n}": v for n, v in zip(names, per_region, strict=True)}
            if with_distances:
                distance = [cm.nanmean(hd95[index, r]) for r in range(len(names))]
                out["hd95"] = cm.nanmean(np.asarray(distance))
                out |= {f"hd95/{n}": v for n, v in zip(names, distance, strict=True)}
            return out

        patients = [p["patient_id"] for p in predictions]
        result = self.bootstrapped(statistic, len(predictions), patients, bootstrap)
        result["n_cases"] = len(predictions)
        result["n_patients"] = len(set(patients))
        return result

    def prediction_rows(self, predictions: list[Prediction]) -> list[dict[str, Any]]:
        rows = []
        for p in predictions:
            row: dict[str, Any] = {"id": p["id"], "patient_id": p["patient_id"]}
            row |= {f"dice/{n}": float(v) for n, v in zip(self.regions, p["dice"], strict=True)}
            row |= {f"hd95/{n}": float(v) for n, v in zip(self.regions, p["hd95"], strict=True)}
            rows.append(row)
        return rows
