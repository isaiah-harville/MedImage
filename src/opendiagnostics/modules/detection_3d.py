"""3D lesion detection with MONAI's RetinaNet, scored by FROC and the LUNA16 CPM."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.networks.nets import resnet
from torch import nn

from opendiagnostics.data import DatasetInfo
from opendiagnostics.data.volumes import voxel_boxes_to_world
from opendiagnostics.metrics import froc
from opendiagnostics.modules.base import DiagnosticModule, Prediction


class VolumeDetector(DiagnosticModule):
    """A RetinaNet over a MONAI 3D ``ResNet`` backbone (``network``), with an FPN on
    ``returned_layers`` and MONAI's ATSS matcher, hard negative sampler and NMS.

    ``base_anchor_shapes`` are anchor sizes in voxels of the resampled grid at the finest
    level; each coarser level doubles them. With ``roi_size``, evaluation runs a sliding
    window of that size (``sw_overlap`` overlap) and stitches on the CPU to bound GPU
    memory; without it, whole volumes go through the network at once. Predictions are
    mapped to world LPS millimetres and scored against the world boxes with FROC.
    """

    default_monitor = "cpm"

    def __init__(
        self,
        network: nn.Module,
        info: DatasetInfo | None = None,
        returned_layers: list[int] | None = None,
        base_anchor_shapes: list[list[int]] | None = None,
        roi_size: list[int] | None = None,
        sw_overlap: float = 0.25,
        score_thresh: float = 0.02,
        nms_thresh: float = 0.22,
        detections_per_image: int = 300,
    ) -> None:
        if info is None or info.label_mode != "detection" or info.spatial_dims != 3:
            raise ValueError("VolumeDetector needs a 3D detection data module")
        if info.num_classes != 1:
            raise ValueError(
                f"VolumeDetector scores FROC over one lesion class; the data has {info.class_names}"
            )
        if not isinstance(network, resnet.ResNet):
            raise ValueError("network must be a monai.networks.nets.ResNet backbone")
        channels = network.conv1.in_channels
        if channels != info.in_channels:
            raise ValueError(
                f"the backbone takes {channels} input channels but the data has "
                f"{info.in_channels}; set the ResNet's n_input_channels to {info.in_channels}"
            )
        returned_layers = returned_layers or [1, 2]
        anchors = AnchorGeneratorWithAnchorShape(
            feature_map_scales=[2**level for level in range(len(returned_layers) + 1)],
            base_anchor_shapes=base_anchor_shapes or [[6, 8, 4], [8, 6, 5], [10, 10, 6]],
        )
        extractor = resnet_fpn_feature_extractor(
            network, spatial_dims=3, returned_layers=returned_layers
        )
        stride = network.conv1.stride
        retinanet = RetinaNet(
            spatial_dims=3,
            num_classes=info.num_classes,
            num_anchors=anchors.num_anchors_per_location()[0],
            feature_extractor=extractor,
            size_divisible=[int(s) * 2 * 2 ** max(returned_layers) for s in stride],
        )
        detector = RetinaNetDetector(retinanet, anchors)
        detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
        detector.set_hard_negative_sampler(
            batch_size_per_image=64, positive_fraction=0.3, pool_size=20, min_neg=16
        )
        detector.set_box_selector_parameters(
            score_thresh=score_thresh,
            topk_candidates_per_level=1000,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_image,
        )
        if roi_size is not None:
            detector.set_sliding_window_inferer(
                roi_size=roi_size, overlap=sw_overlap, sw_batch_size=1, device="cpu"
            )
        super().__init__(detector, info)
        self.use_inferer = roi_size is not None

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        targets = batch["target"]
        losses = self.network(batch["image"], targets)
        loss = losses[self.network.cls_key] + losses[self.network.box_reg_key]
        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=len(targets))
        return loss

    @torch.no_grad()
    def predict_batch(self, batch: dict) -> list[Prediction]:
        images = batch["image"]
        images = list(images) if isinstance(images, torch.Tensor) else images
        outputs = self.network(images, use_inferer=self.use_inferer)
        predictions = []
        for out, target, affine, sample_id, patient in zip(
            outputs, batch["target"], batch["affine"], batch["id"], batch["patient_id"], strict=True
        ):
            voxel = out[self.network.target_box_key].detach().double().cpu()
            world = voxel_boxes_to_world(voxel, affine.cpu())
            predictions.append(
                {
                    "id": sample_id,
                    "patient_id": patient,
                    "boxes": world.numpy(),
                    "scores": out[self.network.pred_score_key].detach().float().cpu().numpy(),
                    "truth": target["world_boxes"].cpu().numpy(),
                    "ignore": target["world_ignore"].cpu().numpy(),
                }
            )
        return predictions

    def score(
        self,
        predictions: list[Prediction],
        calibration: dict[str, Any] | None = None,
        bootstrap: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        hits = [
            froc.match_scan(p["boxes"], p["scores"], p["truth"], p["ignore"]) for p in predictions
        ]

        def statistic(index: np.ndarray) -> dict[str, float]:
            return froc.froc_metrics([hits[i] for i in index])

        patients = [p["patient_id"] for p in predictions]
        result = self.bootstrapped(statistic, len(predictions), patients, bootstrap)
        result["n_scans"] = len(predictions)
        result["n_patients"] = len(set(patients))
        result["n_lesions"] = int(sum(h.lesions for h in hits))
        return result

    def prediction_rows(self, predictions: list[Prediction]) -> list[dict[str, Any]]:
        name = self.info.class_names[0]
        return [
            row
            for p in predictions
            for row in froc.scan_rows(
                froc.Scan(
                    id=p["id"],
                    patient_id=p["patient_id"],
                    boxes=p["boxes"],
                    scores=p["scores"],
                    truth=p["truth"],
                    ignore=p["ignore"],
                ),
                name,
            )
        ]
