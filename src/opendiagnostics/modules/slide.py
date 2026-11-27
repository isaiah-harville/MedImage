"""Slide-level classification from bags of cached tile features."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from opendiagnostics.data import DatasetInfo
from opendiagnostics.modules.classification import ImageClassifier


class SlideClassifier(ImageClassifier):
    """An :class:`ImageClassifier` whose samples are ``(N, D)`` bags of tile features.

    ``network`` is a multiple-instance network over ``(B, N, D, 1, 1)`` input, such as
    ``monai.networks.nets.MILModel`` with ``backbone: torch.nn.Flatten`` and
    ``backbone_num_features: D``: the features are already extracted, so its backbone only
    flattens them. There is no test-time flip, because a feature bag has no orientation, and
    no attribution figures.
    """

    def __init__(
        self,
        network: nn.Module,
        info: DatasetInfo | None = None,
        threshold: float | Literal["youden"] = "youden",
        label_smoothing: float = 0.0,
        pos_weight: list[float] | None = None,
        calibration_bins: int = 15,
    ) -> None:
        super().__init__(
            network,
            info,
            threshold,
            label_smoothing,
            pos_weight=pos_weight,
            calibration_bins=calibration_bins,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 3:
            raise ValueError(
                f"SlideClassifier takes (batch, tiles, features) bags, got {tuple(images.shape)}; "
                "use a SlideFeatureDataModule"
            )
        return super().forward(images[..., None, None])
