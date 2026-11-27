"""Networks that need more than a ``class_path`` to fit this pipeline."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision.models import video

VideoArchitecture = Literal["r3d_18", "mc3_18", "r2plus1d_18"]

_BUILDERS = {
    "r3d_18": (video.r3d_18, video.R3D_18_Weights.KINETICS400_V1),
    "mc3_18": (video.mc3_18, video.MC3_18_Weights.KINETICS400_V1),
    "r2plus1d_18": (video.r2plus1d_18, video.R2Plus1D_18_Weights.KINETICS400_V1),
}


class KineticsVideoResNet(nn.Module):
    """A torchvision video ResNet for one-channel ``(B, 1, D, H, W)`` volumes in ``[0, 1]``.

    The volume's depth axis plays the role of time. The channel is repeated to three and
    normalized with the Kinetics-400 statistics, so the pretrained stem sees the input range
    it was trained on, and the 400-way head is replaced by ``num_classes`` outputs.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self, architecture: VideoArchitecture, num_classes: int, pretrained: bool = True
    ) -> None:
        super().__init__()
        build, weights = _BUILDERS[architecture]
        chosen = weights if pretrained else None
        self.network = build(weights=chosen)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)
        stats = video.R3D_18_Weights.KINETICS400_V1.transforms()
        self.register_buffer("mean", torch.tensor(stats.mean).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor(stats.std).view(1, 3, 1, 1, 1))

    def forward(self, volumes: torch.Tensor) -> torch.Tensor:
        if volumes.shape[1] != 1:
            raise ValueError(
                f"KineticsVideoResNet takes one-channel volumes, got {volumes.shape[1]} channels"
            )
        rgb = volumes.expand(-1, 3, -1, -1, -1)
        return self.network((rgb - self.mean) / self.std)
