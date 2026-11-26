"""Built-in model builders.

Every builder receives ``in_channels``, ``spatial_dims``, and ``num_classes`` from the task
(sized from the dataset), plus the ``params`` from the config. See ``docs/design.md`` for why
classification backbones come from timm, detectors from torchvision, and 3D networks from
MONAI.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from opendiagnostics.registry import MODELS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _conv(spatial_dims: int) -> type[nn.Module]:
    return nn.Conv2d if spatial_dims == 2 else nn.Conv3d


def _norm(spatial_dims: int) -> type[nn.Module]:
    return nn.BatchNorm2d if spatial_dims == 2 else nn.BatchNorm3d


def _pool(spatial_dims: int) -> type[nn.Module]:
    return nn.MaxPool2d if spatial_dims == 2 else nn.MaxPool3d


class SimpleCNN(nn.Module):
    """A small VGG-style network for 2D or 3D inputs. A baseline and a fast test model."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        spatial_dims: int = 2,
        width: int = 16,
        depth: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        conv, norm, pool = _conv(spatial_dims), _norm(spatial_dims), _pool(spatial_dims)
        layers: list[nn.Module] = []
        channels = in_channels
        for i in range(depth):
            out = width * 2**i
            layers += [
                conv(channels, out, 3, padding=1, bias=False),
                norm(out),
                nn.ReLU(inplace=True),
            ]
            layers += [conv(out, out, 3, padding=1, bias=False), norm(out), nn.ReLU(inplace=True)]
            if i < depth - 1:
                layers.append(pool(2))
            channels = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = features.mean(dim=tuple(range(2, features.ndim)))
        return self.head(pooled)


@MODELS.register("simple_cnn")
def simple_cnn(
    in_channels: int, num_classes: int, spatial_dims: int = 2, **params: Any
) -> nn.Module:
    return SimpleCNN(in_channels, num_classes, spatial_dims, **params)


@MODELS.register("timm")
def timm_model(
    in_channels: int,
    num_classes: int,
    spatial_dims: int = 2,
    model_name: str = "resnet18",
    pretrained: bool = False,
    **params: Any,
) -> nn.Module:
    """Any 2D classifier from timm, e.g. ``resnet50``, ``convnext_tiny``, ``efficientnet_b0``.

    timm adapts pretrained first-layer weights to ``in_channels`` itself, so grayscale input
    works with ImageNet weights.
    """
    if spatial_dims != 2:
        raise ValueError("timm models are 2D; use the 'monai' model or 'simple_cnn' for 3D")
    import timm

    return timm.create_model(
        model_name, pretrained=pretrained, in_chans=in_channels, num_classes=num_classes, **params
    )


@MODELS.register("monai")
def monai_model(
    in_channels: int,
    num_classes: int,
    spatial_dims: int = 2,
    net: str = "DenseNet121",
    **params: Any,
) -> nn.Module:
    """A network from ``monai.networks.nets`` (needs the ``monai`` extra).

    ``spatial_dims``, ``in_channels``, and ``out_channels`` are passed when the network's
    constructor accepts them; anything else goes in ``params``.
    """
    try:
        from monai.networks import nets
    except ImportError as error:
        raise ImportError(
            "the 'monai' model needs MONAI: pip install 'opendiagnostics[monai]'"
        ) from error
    cls = getattr(nets, net)
    accepted = inspect.signature(cls).parameters
    sizing = {"spatial_dims": spatial_dims, "in_channels": in_channels, "out_channels": num_classes}
    kwargs = {k: v for k, v in sizing.items() if k in accepted} | params
    return cls(**kwargs)


class TinyBackbone(nn.Sequential):
    """A four-layer stride-8 feature extractor so detection tests run in seconds on CPU."""

    def __init__(self, in_channels: int, out_channels: int = 32) -> None:
        layers: list[nn.Module] = []
        channels = in_channels
        for out in (16, 32, out_channels):
            layers += [
                nn.Conv2d(channels, out, 3, stride=2, padding=1),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True),
            ]
            channels = out
        super().__init__(*layers)
        self.out_channels = out_channels


def _detection_kwargs(
    in_channels: int,
    pretrained: bool,
    min_size: int,
    max_size: int,
    image_mean: Sequence[float] | None,
    image_std: Sequence[float] | None,
) -> dict[str, Any]:
    if pretrained and in_channels != 3:
        raise ValueError(
            "pretrained detection backbones expect 3 channels; add the repeat_channels transform"
        )
    if image_mean is None:
        image_mean = IMAGENET_MEAN if pretrained else (0.0,) * in_channels
    if image_std is None:
        image_std = IMAGENET_STD if pretrained else (1.0,) * in_channels
    return {
        "min_size": min_size,
        "max_size": max_size,
        "image_mean": list(image_mean),
        "image_std": list(image_std),
    }


@MODELS.register("fasterrcnn")
def fasterrcnn(
    in_channels: int,
    num_classes: int,
    spatial_dims: int = 2,
    backbone: str = "resnet50_fpn_v2",
    weights: str | None = None,
    weights_backbone: str | None = None,
    min_size: int = 512,
    max_size: int = 1024,
    image_mean: Sequence[float] | None = None,
    image_std: Sequence[float] | None = None,
    **params: Any,
) -> nn.Module:
    """torchvision Faster R-CNN. ``num_classes`` counts foreground classes plus background.

    ``backbone`` is ``resnet50_fpn_v2``, ``resnet50_fpn``, ``mobilenet_v3_large_fpn``,
    ``mobilenet_v3_large_320_fpn``, or ``tiny``. ``weights: coco`` starts from COCO weights
    and replaces the box predictor; ``weights_backbone: imagenet`` starts only the backbone
    from ImageNet. Images are resized by the model to ``min_size``/``max_size``.
    """
    if spatial_dims != 2:
        raise ValueError("fasterrcnn is 2D only")
    from torchvision.models import detection
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    kwargs = (
        _detection_kwargs(
            in_channels,
            bool(weights or weights_backbone),
            min_size,
            max_size,
            image_mean,
            image_std,
        )
        | params
    )
    if backbone == "tiny":
        if weights or weights_backbone:
            raise ValueError("the tiny backbone has no pretrained weights")
        from torchvision.models.detection.anchor_utils import AnchorGenerator
        from torchvision.ops import MultiScaleRoIAlign

        anchors = AnchorGenerator(sizes=((8, 16, 32, 64),), aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pool = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
        return detection.FasterRCNN(
            TinyBackbone(in_channels),
            num_classes=num_classes,
            rpn_anchor_generator=anchors,
            box_roi_pool=roi_pool,
            **kwargs,
        )
    builder = getattr(detection, f"fasterrcnn_{backbone}", None)
    if builder is None:
        raise ValueError(f"unknown fasterrcnn backbone {backbone!r}")
    if weights == "coco":
        model = builder(weights="DEFAULT", **kwargs)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    if weights is not None:
        raise ValueError("weights must be null or 'coco'")
    return builder(
        weights=None,
        weights_backbone="DEFAULT" if weights_backbone == "imagenet" else None,
        num_classes=num_classes,
        **kwargs,
    )


def _one_stage(name: str):
    def build(
        in_channels: int,
        num_classes: int,
        spatial_dims: int = 2,
        weights_backbone: str | None = None,
        min_size: int = 512,
        max_size: int = 1024,
        image_mean: Sequence[float] | None = None,
        image_std: Sequence[float] | None = None,
        **params: Any,
    ) -> nn.Module:
        if spatial_dims != 2:
            raise ValueError(f"{name} is 2D only")
        from torchvision.models import detection

        kwargs = (
            _detection_kwargs(
                in_channels, bool(weights_backbone), min_size, max_size, image_mean, image_std
            )
            | params
        )
        builder = getattr(
            detection, f"{name}_resnet50_fpn" + ("_v2" if name == "retinanet" else "")
        )
        return builder(
            weights=None,
            weights_backbone="DEFAULT" if weights_backbone == "imagenet" else None,
            num_classes=num_classes,
            **kwargs,
        )

    build.__doc__ = (
        f"torchvision {name} with a ResNet-50 FPN backbone. ``num_classes`` includes background."
    )
    return build


MODELS.register("retinanet")(_one_stage("retinanet"))
MODELS.register("fcos")(_one_stage("fcos"))

__all__ = ["SimpleCNN", "TinyBackbone"]
