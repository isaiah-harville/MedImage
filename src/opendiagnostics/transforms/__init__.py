"""Sample transforms that work on 2D and 3D images alike.

Each transform takes and returns a sample dict (see :mod:`opendiagnostics.data.base`). Spatial
transforms also update detection ``boxes`` so images and annotations stay aligned. Boxes are
supported for 2D images; a spatial transform given 3D boxes raises instead of guessing.

Random transforms draw from torch's global generator. The trainer seeds it and gives each
DataLoader worker its own derived seed, so augmentation is reproducible.

Transforms are registered by name and built from config lists such as::

    train_transforms:
      - {name: resize, params: {size: [224, 224]}}
      - {name: random_flip, params: {axes: [-1]}}
      - {name: normalize_intensity, params: {mode: zscore}}
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from opendiagnostics.data.base import Sample
from opendiagnostics.registry import TRANSFORMS


class Compose:
    def __init__(self, transforms: Sequence[Callable[[Sample], Sample]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: Sample) -> Sample:
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __repr__(self) -> str:
        return f"Compose({self.transforms!r})"


def build_transforms(specs: Sequence[Any]) -> Compose:
    """Build a :class:`Compose` from config components (objects with ``name`` and ``params``)."""
    return Compose([TRANSFORMS.build(spec.name, **spec.params) for spec in specs])


def _spatial_dims(image: torch.Tensor) -> int:
    if image.ndim not in (3, 4):
        raise ValueError(
            f"expected a (C, H, W) or (C, D, H, W) image, got shape {tuple(image.shape)}"
        )
    return image.ndim - 1


def _check_boxes_2d(sample: Sample, name: str) -> None:
    boxes = sample.get("boxes")
    if boxes is not None and (sample["image"].ndim != 3 or boxes.shape[-1] != 4):
        raise NotImplementedError(f"{name} supports boxes on 2D images only")


@TRANSFORMS.register("resize")
class Resize:
    """Resize to ``size`` (one int for every axis, or one per spatial axis)."""

    def __init__(self, size: int | Sequence[int], mode: str | None = None) -> None:
        self.size = size
        self.mode = mode

    def __call__(self, sample: Sample) -> Sample:
        _check_boxes_2d(sample, "resize")
        image = sample["image"]
        dims = _spatial_dims(image)
        size = [self.size] * dims if isinstance(self.size, int) else list(self.size)
        if len(size) != dims:
            raise ValueError(f"resize size {size} does not match a {dims}D image")
        mode = self.mode or ("bilinear" if dims == 2 else "trilinear")
        old = image.shape[1:]
        resized = F.interpolate(
            image[None].float(),
            size=size,
            mode=mode,
            align_corners=False if mode in ("bilinear", "trilinear") else None,
            antialias=mode == "bilinear",
        )[0]
        out = {**sample, "image": resized}
        if sample.get("boxes") is not None:
            scale_y, scale_x = size[0] / old[0], size[1] / old[1]
            out["boxes"] = sample["boxes"] * torch.tensor([scale_x, scale_y, scale_x, scale_y])
        return out

    def __repr__(self) -> str:
        return f"Resize({self.size})"


@TRANSFORMS.register("random_flip")
class RandomFlip:
    """Flip each listed spatial axis independently with probability ``p``.

    ``axes`` are spatial axis indices (``-1`` is width). Left-right flips change laterality,
    so leave them out for tasks where side matters.
    """

    def __init__(self, axes: Sequence[int] = (-1,), p: float = 0.5) -> None:
        self.axes = list(axes)
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        _check_boxes_2d(sample, "random_flip")
        image = sample["image"]
        dims = _spatial_dims(image)
        out = dict(sample)
        for axis in self.axes:
            spatial_axis = axis % dims
            if torch.rand(()) >= self.p:
                continue
            out["image"] = out["image"].flip(1 + spatial_axis)
            boxes = out.get("boxes")
            if boxes is not None and len(boxes):
                extent = image.shape[1 + spatial_axis]
                boxes = boxes.clone()
                lo, hi = (0, 2) if spatial_axis == 1 else (1, 3)  # width axis is x
                boxes[:, lo], boxes[:, hi] = extent - boxes[:, hi], extent - boxes[:, lo]
                out["boxes"] = boxes
        return out

    def __repr__(self) -> str:
        return f"RandomFlip(axes={self.axes}, p={self.p})"


@TRANSFORMS.register("normalize_intensity")
class NormalizeIntensity:
    """Per-image intensity normalization.

    ``zscore`` subtracts the mean and divides by the standard deviation; ``minmax`` maps to
    [0, 1]. With ``nonzero`` the statistics come from nonzero voxels only, the usual choice
    for skull-stripped MRI where the background is zero.
    """

    def __init__(
        self,
        mode: str = "zscore",
        nonzero: bool = False,
        per_channel: bool = True,
        eps: float = 1e-8,
    ) -> None:
        if mode not in ("zscore", "minmax"):
            raise ValueError(f"mode must be 'zscore' or 'minmax', got {mode!r}")
        self.mode, self.nonzero, self.per_channel, self.eps = mode, nonzero, per_channel, eps

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        mask = x != 0 if self.nonzero else torch.ones_like(x, dtype=torch.bool)
        if not mask.any():
            return x
        values = x[mask]
        if self.mode == "zscore":
            result = (x - values.mean()) / (values.std(unbiased=False) + self.eps)
        else:
            result = (x - values.min()) / (values.max() - values.min() + self.eps)
        return torch.where(mask, result, x) if self.nonzero else result

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"].float()
        if self.per_channel:
            image = torch.stack([self._apply(c) for c in image])
        else:
            image = self._apply(image)
        return {**sample, "image": image}

    def __repr__(self) -> str:
        return f"NormalizeIntensity({self.mode!r}, nonzero={self.nonzero})"


@TRANSFORMS.register("scale_intensity_range")
class ScaleIntensityRange:
    """Linearly map ``[a_min, a_max]`` to ``[b_min, b_max]``; CT windowing in Hounsfield units.

    For example a lung window is ``a_min=-1350, a_max=150``.
    """

    def __init__(
        self, a_min: float, a_max: float, b_min: float = 0.0, b_max: float = 1.0, clip: bool = True
    ) -> None:
        if a_max <= a_min:
            raise ValueError("a_max must be greater than a_min")
        self.a_min, self.a_max, self.b_min, self.b_max, self.clip = a_min, a_max, b_min, b_max, clip

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"].float()
        image = (image - self.a_min) / (self.a_max - self.a_min) * (
            self.b_max - self.b_min
        ) + self.b_min
        if self.clip:
            image = image.clamp(self.b_min, self.b_max)
        return {**sample, "image": image}


@TRANSFORMS.register("clip_percentile")
class ClipPercentile:
    """Clip each image to its own ``[lower, upper]`` percentiles to tame outliers."""

    def __init__(self, lower: float = 0.5, upper: float = 99.5) -> None:
        self.lower, self.upper = lower / 100, upper / 100

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"].float()
        flat = image.flatten()
        if flat.numel() > 16_000_000:  # torch.quantile has an input size limit
            flat = flat[torch.randperm(flat.numel())[:16_000_000]]
        lo, hi = torch.quantile(flat, torch.tensor([self.lower, self.upper]))
        return {**sample, "image": image.clamp(lo, hi)}


@TRANSFORMS.register("scale")
class Scale:
    """Multiply by ``factor``; for example ``1/255`` for 8-bit images."""

    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(self, sample: Sample) -> Sample:
        return {**sample, "image": sample["image"].float() * self.factor}


@TRANSFORMS.register("normalize")
class Normalize:
    """Subtract a fixed per-channel ``mean`` and divide by ``std`` (ImageNet statistics, say)."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"].float()
        shape = (-1,) + (1,) * (image.ndim - 1)
        return {**sample, "image": (image - self.mean.view(shape)) / self.std.view(shape)}


@TRANSFORMS.register("repeat_channels")
class RepeatChannels:
    """Repeat a single channel ``n`` times, for 3-channel pretrained backbones."""

    def __init__(self, n: int = 3) -> None:
        self.n = n

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"]
        if image.shape[0] != 1:
            raise ValueError(f"repeat_channels expects 1 channel, got {image.shape[0]}")
        return {**sample, "image": image.repeat(self.n, *([1] * (image.ndim - 1)))}


@TRANSFORMS.register("random_intensity")
class RandomIntensity:
    """Random affine intensity change ``x * (1 + a) + b`` with ``a, b`` uniform in ``±scale, ±shift``."""

    def __init__(self, scale: float = 0.1, shift: float = 0.1, p: float = 0.5) -> None:
        self.scale, self.shift, self.p = scale, shift, p

    def __call__(self, sample: Sample) -> Sample:
        if torch.rand(()) >= self.p:
            return sample
        a = (torch.rand(()) * 2 - 1) * self.scale
        b = (torch.rand(()) * 2 - 1) * self.shift
        return {**sample, "image": sample["image"].float() * (1 + a) + b}


__all__ = ["Compose", "build_transforms"]
