"""Per-case segmentation metrics from MONAI: Dice and the 95th-percentile Hausdorff distance.

Each region is a set of label values scored as one binary mask, so a region can be a single
class (``liver: [1]``) or a union such as KiTS23's ``kidney_and_masses: [1, 2, 3]``.

Dice follows the KiTS23 convention by default: 1 when the region is absent from both the
reference and the prediction, 0 when it is in only one of them. ``ignore_empty`` makes it
NaN whenever the reference is empty, as MONAI's ``ignore_empty`` does. HD95 is in the
units of ``spacing`` (millimetres for NIfTI) and NaN when either mask is empty, because a
distance to nothing is undefined.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from monai.metrics import compute_dice, compute_hausdorff_distance


def _bounding_box(mask: torch.Tensor) -> tuple[slice, ...]:
    """The smallest box holding every true voxel, one voxel wider on each side."""
    index = mask.nonzero()
    low = (index.min(0).values - 1).clamp_min(0)
    high = index.max(0).values + 2
    return tuple(slice(int(a), int(b)) for a, b in zip(low, high, strict=True))


def hd95(pred: torch.Tensor, target: torch.Tensor, spacing: Sequence[float]) -> float:
    """HD95 between two boolean masks of the same spatial shape, NaN if either is empty."""
    if not pred.any() or not target.any():
        return float("nan")
    box = _bounding_box(pred | target)
    distance = compute_hausdorff_distance(
        pred[box][None, None].float(),
        target[box][None, None].float(),
        include_background=True,
        percentile=95,
        spacing=list(spacing),
    )
    return float(distance)


def dice(pred: torch.Tensor, target: torch.Tensor, ignore_empty: bool = False) -> float:
    """Dice between two boolean masks of the same spatial shape."""
    score = compute_dice(
        pred[None, None].float(),
        target[None, None].float(),
        include_background=True,
        ignore_empty=ignore_empty,
    )
    return float(score)


def case_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    regions: Mapping[str, Sequence[int]],
    spacing: Sequence[float],
    distances: bool = True,
    ignore_empty: bool = False,
) -> dict[str, np.ndarray]:
    """Dice and HD95 for each region of one case, from integer label maps.

    ``pred`` and ``target`` hold label values shaped like the spatial dimensions. With
    ``distances`` false, HD95 is skipped (NaN), which saves time during training.
    """
    if pred.shape != target.shape:
        raise ValueError(f"prediction {tuple(pred.shape)} and label {tuple(target.shape)} differ")
    dices, distances_mm = [], []
    for values in regions.values():
        wanted = torch.as_tensor(list(values), device=target.device)
        p, t = torch.isin(pred, wanted), torch.isin(target, wanted)
        dices.append(dice(p, t, ignore_empty))
        distances_mm.append(hd95(p, t, spacing) if distances else float("nan"))
    return {"dice": np.asarray(dices), "hd95": np.asarray(distances_mm)}
