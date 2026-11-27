"""Example figures for segmentation runs: an axial slice beside the predicted classes.

The slice is the one with the most labelled voxels, shown on the label's voxel grid. The
examples are the test cases with the highest and lowest mean Dice, taken in turn, so the
page shows failures as well as successes.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import ResampleToMatch
from torch.nn import functional as F
from torchvision.utils import draw_segmentation_masks

from opendiagnostics.data.segmentation import SegmentationDataModule, collate_cases
from opendiagnostics.modules import SegmentationModule


def choose_segmentation(rows: Sequence[dict[str, str]], limit: int) -> list[dict[str, Any]]:
    """The cases with the highest and lowest mean Dice over regions, taken in turn."""
    columns = [c for c in rows[0] if c.startswith("dice/")] if rows else []
    scored: list[tuple[float, str]] = []
    for row in rows:
        values = np.array([float(row[c]) for c in columns])
        if np.any(~np.isnan(values)):
            scored.append((float(np.nanmean(values)), row["id"]))
    ranked = [{"id": i, "dice": d} for d, i in sorted(scored, key=lambda s: (-s[0], s[1]))]
    picks: list[dict[str, Any]] = []
    while ranked and len(picks) < limit:
        picks.append(ranked.pop(0))
        if ranked and len(picks) < limit:
            picks.append(ranked.pop())
    return picks


def _axial(volume: torch.Tensor, index: int) -> torch.Tensor:
    """Slice ``index`` of the last axis of an ``(X, Y, Z)`` RAS volume, anterior up."""
    return volume[..., index].T.flip(0)


@torch.no_grad()
def segmentation_pair(
    module: SegmentationModule, case: dict[str, Any], dice: float
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """The slice as ``uint8 (3, H, W)``, the same slice with predicted classes over it, and
    a caption naming the classes in the reference and in the prediction."""
    from opendiagnostics.examples import _scale, _to_rgb

    batch = collate_cases([case])
    target = batch["target"][0, 0].long()
    labels = module.segment(
        batch["image"][0].to(module.device),
        batch["image_affine"][0],
        batch["target_affine"][0],
        tuple(target.shape),
    ).cpu()
    image = MetaTensor(batch["image"][0, :1], affine=batch["image_affine"][0])
    grid = MetaTensor(target[None].float(), affine=batch["target_affine"][0])
    image = torch.as_tensor(ResampleToMatch(mode="bilinear")(image, grid))[0]
    counts = (target > 0).flatten(0, -2).sum(0)
    index = int(counts.argmax()) if counts.any() else target.shape[-1] // 2
    shown = _axial(image, index).float()
    low, high = torch.quantile(shown.flatten(), torch.tensor([0.01, 0.99]))
    shown = ((shown - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    rgb, factor = _scale(_to_rgb(shown[None]))
    predicted = _axial(labels, index).float()
    if factor > 1:
        predicted = F.interpolate(predicted[None, None], scale_factor=factor)[0, 0]
    names = module.info.class_names
    classes = [k for k in range(1, len(names)) if (predicted == k).any()]
    overlay = rgb
    if classes:
        overlay = draw_segmentation_masks(rgb, torch.stack([predicted == k for k in classes]), 0.45)
    caption = {
        "truth": ", ".join(names[k] for k in range(1, len(names)) if (target == k).any()) or "none",
        "prediction": ", ".join(names[k] for k in classes) or "none",
        "confidence": dice,
        "correct": None,
    }
    return rgb, overlay, caption


def write_segmentation_examples(
    module: SegmentationModule, data: SegmentationDataModule, run_dir: Path, out: Path, limit: int
) -> list[dict[str, Any]]:
    """Render up to ``limit`` test cases into ``out`` and return their manifest entries."""
    from opendiagnostics.examples import _save

    with (run_dir / "predictions_test.csv").open(newline="") as handle:
        picks = choose_segmentation(list(csv.DictReader(handle)), limit)
    cases = data.dataset("test")
    position = {data.source.sample_ids[i]: n for n, i in enumerate(data.splits["test"])}
    manifest = []
    for n, pick in enumerate(picks, start=1):
        shown, overlay, caption = segmentation_pair(
            module, cases[position[pick["id"]]], pick["dice"]
        )
        names = (f"{n:02d}-input.png", f"{n:02d}-segmentation.png")
        _save(shown, out / names[0])
        _save(overlay, out / names[1])
        entry = {"id": pick["id"], "input": names[0], "overlay": names[1], "kind": "segmentation"}
        manifest.append(entry | caption)
    return manifest
