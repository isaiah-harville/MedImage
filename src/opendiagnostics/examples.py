"""Example figures for the lab notebook: test images beside what the model made of them.

For a classifier, the second image is an Integrated Gradients attribution map (Captum) for
the predicted class, laid over the input. For a detector, it is the input with ground-truth
boxes in green and predicted boxes, with scores, in red. Segmentation runs are rendered by
:mod:`opendiagnostics.segmentation_examples`. ``examples/examples.json`` lists
each pair with its truth and prediction. ``opendiagnostics publish`` copies the figures only
when the dataset's license allows redistributing its images.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from captum.attr import IntegratedGradients
from PIL import Image
from torch.nn import functional as F
from torchvision.utils import draw_bounding_boxes

from opendiagnostics.data import MedicalDataModule
from opendiagnostics.data.base import prepare
from opendiagnostics.data.segmentation import SegmentationDataModule
from opendiagnostics.modules import (
    DiagnosticModule,
    ImageClassifier,
    ObjectDetector,
    SegmentationModule,
    SlideClassifier,
    VolumeDetector,
)
from opendiagnostics.publish import EXAMPLES, EXAMPLES_MANIFEST

DISPLAY_SIZE = 224
GROUND_TRUTH = (46, 204, 113)
PREDICTED = (231, 76, 60)


def _class_columns(rows: Sequence[dict[str, str]], prefix: str) -> list[str]:
    return [c for c in rows[0] if c.startswith(prefix)]


def choose_classification(
    rows: Sequence[dict[str, str]], class_names: Sequence[str], threshold: float, limit: int
) -> list[dict[str, Any]]:
    """Pick test samples to show, from ``predictions_test.csv`` rows.

    For each class, the most confident correct prediction and the most confident wrong one,
    taken in turn across classes until ``limit``. Multilabel runs show, for each label, the
    sample the model is most confident has it.
    """
    if not rows:
        return []
    probs = _class_columns(rows, "prob/")
    names = [c.removeprefix("prob/") for c in probs]
    if "target" not in rows[0]:
        picks = []
        for column, name in zip(probs, names, strict=True):
            positives = [r for r in rows if float(r[f"target/{name}"]) == 1]
            if positives:
                best = max(positives, key=lambda r: (float(r[column]), r["id"]))
                picks.append(
                    {
                        "id": best["id"],
                        "truth": ", ".join(n for n in names if float(best[f"target/{n}"]) == 1),
                        "prediction": ", ".join(
                            n for c, n in zip(probs, names, strict=True) if float(best[c]) >= 0.5
                        )
                        or "none",
                        "confidence": float(best[column]),
                        "correct": None,
                        "label": names.index(name),
                    }
                )
        return picks[:limit]
    binary = len(probs) == 1
    candidates: dict[tuple[int, bool], list[dict[str, Any]]] = {}
    for row in rows:
        truth = int(row["target"])
        if binary:
            p = float(row[probs[0]])
            predicted = int(p >= threshold)
            confidence = p if predicted else 1 - p
        else:
            values = [float(row[c]) for c in probs]
            predicted = int(np.argmax(values))
            confidence = values[predicted]
        correct = predicted == truth
        candidates.setdefault((truth, correct), []).append(
            {
                "id": row["id"],
                "truth": class_names[truth],
                "prediction": class_names[predicted],
                "confidence": confidence,
                "correct": correct,
                "label": 0 if binary else predicted,
            }
        )
    for group in candidates.values():
        group.sort(key=lambda c: (-c["confidence"], c["id"]))
    keys = sorted(candidates, key=lambda key: (not key[1], key[0]))
    picks: list[dict[str, Any]] = []
    depth = 0
    while len(picks) < limit and any(depth < len(g) for g in candidates.values()):
        for key in keys:
            if depth < len(candidates[key]) and len(picks) < limit:
                picks.append(candidates[key][depth])
        depth += 1
    return picks


def _to_rgb(image: torch.Tensor) -> torch.Tensor:
    """A ``(C, H, W)`` or ``(C, D, H, W)`` image in ``[0, 1]`` as ``uint8 (3, H, W)``.

    3D volumes show their middle slice along the first spatial axis.
    """
    if image.ndim == 4:
        image = image[:, image.shape[1] // 2]
    image = image[:3] if image.shape[0] >= 3 else image[:1].expand(3, -1, -1)
    return (image.clamp(0, 1) * 255).round().to(torch.uint8)


def _scale(image: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Upscale small images with nearest neighbour so they are legible on the page."""
    factor = max(1, math.ceil(DISPLAY_SIZE / min(image.shape[-2:])))
    if factor == 1:
        return image, 1.0
    return F.interpolate(image[None].float(), scale_factor=factor, mode="nearest")[0].to(
        torch.uint8
    ), float(factor)


def _save(image: torch.Tensor, path: Path) -> None:
    Image.fromarray(image.permute(1, 2, 0).cpu().numpy()).save(path, optimize=True)


def _heat(rgb: torch.Tensor, attribution: torch.Tensor) -> torch.Tensor:
    """Blend a ``[0, 1]`` heat map over an image, from transparent to yellow-red."""
    heat = attribution[None]
    color = torch.stack([torch.ones_like(heat[0]), heat[0], torch.zeros_like(heat[0])]) * 255
    alpha = 0.65 * heat
    return (rgb.float() * (1 - alpha) + color * alpha).round().clamp(0, 255).to(torch.uint8)


def saliency(
    module: ImageClassifier, image: torch.Tensor, label: int, size: tuple[int, ...], steps: int
) -> torch.Tensor:
    """Integrated Gradients for output ``label``, summed over channels, scaled to ``[0, 1]``
    by its 99th percentile, and resized to the spatial ``size``: ``(H, W)`` or ``(D, H, W)``."""
    inputs = image[None].to(module.device).requires_grad_()
    maps = IntegratedGradients(module).attribute(
        inputs, target=label, n_steps=steps, internal_batch_size=8
    )
    heat = maps[0].detach().abs().sum(0).float()
    top = torch.quantile(heat.flatten(), 0.99).clamp_min(1e-12)
    heat = (heat / top).clamp(0, 1)
    mode = "trilinear" if heat.ndim == 3 else "bilinear"
    return F.interpolate(heat[None, None], size=size, mode=mode)[0, 0].cpu()


def explain(
    module: ImageClassifier,
    raw: torch.Tensor,
    model_input: torch.Tensor,
    label: int,
    steps: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The input as ``uint8 (3, H, W)`` and the same image with an Integrated Gradients map
    for output ``label`` blended over it. ``raw`` is the image in ``[0, 1]`` before the
    model's transforms; ``model_input`` is what the network sees. A volume shows its middle
    slice along the first spatial axis."""
    rgb, _ = _scale(_to_rgb(raw))
    if raw.ndim == 4:
        heat = saliency(module, model_input, label, tuple(raw.shape[1:]), steps)
        heat = heat[heat.shape[0] // 2][None, None]
        heat = F.interpolate(heat, size=tuple(rgb.shape[-2:]), mode="bilinear")[0, 0]
    else:
        heat = saliency(module, model_input, label, tuple(rgb.shape[-2:]), steps)
    return rgb, _heat(rgb, heat)


def explain_volume(
    module: ImageClassifier,
    raw: torch.Tensor,
    model_input: torch.Tensor,
    label: int,
    steps: int = 32,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Every slice of a ``(C, D, H, W)`` volume along its first spatial axis, as ``uint8
    (3, H, W)``, without and with the Integrated Gradients map blended over it."""
    heat = saliency(module, model_input, label, tuple(raw.shape[1:]), steps)
    slices = [_to_rgb(raw[:, depth]) for depth in range(raw.shape[1])]
    return slices, [_heat(rgb, heat[depth]) for depth, rgb in enumerate(slices)]


@torch.no_grad()
def _detector_pair(
    module: ObjectDetector,
    raw: torch.Tensor,
    target: dict[str, torch.Tensor],
    model_input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    rgb, factor = _scale(_to_rgb(raw))
    output = module.network([model_input.to(module.device)])[0]
    keep = output["scores"] >= 0.5
    if not keep.any():
        keep = torch.arange(len(output["scores"])) < 3
    sx = raw.shape[-1] / model_input.shape[-1] * factor
    sy = raw.shape[-2] / model_input.shape[-2] * factor
    scale = torch.tensor([sx, sy, sx, sy])
    predicted = output["boxes"][keep].float().cpu() * scale
    scores = output["scores"][keep].float().cpu()
    names = module.info.class_names
    labels = (output["labels"][keep].cpu() - 1).tolist()
    drawn = draw_bounding_boxes(rgb, target["boxes"].float() * factor, colors=GROUND_TRUTH, width=2)
    drawn = draw_bounding_boxes(
        drawn,
        predicted,
        labels=[f"{names[k]} {s:.2f}" for k, s in zip(labels, scores.tolist(), strict=True)],
        colors=PREDICTED,
        width=2,
    )
    caption = {
        "truth": f"{len(target['boxes'])} box(es)",
        "prediction": f"{int(keep.sum())} box(es)",
        "confidence": float(scores.max()) if len(scores) else 0.0,
        "correct": None,
    }
    return rgb, drawn, caption


def write_examples(
    module: DiagnosticModule,
    data: MedicalDataModule,
    run_dir: Path,
    checkpoint: Path,
    device: torch.device,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Render example pairs for the run's test split into ``run_dir/examples``.

    Slide classifiers see feature bags, not images, and 3D detection has no figures yet, so
    their manifest is empty.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    module.load_state_dict(state)
    module.to(device).eval()
    source = data.source
    position = {sample: i for i, sample in enumerate(source.sample_ids)}
    out = run_dir / EXAMPLES
    out.mkdir(exist_ok=True)
    for old in [*out.glob("*.png"), *out.glob("*.npy")]:
        old.unlink()
    if isinstance(module, SegmentationModule):
        from opendiagnostics.segmentation_examples import write_segmentation_examples

        assert isinstance(data, SegmentationDataModule)
        manifest = write_segmentation_examples(module, data, run_dir, out, limit)
        (out / EXAMPLES_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
        module.cpu()
        return manifest
    mode = data.info.label_mode
    picks: list[dict[str, Any]] = []
    if isinstance(module, ObjectDetector):
        ids = [source.sample_ids[i] for i in data.splits["test"]]
        picks = [{"id": sample} for sample in ids[:limit]]
    elif not isinstance(module, SlideClassifier | VolumeDetector):
        metrics = json.loads((run_dir / "metrics.json").read_text())
        threshold = metrics.get("calibration", {}).get("threshold", 0.5)
        with (run_dir / "predictions_test.csv").open(newline="") as handle:
            picks = choose_classification(
                list(csv.DictReader(handle)), data.info.class_names, threshold, limit
            )
    manifest = []
    for n, pick in enumerate(picks, start=1):
        image, target = source.dataset[position[pick["id"]]]
        raw, raw_target = prepare(image, target, None, mode)
        model_input, _ = prepare(image, target, data.eval_transforms, mode)
        if isinstance(module, ObjectDetector):
            shown, overlay, caption = _detector_pair(module, raw, raw_target, model_input)
            kind = "boxes"
        else:
            assert isinstance(module, ImageClassifier)
            shown, overlay = explain(module, raw, model_input, pick["label"])
            caption = {k: pick[k] for k in ("truth", "prediction", "confidence", "correct")}
            kind = "attribution"
        names = (f"{n:02d}-input.png", f"{n:02d}-{kind}.png")
        _save(shown, out / names[0])
        _save(overlay, out / names[1])
        entry = {"id": pick["id"], "input": names[0], "overlay": names[1], "kind": kind}
        if raw.ndim == 4:
            entry["volume"] = f"{n:02d}-volume.npy"
            np.save(out / entry["volume"], (raw.clamp(0, 1) * 255).round().to(torch.uint8).numpy())
        manifest.append(entry | caption)
    (out / EXAMPLES_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    module.cpu()
    return manifest
