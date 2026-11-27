"""Load published models from the cache and run them on single images.

Each model is a folder ``<project>/<run-id>/`` that ``weights.sync`` downloaded: the run's
``config.yaml``, ``metrics.json``, ``artifacts.yaml`` (the checkpoint's URL and SHA-256), and
optionally ``examples/`` with its license-cleared example images. Its checkpoint is in the
cache, named by SHA-256.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from opendiagnostics.cli import load_run
from opendiagnostics.data.base import DatasetInfo, Transform, prepare
from opendiagnostics.examples import explain
from opendiagnostics.modules import ImageClassifier
from opendiagnostics.publish import EXAMPLES, EXAMPLES_MANIFEST
from opendiagnostics.serve.catalog import ORDER, describe
from opendiagnostics.serve.weights import cached, checkpoint_of

log = logging.getLogger("opendiagnostics.serve")

DISCLAIMER = (
    "Research model, not a medical device. It has not been cleared or approved for clinical "
    "use and must not be used to diagnose, treat, or make any decision about a patient. "
    "Results can be wrong. Its accuracy was measured on a public benchmark, not on images "
    "like yours."
)


@dataclass
class DemoImage:
    sample_id: str
    truth: str
    path: Path


@dataclass
class ServedModel:
    id: str
    module: ImageClassifier
    transform: Transform | None
    info: DatasetInfo
    size: int
    threshold: float
    description: dict[str, Any]
    demos: dict[str, DemoImage] = field(default_factory=dict)


def _input_size(config: dict[str, Any]) -> int:
    data = config["data"].get("init_args") or {}
    return int(data.get("size") or data.get("image_size") or 224)


def load_model(folder: Path, cache: Path) -> ServedModel:
    model_id = folder.parent.name
    artifact = checkpoint_of(folder)
    if artifact is None:
        raise ValueError(f"{model_id}: artifacts.yaml has no checkpoint URL")
    checkpoint = cached(cache, artifact)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{model_id}: {checkpoint} is missing; sync the cache first")
    config = yaml.safe_load((folder / "config.yaml").read_text())
    cli = load_run(folder, with_data=False)
    module = cli.model
    if not isinstance(module, ImageClassifier):
        raise ValueError(f"{model_id}: only classifiers can be served, got {type(module).__name__}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    module.load_state_dict(state["state_dict"])
    module.eval()
    metrics = json.loads((folder / "metrics.json").read_text())
    info = cli.datamodule.info
    flag = (config["data"].get("init_args") or {}).get("flag")
    scan = describe(flag, model_id)
    test_auroc = metrics.get("test", {}).get("auroc")
    demos = {}
    manifest = folder / EXAMPLES / EXAMPLES_MANIFEST
    if manifest.exists():
        for example in json.loads(manifest.read_text()):
            demos[example["id"]] = DemoImage(
                example["id"], example["truth"], folder / EXAMPLES / example["input"]
            )
    return ServedModel(
        id=model_id,
        module=module,
        transform=cli.datamodule.eval_transforms,
        info=info,
        size=_input_size(config),
        threshold=float(metrics.get("calibration", {}).get("threshold", 0.5)),
        description={
            "id": model_id,
            "scan_type": scan.scan_type,
            "modality": scan.modality,
            "question": scan.question,
            "label_mode": info.label_mode,
            "classes": list(info.class_names),
            "input": {"channels": info.in_channels, "size": _input_size(config)},
            "dataset": f"{flag} (MedMNIST v2)" if flag else type(cli.datamodule).__name__,
            "license": info.license,
            "citation": info.citation,
            "run_id": folder.name,
            "test_auroc": (
                {k: test_auroc[k] for k in ("value", "lower", "upper")}
                if isinstance(test_auroc, dict)
                else None
            ),
            "demo_count": len(demos),
            "weights": {"url": artifact.url, "bytes": artifact.bytes, "sha256": artifact.sha256},
        },
        demos=demos,
    )


def _rank(model: ServedModel) -> tuple[int, str]:
    flag = model.id.split("-")[0]
    return (ORDER.index(flag) if flag in ORDER else len(ORDER), model.id)


def load_models(folders: list[Path], cache: Path) -> dict[str, ServedModel]:
    """Load each model folder. A model that fails to load is logged and left out."""
    models = []
    for folder in folders:
        try:
            models.append(load_model(folder, cache))
        except Exception:
            log.exception("could not load %s/%s", folder.parent.name, folder.name)
    return {m.id: m for m in sorted(models, key=_rank)}


def _png(image: torch.Tensor) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image.permute(1, 2, 0).numpy()).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def predict(
    model: ServedModel, image: Image.Image, truth: str | None, steps: int
) -> dict[str, Any]:
    """Run the model on one image and explain its top prediction."""
    started = time.perf_counter()
    mode = model.info.label_mode
    names = list(model.info.class_names)
    pixels = np.array(
        image.convert("L" if model.info.in_channels == 1 else "RGB").resize(
            (model.size, model.size), Image.Resampling.BILINEAR
        )
    )
    raw, _ = prepare(pixels, 0, None, mode)
    model_input, _ = prepare(pixels, 0, model.transform, mode)
    with torch.no_grad():
        probs = model.module.probabilities(model_input[None])[0].tolist()
    if mode == "binary":
        positive = probs[0]
        scores = [1 - positive, positive]
        predicted = [names[1] if positive >= model.threshold else names[0]]
        threshold, label = model.threshold, 0
    elif mode == "multiclass":
        scores = probs
        predicted = [names[int(np.argmax(probs))]]
        threshold, label = None, int(np.argmax(probs))
    else:
        scores = probs
        predicted = [n for n, p in zip(names, probs, strict=True) if p >= 0.5]
        threshold, label = 0.5, int(np.argmax(probs))
    shown, overlay = explain(model.module, raw, model_input, label, steps)
    ranked = sorted(zip(names, scores, strict=True), key=lambda pair: -pair[1])
    return {
        "model": model.id,
        "label_mode": mode,
        "probabilities": [{"class": n, "probability": float(p)} for n, p in ranked],
        "predicted": predicted,
        "threshold": threshold,
        "input_png": _png(shown),
        "attribution_png": _png(overlay),
        "truth": truth,
        "disclaimer": DISCLAIMER,
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }
