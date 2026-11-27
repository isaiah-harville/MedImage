"""Load published models from a weights directory and run them on single images.

The weights directory holds one folder per model, made by ``scripts/bundle_models.py``:
``config.yaml`` (the run's resolved config), ``checkpoint.ckpt``, ``metrics.json``,
``artifacts.yaml`` (where the weights can be downloaded), and optionally ``examples/`` with
the run's license-cleared example images.
"""

from __future__ import annotations

import base64
import io
import json
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
from opendiagnostics.publish import EXAMPLES, EXAMPLES_MANIFEST, load_artifacts
from opendiagnostics.serve.catalog import ORDER, describe

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


def _weights(folder: Path) -> dict[str, Any] | None:
    for artifact in load_artifacts(folder / "artifacts.yaml").files:
        if artifact.file == "checkpoint.ckpt" and artifact.url:
            return {"url": artifact.url, "bytes": artifact.bytes, "sha256": artifact.sha256}
    return None


def load_model(folder: Path) -> ServedModel:
    config = yaml.safe_load((folder / "config.yaml").read_text())
    cli = load_run(folder, with_data=False)
    module = cli.model
    if not isinstance(module, ImageClassifier):
        raise ValueError(
            f"{folder.name}: only classifiers can be served, got {type(module).__name__}"
        )
    state = torch.load(folder / "checkpoint.ckpt", map_location="cpu", weights_only=True)
    module.load_state_dict(state["state_dict"])
    module.eval()
    metrics = json.loads((folder / "metrics.json").read_text())
    info = cli.datamodule.info
    flag = (config["data"].get("init_args") or {}).get("flag")
    scan = describe(flag, folder.name)
    test_auroc = metrics.get("test", {}).get("auroc")
    demos = {}
    manifest = folder / EXAMPLES / EXAMPLES_MANIFEST
    if manifest.exists():
        for example in json.loads(manifest.read_text()):
            demos[example["id"]] = DemoImage(
                example["id"], example["truth"], folder / EXAMPLES / example["input"]
            )
    return ServedModel(
        id=folder.name,
        module=module,
        transform=cli.datamodule.eval_transforms,
        info=info,
        size=_input_size(config),
        threshold=float(metrics.get("calibration", {}).get("threshold", 0.5)),
        description={
            "id": folder.name,
            "scan_type": scan.scan_type,
            "modality": scan.modality,
            "question": scan.question,
            "label_mode": info.label_mode,
            "classes": list(info.class_names),
            "input": {"channels": info.in_channels, "size": _input_size(config)},
            "dataset": f"{flag} (MedMNIST v2)" if flag else type(cli.datamodule).__name__,
            "license": info.license,
            "citation": info.citation,
            "run_id": (folder / "RUN_ID").read_text().strip()
            if (folder / "RUN_ID").exists()
            else None,
            "test_auroc": (
                {k: test_auroc[k] for k in ("value", "lower", "upper")}
                if isinstance(test_auroc, dict)
                else None
            ),
            "demo_count": len(demos),
            "weights": _weights(folder),
        },
        demos=demos,
    )


def load_models(weights_dir: Path) -> dict[str, ServedModel]:
    folders = [p for p in sorted(weights_dir.iterdir()) if (p / "checkpoint.ckpt").is_file()]
    if not folders:
        raise FileNotFoundError(f"{weights_dir} holds no model folders with a checkpoint.ckpt")
    models = [load_model(folder) for folder in folders]

    def rank(model: ServedModel) -> tuple[int, str]:
        flag = model.id.split("-")[0]
        return (ORDER.index(flag) if flag in ORDER else len(ORDER), model.id)

    return {m.id: m for m in sorted(models, key=rank)}


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
