"""Final evaluation of the selected checkpoint, with patient-level bootstrap intervals.

The binary decision threshold is fitted on validation predictions, then applied unchanged
to every split, so the test split never informs a choice.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from lightning.pytorch import Trainer

from opendiagnostics.data import MedicalDataModule
from opendiagnostics.modules import DiagnosticModule, Prediction


@dataclass
class EvalSettings:
    """``bootstrap_unit: patient`` resamples whole patients, which keeps one patient's
    correlated images together and gives honest intervals."""

    bootstrap_samples: int = 1000
    confidence: float = 0.95
    bootstrap_unit: Literal["patient", "sample"] = "patient"
    splits: list[str] = field(default_factory=lambda: ["val", "test"])


def _default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray | torch.Tensor):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=_default, allow_nan=True) + "\n")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)


def predict(
    trainer: Trainer,
    module: DiagnosticModule,
    data: MedicalDataModule,
    split: str,
    checkpoint: Path,
) -> list[Prediction]:
    batches = trainer.predict(module, dataloaders=data.loader(split), ckpt_path=str(checkpoint))
    return [p for batch in batches or [] for p in batch]  # type: ignore[union-attr]


def evaluate(
    trainer: Trainer,
    module: DiagnosticModule,
    data: MedicalDataModule,
    settings: EvalSettings,
    seed: int,
    checkpoint: Path,
    run_dir: Path,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score the checkpoint on each split. ``calibration`` replaces the one fitted on val."""
    val = predict(trainer, module, data, "val", checkpoint)
    calibration = calibration or module.calibrate(val)
    epoch = torch.load(checkpoint, map_location="cpu", weights_only=True)["epoch"]
    unit = settings.bootstrap_unit if data.info.patient_ids_available else "sample"
    bootstrap = {
        "replicates": settings.bootstrap_samples,
        "confidence": settings.confidence,
        "unit": unit,
        "seed": seed,
    }
    results: dict[str, Any] = {
        "checkpoint_epoch": epoch,
        "calibration": calibration,
        "bootstrap": bootstrap
        | {
            "method": "percentile",
            "unit": unit if data.info.patient_ids_available else "sample (no patient IDs)",
        },
    }
    for split in settings.splits:
        predictions = val if split == "val" else predict(trainer, module, data, split, checkpoint)
        results[split] = module.score(
            predictions, calibration, bootstrap if settings.bootstrap_samples else None
        )
        write_rows(run_dir / f"predictions_{split}.csv", module.prediction_rows(predictions))
    return results
