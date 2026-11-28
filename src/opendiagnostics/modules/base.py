"""The Lightning module contract shared by classification and detection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import numpy as np
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import WeightAveraging
from torch import nn

from opendiagnostics.data import DatasetInfo
from opendiagnostics.metrics.bootstrap import bootstrap as run_bootstrap

Prediction = dict[str, Any]


class DiagnosticModule(LightningModule):
    """Wraps a network. Subclasses define the loss, predictions, and how to score them.

    Validation collects every prediction and scores the whole split at epoch end, because
    AUROC and mAP are not averages of per-batch values. Scores are logged as ``val/<name>``.
    ``info`` is linked from the data module by the CLI.
    """

    default_monitor: ClassVar[str]

    def __init__(self, network: nn.Module, info: DatasetInfo | None = None) -> None:
        super().__init__()
        if info is None:
            raise ValueError(
                f"{type(self).__name__} needs info from a data module; build it through "
                "`opendiagnostics fit`, which links data.info into the model"
            )
        self.network = network
        self.info: DatasetInfo = info
        self._val_predictions: list[Prediction] = []

    def predict_batch(self, batch: dict) -> list[Prediction]:
        raise NotImplementedError

    def calibrate(self, predictions: list[Prediction]) -> dict[str, Any]:
        """Settings fitted on validation predictions and reused unchanged on test."""
        return {}

    def score(
        self,
        predictions: list[Prediction],
        calibration: dict[str, Any] | None = None,
        bootstrap: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def prediction_rows(self, predictions: list[Prediction]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._val_predictions.extend(self.predict_batch(batch))

    def on_validation_epoch_end(self) -> None:
        metrics = self.score(self._val_predictions)
        self._val_predictions = []
        self.log_dict({f"val/{k}": float(v) for k, v in metrics.items() if isinstance(v, float)})

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Lightning skips callback save hooks for weights-only checkpoints, so WeightAveraging
        # would leave the raw weights in the checkpoint instead of the averaged ones that
        # validation scored and selection chose.
        if "current_model_state" in checkpoint:
            return
        for callback in self.trainer.callbacks:  # ty: ignore[missing-attribute]
            if isinstance(callback, WeightAveraging):
                callback.on_save_checkpoint(self.trainer, self, checkpoint)
                checkpoint.pop("current_model_state", None)
                checkpoint.pop("averaging_state", None)

    def predict_step(self, batch: dict, batch_idx: int) -> list[Prediction]:
        return self.predict_batch(batch)

    @staticmethod
    def bootstrapped(
        statistic: Callable[[np.ndarray], dict[str, float]],
        n: int,
        patients: Sequence[str],
        bootstrap: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not bootstrap:
            return statistic(np.arange(n))
        estimates = run_bootstrap(
            statistic,
            n,
            groups=patients if bootstrap["unit"] == "patient" else None,
            replicates=bootstrap["replicates"],
            confidence=bootstrap["confidence"],
            seed=bootstrap["seed"],
        )
        return {k: v.to_dict() for k, v in estimates.items()}
