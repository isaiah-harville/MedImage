"""Small generated datasets with a learnable signal, for tests and smoke runs.

Every sample is generated from ``(seed, index)``, so the data is identical across runs and
machines. Several images share a patient, which exercises patient-level splitting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from medimage.data.base import DatasetInfo, DatasetSource, LabelMode, Sample
from medimage.registry import DATASETS


class _Synthetic(DatasetSource):
    def __init__(self, num_samples: int, images_per_patient: int, seed: int) -> None:
        self.num_samples = num_samples
        self.seed = seed
        self._ids = [f"synthetic-{i}" for i in range(num_samples)]
        self._patients = [f"patient-{i // images_per_patient}" for i in range(num_samples)]

    @property
    def sample_ids(self) -> list[str]:
        return self._ids

    @property
    def patient_ids(self) -> list[str]:
        return self._patients

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, index])

    def fingerprint(self) -> dict[str, Any]:
        return {
            "generator": type(self).__name__,
            "seed": self.seed,
            "num_samples": self.num_samples,
        }


@DATASETS.register("synthetic_classification")
class SyntheticClassification(_Synthetic):
    """Noise images where class ``k`` adds a bright cube in the ``k``-th region of the image."""

    def __init__(
        self,
        num_samples: int = 200,
        num_classes: int = 2,
        image_size: int = 32,
        spatial_dims: int = 2,
        images_per_patient: int = 2,
        label_mode: LabelMode = "binary",
        noise: float = 0.5,
        seed: int = 0,
    ) -> None:
        super().__init__(num_samples, images_per_patient, seed)
        if label_mode == "binary" and num_classes != 2:
            raise ValueError("binary label_mode needs num_classes: 2")
        self.num_classes, self.size, self.dims, self.noise = (
            num_classes,
            image_size,
            spatial_dims,
            noise,
        )
        # A patient's images share the label, as repeat studies of one patient often do.
        patient_rng = np.random.default_rng([seed, 1_000_003])
        n_patients = len(set(self._patients))
        patient_labels = patient_rng.integers(0, num_classes, size=n_patients)
        self.labels = [int(patient_labels[i // images_per_patient]) for i in range(num_samples)]
        self.info = DatasetInfo(
            task_type="classification",
            spatial_dims=spatial_dims,
            in_channels=1,
            class_names=tuple(f"class_{k}" for k in range(num_classes)),
            label_mode=label_mode,
            license="CC0-1.0 (generated)",
        )

    @property
    def strata(self) -> list[int]:
        return self.labels

    def __getitem__(self, index: int) -> Sample:
        rng = self._rng(index)
        shape = (self.size,) * self.dims
        image = rng.normal(0.0, self.noise, size=shape).astype(np.float32)
        label = self.labels[index]
        width = self.size // (self.num_classes + 1)
        start = width * label + width // 2
        region = tuple(slice(start, start + width) for _ in range(self.dims))
        image[region] += 1.0
        return {
            "image": torch.from_numpy(image[None]),
            "label": torch.tensor(label, dtype=torch.int64),
            "id": self._ids[index],
            "patient_id": self._patients[index],
        }


@DATASETS.register("synthetic_detection")
class SyntheticDetection(_Synthetic):
    """2D noise images with up to ``max_objects`` bright squares; class sets the brightness."""

    def __init__(
        self,
        num_samples: int = 64,
        num_classes: int = 1,
        image_size: int = 64,
        max_objects: int = 3,
        images_per_patient: int = 2,
        noise: float = 0.2,
        seed: int = 0,
    ) -> None:
        super().__init__(num_samples, images_per_patient, seed)
        self.num_classes, self.size, self.max_objects, self.noise = (
            num_classes,
            image_size,
            max_objects,
            noise,
        )
        self.info = DatasetInfo(
            task_type="detection",
            spatial_dims=2,
            in_channels=1,
            class_names=tuple(f"class_{k}" for k in range(num_classes)),
            license="CC0-1.0 (generated)",
        )

    def _objects(self, index: int) -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
        rng = self._rng(index)
        count = int(rng.integers(0, self.max_objects + 1))
        boxes, labels = [], []
        for _ in range(count):
            side = int(rng.integers(self.size // 8, self.size // 3))
            x, y = (int(v) for v in rng.integers(0, self.size - side, size=2))
            boxes.append([x, y, x + side, y + side])
            labels.append(int(rng.integers(0, self.num_classes)))
        return np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(labels, np.int64), rng

    @property
    def strata(self) -> list[int]:
        return [int(len(self._objects(i)[0]) > 0) for i in range(self.num_samples)]

    def __getitem__(self, index: int) -> Sample:
        boxes, labels, rng = self._objects(index)
        image = rng.normal(0.0, self.noise, size=(self.size, self.size)).astype(np.float32)
        for (x1, y1, x2, y2), label in zip(boxes.astype(int), labels, strict=True):
            image[y1:y2, x1:x2] += 0.5 + 0.5 * (label + 1) / self.num_classes
        return {
            "image": torch.from_numpy(image[None]),
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "id": self._ids[index],
            "patient_id": self._patients[index],
        }
