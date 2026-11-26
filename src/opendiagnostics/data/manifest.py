"""Images listed in a CSV manifest, read with MONAI's ``LoadImage`` (DICOM, NIfTI, PNG, ...)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from monai.transforms import LoadImage

from opendiagnostics.data.base import SPLIT_NAMES, DatasetInfo, MedicalDataModule, Source


class LoadedImages:
    def __init__(self, paths: list[str], labels: list[int]) -> None:
        self.paths = paths
        self.labels = labels
        self.reader = LoadImage(image_only=True, ensure_channel_first=True)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        image = np.asarray(self.reader(self.paths[index]), dtype=np.float32)
        span = image.max() - image.min()
        image = (image - image.min()) / span if span else image * 0
        if image.ndim == 3:
            image = np.swapaxes(image, -1, -2)
        return image, self.labels[index]


class CSVImageDataModule(MedicalDataModule):
    """One row per image: a path (relative to the manifest), a patient ID, and a label.

    ``split_column`` names a column holding ``train``/``val``/``test`` for an official split.
    Images are min-max scaled to ``[0, 1]`` per image.
    """

    def __init__(
        self,
        manifest: str,
        image_column: str = "path",
        label_column: str = "label",
        patient_column: str = "patient_id",
        split_column: str | None = None,
        class_names: list[str] | None = None,
        in_channels: int = 1,
        license: str = "unknown",
        citation: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.manifest = Path(manifest)
        with self.manifest.open(newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"{manifest} has no rows")
        missing = {image_column, label_column, patient_column} - set(self.rows[0])
        if missing:
            raise ValueError(f"{manifest} lacks columns {sorted(missing)}")
        self.columns = (image_column, label_column, patient_column, split_column)
        names = class_names or sorted({r[label_column] for r in self.rows})
        self.class_names = list(names)
        self.info = DatasetInfo(
            label_mode="binary" if len(names) == 2 else "multiclass",
            class_names=tuple(names),
            in_channels=in_channels,
            license=license,
            citation=citation,
        )

    def load(self) -> Source:
        image, label, patient, split = self.columns
        base = self.manifest.parent
        labels = [self.class_names.index(r[label]) for r in self.rows]
        splits = None
        if split:
            splits = {n: [i for i, r in enumerate(self.rows) if r[split] == n] for n in SPLIT_NAMES}
        return Source(
            dataset=LoadedImages([str(base / r[image]) for r in self.rows], labels),
            sample_ids=[r[image] for r in self.rows],
            patient_ids=[r[patient] for r in self.rows],
            strata=list(labels),
            splits=splits,
            fingerprint={"manifest": str(self.manifest), "rows": len(self.rows)},
        )
