"""Images listed in a CSV manifest, read with MONAI's ``LoadImage`` (DICOM, NIfTI, PNG, ...)."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from functools import cache, partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.data import PILReader
from monai.transforms import LoadImage
from PIL import Image

from opendiagnostics.data.base import SPLIT_NAMES, DatasetInfo, MedicalDataModule, Source

_reader = LoadImage(image_only=True, ensure_channel_first=True)


@cache
def _reader_for(mode: str | None) -> LoadImage:
    if mode is None:
        return _reader
    converter = partial(Image.Image.convert, mode=mode)
    return LoadImage(
        reader=PILReader(converter=converter), image_only=True, ensure_channel_first=True
    )


def read_image(path: str | Path, mode: str | None = None) -> np.ndarray:
    """A float32 image min-max scaled to ``[0, 1]``, shaped ``(C, H, W)`` for 2D files, so
    pixel ``x`` is the column and ``y`` the row. ``mode`` reads the file with Pillow and
    converts it to that mode, such as ``L``."""
    image = np.asarray(_reader_for(mode)(str(path)), dtype=np.float32)
    span = image.max() - image.min()
    image = (image - image.min()) / span if span else image * 0
    if image.ndim == 3:
        image = np.ascontiguousarray(np.swapaxes(image, -1, -2))
    return image


class LoadedImages:
    def __init__(
        self, paths: list[str], targets: Sequence[Any] | np.ndarray, mode: str | None = None
    ) -> None:
        self.paths = paths
        self.targets = targets
        self.mode = mode

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        # A tensor, not an array: torchvision reads a numpy array as height x width x channel.
        image = read_image(self.paths[index], self.mode)
        return torch.from_numpy(image), self.targets[index]


def _tokens(label: str, separator: str) -> list[str]:
    return [token for token in (t.strip() for t in label.split(separator)) if token]


class CSVImageDataModule(MedicalDataModule):
    """One row per image: a path (relative to the manifest), a patient ID, and a label.

    ``split_column`` names a column holding ``train``/``val``/``test`` for an official split.
    Images are min-max scaled to ``[0, 1]`` per image.

    With ``label_separator``, a label holds any number of ``class_names`` joined by the
    separator, such as ``Effusion|Mass``, and the task is multilabel; an empty label means
    none of them. ``image_mode`` reads every image with Pillow and converts it to that mode
    (``L`` or ``RGB``), for PNG or JPEG datasets that mix grayscale, RGB and RGBA files.
    """

    def __init__(
        self,
        manifest: str,
        image_column: str = "path",
        label_column: str = "label",
        patient_column: str = "patient_id",
        split_column: str | None = None,
        class_names: list[str] | None = None,
        label_separator: str | None = None,
        image_mode: str | None = None,
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
        self.label_separator = label_separator
        self.image_mode = image_mode
        if label_separator is not None:
            if not class_names:
                raise ValueError("label_separator needs class_names, listed in output order")
            found = {t for r in self.rows for t in _tokens(r[label_column], label_separator)}
            unknown = sorted(found - set(class_names))
            if unknown:
                raise ValueError(
                    f"{manifest} has labels that are not in class_names: {unknown[:10]}. Add "
                    "them to class_names, or leave a row's label empty when it has none."
                )
        names = class_names or sorted({r[label_column] for r in self.rows})
        self.class_names = list(names)
        mode = "binary" if len(names) == 2 else "multiclass"
        self.info = DatasetInfo(
            label_mode="multilabel" if label_separator is not None else mode,
            class_names=tuple(names),
            in_channels=in_channels,
            license=license,
            citation=citation,
        )

    def _targets(self) -> tuple[Sequence[Any] | np.ndarray, list[Any] | None]:
        """Per-row targets, and the strata to split on (none for multilabel)."""
        label = self.columns[1]
        if self.label_separator is None:
            labels = [self.class_names.index(r[label]) for r in self.rows]
            return labels, list(labels)
        index = {name: i for i, name in enumerate(self.class_names)}
        hot = np.zeros((len(self.rows), len(self.class_names)), dtype=np.float32)
        for row, r in enumerate(self.rows):
            for token in _tokens(r[label], self.label_separator):
                hot[row, index[token]] = 1
        return hot, None

    def load(self) -> Source:
        image, _, patient, split = self.columns
        base = self.manifest.parent
        labels, strata = self._targets()
        splits = None
        if split:
            splits = {n: [i for i, r in enumerate(self.rows) if r[split] == n] for n in SPLIT_NAMES}
        return Source(
            dataset=LoadedImages(
                [str(base / r[image]) for r in self.rows], labels, self.image_mode
            ),
            sample_ids=[r[image] for r in self.rows],
            patient_ids=[r[patient] for r in self.rows],
            strata=strata,
            splits=splits,
            fingerprint={"manifest": str(self.manifest), "rows": len(self.rows)},
        )
