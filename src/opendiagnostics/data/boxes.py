"""Images with labelled boxes, listed in a CSV manifest and read with MONAI's ``LoadImage``."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from opendiagnostics.data.base import SPLIT_NAMES, DatasetInfo, MedicalDataModule, Source
from opendiagnostics.data.manifest import LoadedImages

COLUMNS = ("path", "patient_id", "label", "x1", "y1", "x2", "y2")
CORNERS = COLUMNS[3:]


@dataclass
class Annotated:
    patient_id: str
    split: str | None
    boxes: list[list[float]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


def read_box_manifest(path: Path, split_column: str | None) -> dict[str, Annotated]:
    """Group a manifest's rows by image; an image whose one row has no label has no boxes."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no rows")
    missing = {*COLUMNS, *([split_column] if split_column else [])} - set(rows[0])
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    images: dict[str, Annotated] = {}
    for line, row in enumerate(rows, start=2):
        split = row[split_column] if split_column else None
        if split is not None and split not in SPLIT_NAMES:
            raise ValueError(f"{path}:{line}: split must be one of {SPLIT_NAMES}, got {split!r}")
        image = images.setdefault(row["path"], Annotated(row["patient_id"], split))
        if (image.patient_id, image.split) != (row["patient_id"], split):
            raise ValueError(
                f"{path}:{line}: {row['path']} has more than one patient ID or split; "
                "every row of one image must agree"
            )
        if not row["label"]:
            if any(row[c] for c in CORNERS):
                raise ValueError(f"{path}:{line}: a box needs a label")
            continue
        try:
            box = [float(row[c]) for c in CORNERS]
        except ValueError:
            raise ValueError(f"{path}:{line}: x1, y1, x2, y2 must be numbers") from None
        if not (box[0] < box[2] and box[1] < box[3]):
            raise ValueError(f"{path}:{line}: a box needs x1 < x2 and y1 < y2, got {box}")
        image.boxes.append(box)
        image.labels.append(row["label"])
    return images


class BoxManifestDataModule(MedicalDataModule):
    """2D detection from a CSV with one row per box.

    Columns are ``path`` (relative to the manifest), ``patient_id``, ``label``, and the box
    corners ``x1, y1, x2, y2`` in pixels, ``x`` along columns. An image with no boxes is a
    negative: one row with ``label`` and the corners empty. ``split_column`` names a column
    holding ``train``/``val``/``test`` for an official split; leave ``val`` empty and use
    ``split: kfold`` to hold out validation patients from ``train``. Images are min-max
    scaled to ``[0, 1]`` per image.
    """

    def __init__(
        self,
        manifest: str,
        class_names: list[str] | None = None,
        split_column: str | None = None,
        in_channels: int = 1,
        license: str = "unknown",
        citation: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.manifest = Path(manifest)
        self.split_column = split_column
        self.images = read_box_manifest(self.manifest, split_column)
        found = sorted({label for image in self.images.values() for label in image.labels})
        names = class_names or found
        unknown = set(found) - set(names)
        if unknown:
            raise ValueError(f"{manifest} has labels {sorted(unknown)} not in class_names {names}")
        self.class_names = list(names)
        self.info = DatasetInfo(
            label_mode="detection",
            class_names=tuple(names),
            in_channels=in_channels,
            license=license,
            citation=citation,
        )

    def load(self) -> Source:
        paths = list(self.images)
        targets = [
            {
                "boxes": np.asarray(image.boxes, np.float32).reshape(-1, 4),
                "labels": np.asarray([self.class_names.index(n) for n in image.labels], np.int64),
            }
            for image in self.images.values()
        ]
        splits = None
        if self.split_column:
            splits = {
                name: [i for i, p in enumerate(paths) if self.images[p].split == name]
                for name in SPLIT_NAMES
            }
        return Source(
            dataset=LoadedImages([str(self.manifest.parent / p) for p in paths], targets),
            sample_ids=paths,
            patient_ids=[image.patient_id for image in self.images.values()],
            strata=["+".join(sorted(set(image.labels))) for image in self.images.values()],
            splits=splits,
            fingerprint={
                "manifest": str(self.manifest),
                "sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                "images": len(paths),
                "boxes": sum(len(image.boxes) for image in self.images.values()),
            },
        )
