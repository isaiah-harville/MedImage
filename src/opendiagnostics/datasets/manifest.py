"""Generic datasets described by a manifest file.

These cover most public datasets once their labels are converted into one CSV (for
classification) or one COCO JSON (for detection) that also records the patient ID.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from opendiagnostics.data.base import DatasetInfo, DatasetSource, LabelMode, Sample
from opendiagnostics.io import load_image
from opendiagnostics.registry import DATASETS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@DATASETS.register("csv_classification")
class CSVClassification(DatasetSource):
    """Images listed in a CSV with one row per image.

    Required columns: ``path_column`` (relative to ``root``) and ``patient_column``. Labels
    come from ``label_columns``:

    * ``binary``: one column of 0/1.
    * ``multiclass``: one column of class names (listed in ``class_names``) or indices.
    * ``multilabel``: one 0/1 column per finding. Blank cells are treated as
      ``missing_label`` (default 0). Uncertain CheXpert-style ``-1`` labels must be mapped
      before training; ``uncertain_label`` does that here.

    An optional ``split_column`` holding ``train``/``val``/``test`` supplies a predefined split.
    """

    def __init__(
        self,
        manifest: str,
        root: str | None = None,
        label_mode: LabelMode = "binary",
        label_columns: list[str] | None = None,
        class_names: list[str] | None = None,
        path_column: str = "path",
        patient_column: str = "patient_id",
        split_column: str | None = None,
        spatial_dims: int = 2,
        in_channels: int = 1,
        missing_label: float = 0.0,
        uncertain_label: float | None = None,
        license: str = "unknown",
        citation: str = "",
        loader: dict[str, Any] | None = None,
    ) -> None:
        if label_mode not in ("binary", "multiclass", "multilabel"):
            raise ValueError(
                f"label_mode must be binary, multiclass, or multilabel, got {label_mode!r}"
            )
        self.manifest = Path(manifest)
        self.root = Path(root) if root else self.manifest.parent
        self.loader_kwargs = loader or {}
        with self.manifest.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"{manifest} has no rows")
        columns = label_columns or ["label"]
        missing = {path_column, patient_column, *columns} - set(rows[0])
        if missing:
            raise ValueError(f"{manifest} lacks columns {sorted(missing)}")

        self.paths = [r[path_column] for r in rows]
        self._patients = [r[patient_column] for r in rows]
        if len(set(self.paths)) != len(self.paths):
            raise ValueError(f"{manifest} lists the same path more than once")
        self._ids = list(self.paths)

        if label_mode == "multilabel":
            names = tuple(class_names or columns)

            def parse(cell: str) -> float:
                if cell.strip() == "":
                    return missing_label
                value = float(cell)
                if value == -1:
                    if uncertain_label is None:
                        raise ValueError("found -1 (uncertain) labels; set uncertain_label")
                    return uncertain_label
                return value

            self.labels: list[list[float]] = [[parse(r[c]) for c in columns] for r in rows]
        else:
            if len(columns) != 1:
                raise ValueError(f"{label_mode} needs exactly one label column")
            raw = [r[columns[0]] for r in rows]
            if label_mode == "binary":
                names = tuple(class_names or ("negative", "positive"))
                indices = [int(float(v)) for v in raw]
            else:
                names = tuple(class_names or sorted(set(raw)))
                lookup = {n: i for i, n in enumerate(names)}
                indices = [lookup[v] if v in lookup else int(v) for v in raw]
            bad = {v for v in indices if not 0 <= v < len(names)}
            if bad:
                raise ValueError(f"labels {sorted(bad)} are outside 0..{len(names) - 1}")
            self.labels = [[float(v)] for v in indices]

        self._splits = None
        if split_column:
            self._splits = {
                s: [i for i, r in enumerate(rows) if r[split_column] == s]
                for s in ("train", "val", "test")
            }

        self.info = DatasetInfo(
            task_type="classification",
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            class_names=names,
            label_mode=label_mode,
            license=license,
            citation=citation,
        )

    @property
    def sample_ids(self) -> list[str]:
        return self._ids

    @property
    def patient_ids(self) -> list[str]:
        return self._patients

    @property
    def strata(self) -> list[Any]:
        if self.info.label_mode == "multilabel":
            return [tuple(int(v > 0.5) for v in row) for row in self.labels]
        return [int(row[0]) for row in self.labels]

    def predefined_splits(self) -> dict[str, list[int]] | None:
        return self._splits

    def fingerprint(self) -> dict[str, Any]:
        return {"manifest": str(self.manifest), "manifest_sha256": _sha256(self.manifest)}

    def __getitem__(self, index: int) -> Sample:
        image = load_image(self.root / self.paths[index], **self.loader_kwargs)
        row = self.labels[index]
        if self.info.label_mode == "multilabel":
            label = torch.tensor(row, dtype=torch.float32)
        else:
            label = torch.tensor(int(row[0]), dtype=torch.int64)
        return {
            "image": torch.from_numpy(image.array),
            "label": label,
            "id": self._ids[index],
            "patient_id": self._patients[index],
        }


@DATASETS.register("coco_detection")
class COCODetection(DatasetSource):
    """2D detection from a COCO-format JSON.

    Each entry in ``images`` needs ``file_name`` and a patient ID under ``patient_key``
    (default ``patient_id``). Without one, set ``allow_missing_patient: true`` and each image
    becomes its own patient; only do that when the dataset has one image per patient.
    Category IDs are mapped to contiguous foreground indices in ascending ID order.
    """

    def __init__(
        self,
        annotations: str,
        root: str | None = None,
        patient_key: str = "patient_id",
        allow_missing_patient: bool = False,
        split_key: str | None = None,
        in_channels: int = 1,
        license: str = "unknown",
        citation: str = "",
        loader: dict[str, Any] | None = None,
    ) -> None:
        self.annotations = Path(annotations)
        self.root = Path(root) if root else self.annotations.parent
        self.loader_kwargs = loader or {}
        coco = json.loads(self.annotations.read_text())
        categories = sorted(coco["categories"], key=lambda c: c["id"])
        self.category_index = {c["id"]: i for i, c in enumerate(categories)}
        images = sorted(coco["images"], key=lambda im: im["id"])
        boxes: dict[int, list[list[float]]] = {im["id"]: [] for im in images}
        labels: dict[int, list[int]] = {im["id"]: [] for im in images}
        for ann in coco.get("annotations", []):
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes[ann["image_id"]].append([x, y, x + w, y + h])
            labels[ann["image_id"]].append(self.category_index[ann["category_id"]])

        self.files = [im["file_name"] for im in images]
        self._ids = [str(im["id"]) for im in images]
        self.boxes = [boxes[im["id"]] for im in images]
        self.labels = [labels[im["id"]] for im in images]
        patients = []
        for im in images:
            if patient_key in im:
                patients.append(str(im[patient_key]))
            elif allow_missing_patient:
                patients.append(str(im["id"]))
            else:
                raise ValueError(
                    f"image {im['id']} has no {patient_key!r}; see allow_missing_patient"
                )
        self._patients = patients
        self._splits = None
        if split_key:
            self._splits = {
                s: [i for i, im in enumerate(images) if im.get(split_key) == s]
                for s in ("train", "val", "test")
            }
        self.info = DatasetInfo(
            task_type="detection",
            spatial_dims=2,
            in_channels=in_channels,
            class_names=tuple(c["name"] for c in categories),
            patient_ids_available=not allow_missing_patient,
            license=license,
            citation=citation,
        )

    @property
    def sample_ids(self) -> list[str]:
        return self._ids

    @property
    def patient_ids(self) -> list[str]:
        return self._patients

    @property
    def strata(self) -> list[Any]:
        # Stratify on whether an image has any finding, so negatives spread evenly.
        return [int(bool(b)) for b in self.boxes]

    def predefined_splits(self) -> dict[str, list[int]] | None:
        return self._splits

    def fingerprint(self) -> dict[str, Any]:
        return {
            "annotations": str(self.annotations),
            "annotations_sha256": _sha256(self.annotations),
        }

    def __getitem__(self, index: int) -> Sample:
        image = load_image(self.root / self.files[index], **self.loader_kwargs)
        return {
            "image": torch.from_numpy(image.array),
            "boxes": torch.tensor(self.boxes[index], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(self.labels[index], dtype=torch.int64),
            "id": self._ids[index],
            "patient_id": self._patients[index],
        }
