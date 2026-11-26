"""MedMNIST v2 (2D and 3D) read straight from the official Zenodo ``.npz`` files.

The ``medmnist`` pip package is not needed. Files are downloaded once, checked against the
MD5 published by the MedMNIST authors, and read with ``allow_pickle=False``.

MedMNIST does not publish patient IDs, so patient-level leakage cannot be checked; the
official train/val/test split is used as published. Results on MedMNIST are a benchmark of
the pipeline, not a clinical claim.

``DatasetInfo.license`` is the license MedMNIST declares for the subset. Some source datasets
are stricter (RetinaMNIST, the Organ subsets, FractureMNIST3D); see docs/datasets.md.

Citation: Yang, J. et al. "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D
biomedical image classification." Scientific Data 10, 41 (2023).
https://doi.org/10.1038/s41597-022-01721-8
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch

from medimage.data.base import DatasetInfo, DatasetSource, Sample
from medimage.datasets.medmnist_info import SUBSETS, ZENODO_RECORD
from medimage.registry import DATASETS

CITATION = (
    "Yang, J., Shi, R., Wei, D. et al. MedMNIST v2 - A large-scale lightweight benchmark for "
    "2D and 3D biomedical image classification. Sci Data 10, 41 (2023). "
    "https://doi.org/10.1038/s41597-022-01721-8"
)


def default_root() -> Path:
    return Path(os.environ.get("MEDIMAGE_DATA", Path.home() / ".cache" / "medimage")) / "medmnist"


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(subset: str, size: int, root: Path, download: bool) -> tuple[Path, str]:
    meta = SUBSETS[subset]
    md5 = meta["md5"].get(str(size))
    if md5 is None:
        raise ValueError(f"{subset} has no size {size}; available: {sorted(meta['md5'], key=int)}")
    filename = f"{subset}.npz" if size == 28 else f"{subset}_{size}.npz"
    path = root / filename
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} not found; set download: true to fetch it")
        root.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        url = f"{ZENODO_RECORD}/{filename}?download=1"
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        if _md5(partial) != md5:
            partial.unlink()
            raise OSError(f"MD5 mismatch for {url}; the download was discarded")
        partial.replace(path)
    elif _md5(path) != md5:
        raise OSError(f"{path} does not match the published MD5 {md5}; delete it and retry")
    return path, md5


@DATASETS.register("medmnist")
class MedMNIST(DatasetSource):
    """One MedMNIST subset, e.g. ``pneumoniamnist`` or ``nodulemnist3d``.

    Images are scaled to [0, 1]. Ordinal subsets (RetinaMNIST) are treated as multiclass.
    """

    def __init__(
        self,
        subset: str,
        size: int = 28,
        root: str | None = None,
        download: bool = False,
    ) -> None:
        if subset not in SUBSETS:
            raise ValueError(f"unknown MedMNIST subset {subset!r}; choose from {sorted(SUBSETS)}")
        meta = SUBSETS[subset]
        self.subset, self.size = subset, size
        self.path, self.md5 = fetch(subset, size, Path(root) if root else default_root(), download)

        with np.load(self.path, allow_pickle=False) as archive:
            parts = {
                s: (archive[f"{s}_images"], archive[f"{s}_labels"])
                for s in ("train", "val", "test")
            }
        self.images = np.concatenate([parts[s][0] for s in parts])
        self.labels = np.concatenate([parts[s][1] for s in parts])
        offsets = np.cumsum([0] + [len(parts[s][0]) for s in parts])
        self._splits = {s: list(range(offsets[i], offsets[i + 1])) for i, s in enumerate(parts)}
        self._ids = [f"{subset}-{s}-{i}" for s in parts for i in range(len(parts[s][0]))]

        is_3d = subset.endswith("3d")
        task = meta["task"]
        mode = (
            "multilabel"
            if task.startswith("multi-label")
            else "binary"
            if task == "binary-class"
            else "multiclass"
        )
        names = tuple(meta["labels"][str(i)] for i in range(len(meta["labels"])))
        self.info = DatasetInfo(
            task_type="classification",
            spatial_dims=3 if is_3d else 2,
            in_channels=int(meta["n_channels"]),
            class_names=names,
            label_mode=mode,
            patient_ids_available=False,
            license=meta["license"],
            citation=CITATION,
            extra={"subset": subset, "size": size, "source_task": task},
        )

    @property
    def sample_ids(self) -> list[str]:
        return self._ids

    @property
    def strata(self) -> list[Any]:
        if self.info.label_mode == "multilabel":
            return [tuple(int(v) for v in row) for row in self.labels]
        return [int(row[0]) for row in self.labels]

    def predefined_splits(self) -> dict[str, list[int]]:
        return self._splits

    def fingerprint(self) -> dict[str, Any]:
        return {"file": self.path.name, "md5": self.md5, "zenodo": ZENODO_RECORD}

    def __getitem__(self, index: int) -> Sample:
        array = self.images[index].astype(np.float32) / 255.0
        if self.info.spatial_dims == 2:
            image = array[None] if array.ndim == 2 else np.moveaxis(array, -1, 0)
        else:
            image = array[None]
        row = self.labels[index]
        if self.info.label_mode == "multilabel":
            label = torch.as_tensor(row, dtype=torch.float32)
        else:
            label = torch.as_tensor(int(row[0]), dtype=torch.int64)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "label": label,
            "id": self._ids[index],
            "patient_id": self._ids[index],
        }
