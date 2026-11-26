"""MedMNIST v2 through the official ``medmnist`` package.

MedMNIST publishes no patient IDs, so the official split is used as published and the run
records that leakage could not be checked. Each subset keeps its source dataset's license;
see docs/datasets.md.

Citation: Yang, J. et al. "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D
biomedical image classification." Scientific Data 10, 41 (2023).
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from medmnist import INFO
from torchvision.datasets.utils import download_url

from opendiagnostics.data.base import SPLIT_NAMES, DatasetInfo, MedicalDataModule, Source

CITATION = (
    "Yang, J. et al. MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D "
    "biomedical image classification. Scientific Data 10, 41 (2023). "
    "https://doi.org/10.1038/s41597-022-01721-8"
)


class MemmapImages:
    """Images in a ``.npy`` file, opened lazily so each worker process maps the file itself."""

    def __init__(self, images: Path, labels: np.ndarray) -> None:
        self.path = images
        self.labels = labels
        self._images: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if self._images is None:
            self._images = np.load(self.path, mmap_mode="r")
        return np.array(self._images[index]), self.labels[index]

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self.path, "labels": self.labels, "_images": None}


class ConcatSplits:
    def __init__(self, parts: list[MemmapImages]) -> None:
        self.parts = parts
        self.offsets = np.cumsum([0, *(len(p) for p in parts)])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        part = int(np.searchsorted(self.offsets, index, side="right")) - 1
        return self.parts[part][index - int(self.offsets[part])]


class MedMNISTDataModule(MedicalDataModule):
    """One MedMNIST subset (``flag``, e.g. ``pneumoniamnist``) at ``size`` 28, 64, 128, or 224.

    3D subsets have ``size`` 28 or 64. ``root`` defaults to the package's ``~/.medmnist``.
    """

    def __init__(
        self,
        flag: str,
        size: int = 28,
        root: str | None = None,
        download: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if flag not in INFO:
            raise ValueError(f"unknown MedMNIST subset {flag!r}; choose from {sorted(INFO)}")
        self.flag = flag
        self.size = size
        self.root = Path(root).expanduser() if root else Path.home() / ".medmnist"
        self.download = download
        meta: dict[str, Any] = INFO[flag]
        labels = meta["label"]
        task = meta["task"]
        mode = (
            "multilabel"
            if task.startswith("multi-label")
            else "binary"
            if len(labels) == 2
            else "multiclass"
        )
        self.info = DatasetInfo(
            label_mode=mode,
            class_names=tuple(labels[str(i)] for i in range(len(labels))),
            in_channels=meta["n_channels"],
            spatial_dims=3 if flag.endswith("3d") else 2,
            patient_ids_available=False,
            license=meta["license"],
            citation=CITATION,
        )

    @property
    def _suffix(self) -> str:
        return "" if self.size == 28 else f"_{self.size}"

    def _md5(self) -> str | None:
        meta: dict[str, Any] = INFO[self.flag]
        return meta.get(f"MD5{self._suffix}")

    def _array(self, split: str, kind: str) -> Path:
        return self.root / f"{self.flag}{self._suffix}" / f"{split}_{kind}.npy"

    def prepare_data(self) -> None:
        """Download the archive (MD5-checked) and unpack each array to its own ``.npy`` file.

        Unpacking streams from the zip, so no split is ever held in memory; PathMNIST at 224
        pixels is about 16 GB uncompressed.
        """
        if all(self._array(s, k).exists() for s in SPLIT_NAMES for k in ("images", "labels")):
            return
        meta: dict[str, Any] = INFO[self.flag]
        archive = f"{self.flag}{self._suffix}.npz"
        if self.download:
            download_url(meta[f"url{self._suffix}"], str(self.root), archive, self._md5())
        elif not (self.root / archive).exists():
            raise FileNotFoundError(f"{self.root / archive} is missing; set download: true")
        self._array("train", "images").parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.root / archive) as bundle:
            for split in SPLIT_NAMES:
                for kind in ("images", "labels"):
                    target = self._array(split, kind)
                    partial = target.with_suffix(".partial")
                    with bundle.open(f"{split}_{kind}.npy") as src, partial.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                    partial.replace(target)

    def load(self) -> Source:
        self.prepare_data()
        parts, ids, splits, start = [], [], {}, 0
        for split in SPLIT_NAMES:
            labels = np.load(self._array(split, "labels"))
            parts.append(MemmapImages(self._array(split, "images"), labels))
            ids += [f"{split}-{i}" for i in range(len(labels))]
            splits[split] = list(range(start, start + len(labels)))
            start += len(labels)
        return Source(
            dataset=ConcatSplits(parts),
            sample_ids=ids,
            patient_ids=ids,
            splits=splits,
            fingerprint={
                "subset": self.flag,
                "size": self.size,
                "md5": INFO[self.flag].get("MD5" if self.size == 28 else f"MD5_{self.size}"),
            },
        )
