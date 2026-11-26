"""MedMNIST v2 through the official ``medmnist`` package.

MedMNIST publishes no patient IDs, so the official split is used as published and the run
records that leakage could not be checked. Each subset keeps its source dataset's license;
see docs/datasets.md.

Citation: Yang, J. et al. "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D
biomedical image classification." Scientific Data 10, 41 (2023).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import medmnist
import numpy as np
from medmnist import INFO

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

    def _md5(self) -> str | None:
        meta: dict[str, Any] = INFO[self.flag]
        return meta.get("MD5" if self.size == 28 else f"MD5_{self.size}")

    def _npy(self, split: str) -> Path:
        return self.root / f"{self.flag}_{self.size}_{split}_images.npy"

    def prepare_data(self) -> None:
        meta: dict[str, Any] = INFO[self.flag]
        cls = getattr(medmnist, meta["python_class"])
        self.root.mkdir(parents=True, exist_ok=True)
        for split in SPLIT_NAMES:
            if self._npy(split).exists():
                continue
            data = cls(split=split, size=self.size, root=str(self.root), download=self.download)
            tmp = self._npy(split).with_suffix(".tmp.npy")
            np.save(tmp, data.imgs)
            np.save(self.root / f"{self.flag}_{self.size}_{split}_labels.npy", data.labels)
            tmp.replace(self._npy(split))

    def load(self) -> Source:
        self.prepare_data()
        parts, ids, splits, start = [], [], {}, 0
        for split in SPLIT_NAMES:
            labels = np.load(self.root / f"{self.flag}_{self.size}_{split}_labels.npy")
            parts.append(MemmapImages(self._npy(split), labels))
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
