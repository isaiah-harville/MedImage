"""MedMNIST v2 through the official ``medmnist`` package.

MedMNIST publishes no patient IDs, so the official split is used as published and the run
records that leakage could not be checked. A subset records its source dataset's license
where that is stricter than what MedMNIST declares; see docs/datasets.md.

Citation: Yang, J. et al. "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D
biomedical image classification." Scientific Data 10, 41 (2023).
"""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from medmnist import INFO
from torchvision.datasets.utils import download_url

from opendiagnostics.data.base import SPLIT_NAMES, DatasetInfo, MedicalDataModule, Source

CITATION = (
    "Yang, J. et al. MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D "
    "biomedical image classification. Scientific Data 10, 41 (2023). "
    "https://doi.org/10.1038/s41597-022-01721-8"
)

# Subsets whose source is stricter than the CC BY 4.0 the medmnist package declares:
# RibFrac's Zenodo record (3893508) is CC BY-NC 4.0, and the LiTS paper (arXiv 1901.04056,
# section 3.2) licenses its data "as CC BY-NC-SA" without a version.
SOURCE_LICENSES = {"fracturemnist3d": "CC BY-NC 4.0", "organmnist3d": "CC BY-NC-SA"}


@dataclass(frozen=True)
class Variant:
    """A corrected release of a subset, in MedMNIST's own archive layout."""

    flag: str
    archives: dict[int, tuple[str, str]]
    license: str
    citation: str


_DERMAMNIST_C = "https://zenodo.org/api/records/12739457/files/dermamnist_corrected_{}.npz/content"
VARIANTS = {
    "dermamnist_corrected": Variant(
        flag="dermamnist",
        archives={
            28: (_DERMAMNIST_C.format(28), "96e2862980877d45d9cacb5c25de6950"),
            224: (_DERMAMNIST_C.format(224), "84920fb70c83b234c295b6f0d4ae2bc0"),
        },
        license="CC BY-NC 4.0",
        citation=(
            "Abhishek, K., Jain, A. and Hamarneh, G. Investigating the Quality of DermaMNIST and "
            "Fitzpatrick17k Dermatological Image Datasets. arXiv:2401.14497 (2024). "
            "https://doi.org/10.5281/zenodo.11101337"
        ),
    )
}


class MemmapImages:
    """Images in a ``.npy`` file, opened lazily so each worker process maps the file itself.

    2D images stay ``(H, W, C)`` arrays. A 3D volume, stored ``(D, H, W)`` in 0-255, becomes a
    ``(1, D, H, W)`` float tensor in ``[0, 1]``.
    """

    def __init__(self, images: Path, labels: np.ndarray, volumetric: bool = False) -> None:
        self.path = images
        self.labels = labels
        self.volumetric = volumetric
        self._images: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[np.ndarray | torch.Tensor, np.ndarray]:
        if self._images is None:
            self._images = np.load(self.path, mmap_mode="r")
        image = np.array(self._images[index])
        if self.volumetric:
            return torch.from_numpy(image[None]).float().div(255), self.labels[index]
        return image, self.labels[index]

    def __getstate__(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "labels": self.labels,
            "volumetric": self.volumetric,
            "_images": None,
        }


class ConcatSplits:
    def __init__(self, parts: list[MemmapImages]) -> None:
        self.parts = parts
        self.offsets = np.cumsum([0, *(len(p) for p in parts)])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int) -> tuple[np.ndarray | torch.Tensor, np.ndarray]:
        part = int(np.searchsorted(self.offsets, index, side="right")) - 1
        return self.parts[part][index - int(self.offsets[part])]


class MedMNISTDataModule(MedicalDataModule):
    """One MedMNIST subset (``flag``, e.g. ``pneumoniamnist``) at ``size`` 28, 64, 128, or 224.

    3D subsets have ``size`` 28 or 64. ``root`` defaults to the package's ``~/.medmnist``.
    ``variant`` names a corrected release in ``VARIANTS`` to read instead of the original, for
    example ``dermamnist_corrected`` (DermaMNIST-C, with no lesion in both train and test).
    """

    def __init__(
        self,
        flag: str,
        size: int = 28,
        root: str | None = None,
        download: bool = True,
        variant: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if flag not in INFO:
            raise ValueError(f"unknown MedMNIST subset {flag!r}; choose from {sorted(INFO)}")
        self.variant = VARIANTS.get(variant or "")
        if variant is not None and (
            self.variant is None or self.variant.flag != flag or size not in self.variant.archives
        ):
            known = {k: (v.flag, sorted(v.archives)) for k, v in VARIANTS.items()}
            raise ValueError(
                f"no variant {variant!r} of {flag} at size {size}; known variants and their "
                f"subsets and sizes: {known}"
            )
        self.variant_name = variant
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
            license=self.variant.license
            if self.variant
            else SOURCE_LICENSES.get(flag, meta["license"]),
            citation=f"{CITATION} {self.variant.citation}" if self.variant else CITATION,
        )

    @property
    def _suffix(self) -> str:
        return "" if self.size == 28 else f"_{self.size}"

    @property
    def _name(self) -> str:
        return f"{self.variant_name or self.flag}{self._suffix}"

    def _url_md5(self) -> tuple[str, str | None]:
        if self.variant:
            return self.variant.archives[self.size]
        meta: dict[str, Any] = INFO[self.flag]
        return meta[f"url{self._suffix}"], meta.get(f"MD5{self._suffix}")

    def _array(self, split: str, kind: str) -> Path:
        return self.root / self._name / f"{split}_{kind}.npy"

    def prepare_data(self) -> None:
        """Download the archive (MD5-checked) and unpack each array to its own ``.npy`` file.

        Unpacking streams from the zip, so no split is ever held in memory; PathMNIST at 224
        pixels is about 16 GB uncompressed.
        """
        if all(self._array(s, k).exists() for s in SPLIT_NAMES for k in ("images", "labels")):
            return
        archive = f"{self._name}.npz"
        if self.download:
            url, md5 = self._url_md5()
            download_url(url, str(self.root), archive, md5)
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
            parts.append(
                MemmapImages(self._array(split, "images"), labels, self.info.spatial_dims == 3)
            )
            ids += [f"{split}-{i}" for i in range(len(labels))]
            splits[split] = list(range(start, start + len(labels)))
            start += len(labels)
        return Source(
            dataset=ConcatSplits(parts),
            sample_ids=ids,
            patient_ids=ids,
            splits=splits,
            fingerprint={
                "subset": self._name.removesuffix(self._suffix),
                "size": self.size,
                "md5": self._url_md5()[1],
            },
        )
