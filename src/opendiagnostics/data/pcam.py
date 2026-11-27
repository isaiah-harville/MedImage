"""PatchCamelyon (PCam): 96x96 RGB patches from the Camelyon16 lymph node slides.

A patch is positive when its centre 32x32 region holds at least one tumour pixel. The images
and labels come as HDF5 files from the authors' Zenodo record, with the official train, valid
and test splits. The record's metadata CSVs name the Camelyon16 slide each patch came from,
and that slide is the patient ID, so the leakage check and the bootstrap work on slides.
The IDs are slides, not patients: the Camelyon16 paper reports 399 slides from 399 patients,
which implies one slide per patient, but it does not say so outright. The data is CC0 1.0.

Citation: Veeling, B. S. et al. "Rotation Equivariant CNNs for Digital Pathology." MICCAI
2018, arXiv:1806.03962.
"""

from __future__ import annotations

import csv
import gzip
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import h5py
import numpy as np
from torchvision.datasets.utils import check_md5

from opendiagnostics.data.base import SPLIT_NAMES, DatasetInfo, MedicalDataModule, Source
from opendiagnostics.data.medmnist import ConcatSplits, MemmapImages

log = logging.getLogger("opendiagnostics")

CITATION = (
    "Veeling, B. S., Linmans, J., Winkens, J., Cohen, T. & Welling, M. Rotation Equivariant "
    "CNNs for Digital Pathology. MICCAI 2018. https://arxiv.org/abs/1806.03962. Camelyon16: "
    "Ehteshami Bejnordi, B. et al. JAMA 318(22), 2199-2210 (2017). "
    "https://doi.org/10.1001/jama.2017.14585"
)
RECORD = "https://zenodo.org/records/2546921/files/"
USER_AGENT = "opendiagnostics"
CHUNK = 16 * 1024 * 1024
OFFICIAL = {"train": "train", "val": "valid", "test": "test"}
META_COLUMNS = ["", "coord_y", "coord_x", "tumor_patch", "center_tumor_patch", "wsi"]

# The checksums Zenodo lists for record 2546921. The table in the PCam README swaps the
# valid and test checksums.
MD5 = {
    "camelyonpatch_level_2_split_train_x.h5.gz": "1571f514728f59376b705fc836ff4b63",
    "camelyonpatch_level_2_split_train_y.h5.gz": "35c2d7259d906cfc8143347bb8e05be7",
    "camelyonpatch_level_2_split_train_meta.csv": "5a3dd671e465cfd74b5b822125e65b0a",
    "camelyonpatch_level_2_split_valid_x.h5.gz": "d5b63470df7cfa627aeec8b9dc0c066e",
    "camelyonpatch_level_2_split_valid_y.h5.gz": "2b85f58b927af9964a4c15b8f7e8f179",
    "camelyonpatch_level_2_split_valid_meta.csv": "67589e00a4a37ec317f2d1932c7502ca",
    "camelyonpatch_level_2_split_test_x.h5.gz": "d8c2d60d490dbd479f8199bdfa0cf6ec",
    "camelyonpatch_level_2_split_test_y.h5.gz": "60a7035772fbdb7f34eb86d4420cf66a",
    "camelyonpatch_level_2_split_test_meta.csv": "3455fd69135b66734e1008f3af684566",
}

Opener = Callable[..., Any]


def official_name(split: str, kind: str) -> str:
    """The record's file name for the images (``x``), labels (``y``) or ``meta`` of a split."""
    suffix = "csv" if kind == "meta" else "h5.gz"
    return f"camelyonpatch_level_2_split_{OFFICIAL[split]}_{kind}.{suffix}"


def download(url: str, dest: Path, md5: str, opener: Opener = urlopen) -> None:
    """Download ``url`` to ``dest``, resuming an earlier partial download, and check its MD5.

    The partial file sits next to ``dest`` and takes the final name only once it matches. A
    server that ignores the ``Range`` header sends the whole file, which replaces the partial.
    """
    partial = dest.with_name(dest.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        with opener(Request(url, headers=headers), timeout=60) as response:
            resumed = offset > 0 and response.status == 206
            served = response.headers.get("Content-Range", "")
            if resumed and not served.startswith(f"bytes {offset}-"):
                raise RuntimeError(
                    f"{url} answered a resume from byte {offset} with {served!r}. Delete "
                    f"{partial} and run again."
                )
            with partial.open("ab" if resumed else "wb") as out:
                shutil.copyfileobj(response, out, CHUNK)
    except HTTPError as error:
        if error.code != 416 or not offset:
            raise
    if not check_md5(partial, md5):
        partial.unlink()
        raise RuntimeError(
            f"{dest.name} downloaded from {url} does not match MD5 {md5}. The partial file "
            "was deleted, so running again downloads it from the start."
        )
    partial.replace(dest)


def unpack(archive: Path, key: str, target: Path, rows: int = 4096) -> None:
    """Decompress one ``.h5.gz`` and copy its array to the ``.npy`` file ``target``.

    The array is copied ``rows`` patches at a time, so a split is never held in memory.
    Labels, stored as ``(N, 1, 1, 1)``, are written as ``(N,)``.
    """
    scratch = target.with_name(target.name + ".h5")
    partial = target.with_name(target.name + ".partial")
    with gzip.open(archive) as src, scratch.open("wb") as dst:
        shutil.copyfileobj(src, dst, CHUNK)
    with h5py.File(scratch, "r") as handle:
        data = handle[key]
        if not isinstance(data, h5py.Dataset):
            raise ValueError(f"{archive} has no array {key!r}; delete it and run again")
        shape = data.shape if key == "x" else (len(data),)
        out = np.lib.format.open_memmap(partial, mode="w+", dtype=data.dtype, shape=shape)
        for start in range(0, len(data), rows):
            block = data[start : start + rows]
            out[start : start + len(block)] = block.reshape(len(block), *shape[1:])
        out.flush()
        del out
    partial.replace(target)
    scratch.unlink()


def read_slides(meta: Path, labels: np.ndarray) -> list[str]:
    """The source slide of each patch, after checking the CSV rows line up with the labels.

    Row ``i`` must have index ``i`` and a ``center_tumor_patch`` flag equal to label ``i``;
    a CSV in another order would attach patches to the wrong slides.
    """
    with meta.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if header != META_COLUMNS:
            raise ValueError(f"{meta} has columns {header}; expected {META_COLUMNS}")
        rows = list(reader)
    if len(rows) != len(labels):
        raise ValueError(
            f"{meta} has {len(rows)} rows for {len(labels)} patches. Delete {meta.parent} "
            "and run again to download the files afresh."
        )
    for i, (row, label) in enumerate(zip(rows, labels, strict=True)):
        if row[0] != str(i) or (row[4] == "True") != bool(label):
            raise ValueError(
                f"row {i} of {meta} does not match patch {i} of the HDF5 files (index "
                f"{row[0]}, center_tumor_patch {row[4]}, label {int(label)}), so the slide IDs "
                f"cannot be trusted. Delete {meta.parent} and run again to download the files "
                "afresh."
            )
    return [row[5] for row in rows]


class PCamDataModule(MedicalDataModule):
    """PCam with its official split, stored under ``root`` (default ``data/pcam``).

    The first run downloads 8.0 GB and unpacks the images to 9.1 GB of ``.npy`` files.
    """

    def __init__(self, root: str = "data/pcam", download: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.root = Path(root).expanduser()
        self.download = download
        self.info = DatasetInfo(
            label_mode="binary",
            class_names=("normal", "tumor"),
            in_channels=3,
            license="CC0 1.0",
            citation=CITATION,
        )

    def _array(self, split: str, kind: str) -> Path:
        return self.root / f"{split}_{kind}.npy"

    def _fetch(self, name: str) -> Path:
        path = self.root / name
        if not path.exists():
            if not self.download:
                raise FileNotFoundError(f"{path} is missing; set download: true")
            log.info("downloading %s", RECORD + name)
            download(RECORD + name, path, MD5[name])
        return path

    def prepare_data(self) -> None:
        """Download the record's files (resumable, MD5-checked) and unpack the arrays to ``.npy``.

        Running again skips every file already in place, so an interrupted run picks up
        where it stopped. The ``.h5.gz`` archives can be deleted once the ``.npy`` files
        exist; the metadata CSVs are read on every load.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        for split in SPLIT_NAMES:
            self._fetch(official_name(split, "meta"))
            for kind, key in (("images", "x"), ("labels", "y")):
                target = self._array(split, kind)
                if not target.exists():
                    archive = self._fetch(official_name(split, key))
                    log.info("unpacking %s", archive)
                    unpack(archive, key, target)

    def load(self) -> Source:
        self.prepare_data()
        parts, ids, slides, splits, start = [], [], [], {}, 0
        for split in SPLIT_NAMES:
            labels = np.load(self._array(split, "labels"))
            parts.append(MemmapImages(self._array(split, "images"), labels))
            slides += read_slides(self.root / official_name(split, "meta"), labels)
            ids += [f"{split}-{i}" for i in range(len(labels))]
            splits[split] = list(range(start, start + len(labels)))
            start += len(labels)
        return Source(
            dataset=ConcatSplits(parts),
            sample_ids=ids,
            patient_ids=slides,
            splits=splits,
            fingerprint={"record": RECORD, "md5": MD5},
        )
