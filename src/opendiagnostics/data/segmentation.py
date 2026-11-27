"""Segmentation data: volumes with voxel label maps, transformed with MONAI dictionary transforms.

A segmentation :class:`~opendiagnostics.data.base.Source` holds ``(image, label)`` pairs.
The image is a path, a list of paths (one per channel, all on the same voxel grid), or a
channel-first array; the label is a path or a ``(1, *spatial)`` array of class indices.
Files are read with MONAI's ``LoadImaged``.

The configured ``train_transforms`` and ``eval_transforms`` are MONAI dictionary transforms
on the keys ``image`` and ``label``, such as ``Orientationd``, ``Spacingd``,
``ScaleIntensityRanged`` and ``RandCropByPosNegLabeld``. A crop that returns several
patches per volume adds them all to the batch. Evaluation serves one whole volume per batch.
Leave the label out of ``Spacingd`` in ``eval_transforms`` to score in the label's own voxel
grid: the module resamples its prediction onto that grid first, the way challenges score.

Batches add ``image_affine`` and ``target_affine`` (``(B, 4, 4)``, from the MetaTensors)
and ``spacing`` (the label's voxel size per case) to the usual keys.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from monai.data import Dataset, MetaTensor, PersistentDataset
from monai.data.utils import json_hashing
from monai.transforms import Compose, LoadImaged
from monai.utils import convert_to_tensor
from torch.utils.data import DataLoader

from opendiagnostics.data.base import SPLIT_NAMES, DatasetInfo, MedicalDataModule, Source

KEYS = ("image", "label")


def _affine(tensor: torch.Tensor, spatial_dims: int) -> torch.Tensor:
    if isinstance(tensor, MetaTensor):
        return torch.as_tensor(tensor.affine, dtype=torch.float64)
    return torch.eye(spatial_dims + 1, dtype=torch.float64)


def _spacing(tensor: torch.Tensor, spatial_dims: int) -> tuple[float, ...]:
    if isinstance(tensor, MetaTensor):
        return tuple(float(v) for v in tensor.pixdim[:spatial_dims])
    return (1.0,) * spatial_dims


def _plain(tensor: Any) -> torch.Tensor:
    return convert_to_tensor(tensor, track_meta=False)


def collate_cases(samples: Sequence[dict[str, Any] | list[dict[str, Any]]]) -> dict[str, Any]:
    """Stack transformed cases, flattening the patch lists that multi-sample crops return."""
    flat = [p for s in samples for p in (s if isinstance(s, list) else [s])]
    dims = flat[0]["image"].ndim - 1
    return {
        "image": torch.stack([_plain(s["image"]).float() for s in flat]),
        "target": torch.stack([_plain(s["label"]).to(torch.int16) for s in flat]),
        "image_affine": torch.stack([_affine(s["image"], dims) for s in flat]),
        "target_affine": torch.stack([_affine(s["label"], dims) for s in flat]),
        "spacing": [_spacing(s["label"], dims) for s in flat],
        "id": [s["id"] for s in flat],
        "patient_id": [s["patient_id"] for s in flat],
    }


class SegmentationDataModule(MedicalDataModule):
    """Base for segmentation data modules. Subclasses set ``info`` and implement ``load``.

    ``cache_dir`` stores each case after the transforms that come before the first random
    one (loading, reorienting, resampling, intensity scaling) with MONAI's
    ``PersistentDataset``, so later epochs read preprocessed tensors. The cache is keyed on
    the case and the transforms, so a changed transform does not reuse stale entries.
    ``eval_batch_size`` is always 1, because whole volumes differ in shape.
    """

    def __init__(self, cache_dir: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cache_dir = cache_dir

    def dataset(self, split: str, train: bool = False) -> Dataset:
        """The split's cases, loaded and transformed, as MONAI dictionaries."""
        source = self.source
        records = []
        for i in self.splits[split]:
            image, label = source.dataset[i]
            records.append(
                {
                    "image": image,
                    "label": label,
                    "id": source.sample_ids[i],
                    "patient_id": source.patient_ids[i],
                }
            )
        steps: list[Any] = []
        if records and isinstance(records[0]["label"], str | Path):
            steps.append(LoadImaged(KEYS, image_only=True, ensure_channel_first=True))
        configured = self.train_transforms if train else self.eval_transforms
        if configured is not None:
            steps.append(configured)
        transform = Compose(steps)
        if self.cache_dir is None:
            return Dataset(records, transform)
        return PersistentDataset(
            records,
            transform,
            cache_dir=self.cache_dir,
            hash_transform=json_hashing,
            track_meta=True,
            weights_only=False,
        )

    def loader(self, split: str, train: bool = False) -> DataLoader:
        return DataLoader(
            self.dataset(split, train),
            batch_size=self.batch_size if train else 1,
            shuffle=train,
            num_workers=self.num_workers,
            collate_fn=collate_cases,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )


class NiftiSegmentationDataModule(SegmentationDataModule):
    """Cases listed in a CSV manifest, one row per labelled volume, with a unique ``id``.

    ``image_columns`` name one or more image paths per row, stacked as channels; a single
    4D NIfTI file also gives several channels, so set ``in_channels`` for it. Paths are
    relative to the manifest. ``class_names`` lists every label value in order, starting
    with the background at 0. ``split_column`` holds ``train``, ``val`` or ``test``; rows
    with any other value are left out. Set ``patient_ids_available`` false when the patient
    column holds case IDs because the dataset publishes no patient IDs.
    """

    def __init__(
        self,
        manifest: str,
        class_names: list[str],
        image_columns: list[str] | None = None,
        label_column: str = "label",
        id_column: str = "id",
        patient_column: str = "patient_id",
        split_column: str | None = "split",
        in_channels: int | None = None,
        spatial_dims: int = 3,
        patient_ids_available: bool = True,
        license: str = "unknown",
        citation: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.manifest = Path(manifest)
        self.image_columns = image_columns or ["image"]
        if not self.manifest.is_file():
            raise FileNotFoundError(
                f"{manifest} does not exist; run the dataset's script in scripts/datasets/ "
                "to download the data and write it"
            )
        with self.manifest.open(newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"{manifest} has no rows")
        wanted = {*self.image_columns, label_column, id_column, patient_column}
        if split_column:
            wanted.add(split_column)
        missing = wanted - set(self.rows[0])
        if missing:
            raise ValueError(f"{manifest} lacks columns {sorted(missing)}")
        if len(class_names) < 2:
            raise ValueError("class_names needs the background and at least one class")
        self.columns = (label_column, id_column, patient_column, split_column)
        self.info = DatasetInfo(
            label_mode="segmentation",
            class_names=tuple(class_names),
            in_channels=in_channels or len(self.image_columns),
            spatial_dims=spatial_dims,
            patient_ids_available=patient_ids_available,
            license=license,
            citation=citation,
        )

    def load(self) -> Source:
        label, sample, patient, split = self.columns
        base = self.manifest.parent

        def image(row: dict[str, str]) -> str | list[str]:
            paths = [str(base / row[c]) for c in self.image_columns]
            return paths[0] if len(paths) == 1 else paths

        splits = None
        if split:
            splits = {n: [i for i, r in enumerate(self.rows) if r[split] == n] for n in SPLIT_NAMES}
        return Source(
            dataset=[(image(r), str(base / r[label])) for r in self.rows],
            sample_ids=[r[sample] for r in self.rows],
            patient_ids=[r[patient] for r in self.rows],
            splits=splits,
            fingerprint={"manifest": str(self.manifest), "rows": len(self.rows)},
        )
