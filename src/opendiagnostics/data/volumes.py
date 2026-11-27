"""3D volumes with lesion boxes, for :class:`~opendiagnostics.modules.VolumeDetector`.

Boxes are ``(x1, y1, z1, x2, y2, z2)`` in **world LPS millimetres**: the patient coordinate
system of DICOM and ITK, which is also the frame of LUNA16's ``annotations.csv``. World
boxes do not depend on how a volume is stored, oriented, or resampled, so one manifest
serves any spacing. Each volume is reoriented to RAS, optionally resampled, and scaled to
``[0, 1]``; the boxes are then mapped into its voxel grid with MONAI's
``AffineBoxToImageCoordinated``.

A world box's faces are the outer faces of the voxels it covers. An affine maps voxel
index ``i`` to the voxel's center, but MONAI's boxes put voxel ``i`` between ``i`` and
``i + 1``, so voxel boxes are shifted by half a voxel on the way in and back on the way
out (:func:`voxel_boxes_to_world`).

Training serves ``crops_per_volume`` patches per volume from MONAI's
``RandCropBoxByPosNegLabeld``, with boxes in patch voxels. Evaluation serves whole volumes
with the voxel-to-RAS affine, and the ground truth stays in world coordinates, so
predictions are scored in millimetres.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from monai.apps.detection.transforms.box_ops import apply_affine_to_boxes
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated,
    ClipBoxToImaged,
    RandCropBoxByPosNegLabeld,
)
from monai.data import DataLoader
from monai.data.utils import orientation_ras_lps
from monai.transforms import (
    Compose,
    Lambdad,
    LoadImage,
    Orientationd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)
from monai.utils import convert_to_tensor
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import Dataset

from opendiagnostics.data.base import DatasetInfo, MedicalDataModule, Source, Transform

Box = tuple[float, float, float, float, float, float]
RAS_LABELS = (("L", "R"), ("P", "A"), ("I", "S"))
HALF_VOXEL = 0.5


def to_voxel_corners(boxes: torch.Tensor) -> torch.Tensor:
    return boxes + HALF_VOXEL


def voxel_boxes_to_world(boxes: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    """Voxel boxes on an image with voxel-to-RAS ``affine``, as world LPS millimetres."""
    return apply_affine_to_boxes(boxes.double() - HALF_VOXEL, orientation_ras_lps(affine.double()))


def _check_boxes(boxes: Sequence[Box], what: str) -> None:
    for box in boxes:
        if not all(box[k] < box[k + 3] for k in range(3)):
            raise ValueError(f"{what} box {list(box)} needs x1 < x2, y1 < y2 and z1 < z2")


class VolumeEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    patient_id: str
    images: list[str] = Field(min_length=1)
    fold: int | None = None
    split: Literal["train", "val", "test"] | None = None
    boxes: list[Box] = []
    labels: list[int] = []
    ignore: list[Box] = []

    @model_validator(mode="after")
    def _consistent(self) -> VolumeEntry:
        if len(self.labels) != len(self.boxes):
            raise ValueError(f"{self.id}: {len(self.boxes)} boxes but {len(self.labels)} labels")
        _check_boxes(self.boxes, self.id)
        _check_boxes(self.ignore, f"{self.id} ignore")
        return self


class VolumeManifest(BaseModel):
    """The JSON file a ``scripts/datasets/*.py`` download script writes. Image paths are
    relative to the manifest; several paths per volume are stacked as channels."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    license: str
    citation: str = ""
    class_names: list[str] = Field(min_length=1)
    coordinates: Literal["world-lps-mm"]
    patient_ids_available: bool = True
    volumes: list[VolumeEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> VolumeManifest:
        channels = {len(v.images) for v in self.volumes}
        if len(channels) != 1:
            raise ValueError(f"volumes have different channel counts: {sorted(channels)}")
        for v in self.volumes:
            if any(not 0 <= label < len(self.class_names) for label in v.labels):
                raise ValueError(f"{v.id}: a label is outside 0..{len(self.class_names) - 1}")
        return self


def target(boxes: Any, labels: Any, ignore: Any = ()) -> dict[str, np.ndarray]:
    return {
        "boxes": np.asarray(boxes, np.float32).reshape(-1, 6),
        "labels": np.asarray(labels, np.int64).reshape(-1),
        "ignore": np.asarray(ignore, np.float32).reshape(-1, 6),
    }


class VolumeFiles:
    """Volumes read lazily with MONAI's ``LoadImage`` (NIfTI, a DICOM series folder, ...)."""

    def __init__(self, images: list[list[str]], targets: list[dict[str, np.ndarray]]) -> None:
        self.images = images
        self.targets = targets
        self.reader = LoadImage(image_only=True, ensure_channel_first=True)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, np.ndarray]]:
        paths = self.images[index]
        return self.reader(paths[0] if len(paths) == 1 else paths), self.targets[index]


class VolumeSplit(Dataset):
    """One split. Each item is a list of samples: the training crops of one volume, or the
    whole volume for evaluation."""

    def __init__(self, source: Source, indices: list[int], transform: Transform, train: bool):
        self.source = source
        self.indices = indices
        self.transform = transform
        self.train = train

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> list[dict[str, Any]]:
        i = self.indices[index]
        image, raw = self.source.dataset[i]
        data = {
            "image": image,
            "boxes": torch.as_tensor(raw["boxes"], dtype=torch.float32),
            "labels": torch.as_tensor(raw["labels"], dtype=torch.int64),
        }
        out = self.transform(data)
        crops = out if isinstance(out, list) else [out]
        names = {"id": self.source.sample_ids[i], "patient_id": self.source.patient_ids[i]}
        if self.train:
            return [
                {
                    "image": convert_to_tensor(c["image"], track_meta=False).float(),
                    "target": {
                        "boxes": torch.as_tensor(c["boxes"]).float().reshape(-1, 6),
                        "labels": torch.as_tensor(c["labels"]).long().reshape(-1),
                    },
                }
                | names
                for c in crops
            ]
        volume = crops[0]["image"]
        return [
            {
                "image": convert_to_tensor(volume, track_meta=False).float(),
                "affine": torch.as_tensor(volume.affine, dtype=torch.float64),
                "target": {
                    "world_boxes": torch.as_tensor(raw["boxes"], dtype=torch.float64),
                    "labels": torch.as_tensor(raw["labels"], dtype=torch.int64),
                    "world_ignore": torch.as_tensor(raw["ignore"], dtype=torch.float64),
                },
            }
            | names
        ]


def collate_volumes(items: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Flatten each volume's samples into one batch. Same-size images are stacked."""
    samples = [s for item in items for s in item]
    images = [s["image"] for s in samples]
    same = len({tuple(i.shape) for i in images}) == 1
    batch = {
        "image": torch.stack(images) if same else images,
        "target": [s["target"] for s in samples],
        "id": [s["id"] for s in samples],
        "patient_id": [s["patient_id"] for s in samples],
    }
    if "affine" in samples[0]:
        batch["affine"] = [s["affine"] for s in samples]
    return batch


class VolumeDetectionDataModule(MedicalDataModule):
    """Base for 3D detection data. Subclasses set ``info`` and return a :class:`Source` whose
    items are ``(MetaTensor (C, X, Y, Z), target(boxes, labels, ignore))`` with world LPS boxes.

    ``spacing`` resamples to that voxel size in millimetres (``null`` keeps the native grid).
    ``intensity_range`` clips to ``[low, high]`` and scales to ``[0, 1]`` (a CT window in HU);
    ``null`` scales each channel between its 0.5th and 99.5th percentiles (MRI).
    ``batch_size`` counts volumes; a training step sees ``batch_size * crops_per_volume``
    patches. ``positive_fraction`` of the crops are centered on a lesion. ``train_transforms``
    run on each crop, as MONAI dictionary transforms over the keys ``image``, ``boxes``
    and ``labels``.
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (96, 96, 64),
        crops_per_volume: int = 4,
        positive_fraction: float = 0.5,
        spacing: tuple[float, float, float] | None = None,
        intensity_range: tuple[float, float] | None = None,
        batch_size: int = 1,
        eval_batch_size: int | None = 1,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("eval_transforms") is not None:
            raise ValueError(
                "3D detection evaluates whole volumes on the resampled grid; eval_transforms "
                "would move voxels away from the affine used to score in world coordinates"
            )
        super().__init__(batch_size=batch_size, eval_batch_size=eval_batch_size, **kwargs)
        if not 0 <= positive_fraction <= 1:
            raise ValueError(f"positive_fraction must be in [0, 1], got {positive_fraction}")
        self.patch_size = tuple(patch_size)
        self.crops_per_volume = crops_per_volume
        self.positive_fraction = positive_fraction
        self.spacing = spacing
        self.intensity_range = intensity_range

    def volume_transforms(self) -> list[Any]:
        """Reorient, resample and scale; the same for training and evaluation."""
        steps: list[Any] = [Orientationd("image", axcodes="RAS", labels=RAS_LABELS)]
        if self.spacing is not None:
            steps.append(Spacingd("image", pixdim=self.spacing, mode="bilinear"))
        if self.intensity_range is not None:
            low, high = self.intensity_range
            steps.append(ScaleIntensityRanged("image", low, high, 0.0, 1.0, clip=True))
        else:
            steps.append(
                ScaleIntensityRangePercentilesd(
                    "image", 0.5, 99.5, 0.0, 1.0, True, channel_wise=True
                )
            )
        return steps

    def transform(self, train: bool) -> Compose:
        if not train:
            return Compose(self.volume_transforms())
        steps = [
            *self.volume_transforms(),
            AffineBoxToImageCoordinated("boxes", "image", affine_lps_to_ras=True),
            Lambdad("boxes", to_voxel_corners),
            RandCropBoxByPosNegLabeld(
                "image",
                "boxes",
                "labels",
                spatial_size=self.patch_size,
                pos=self.positive_fraction,
                neg=1 - self.positive_fraction,
                num_samples=self.crops_per_volume,
                whole_box=True,
            ),
            ClipBoxToImaged("boxes", "labels", "image", remove_empty=True),
        ]
        if self.train_transforms is not None:
            steps.append(self.train_transforms)
        return Compose(steps)

    def loader(self, split: str, train: bool = False) -> DataLoader:
        return DataLoader(
            VolumeSplit(self.source, self.splits[split], self.transform(train), train),
            batch_size=self.batch_size if train else self.eval_batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            collate_fn=collate_volumes,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )


def fold_splits(folds: Sequence[int], test_fold: int, val_fold: int) -> dict[str, list[int]]:
    """Test on ``test_fold``, select the model on ``val_fold``, and train on the rest."""
    known = sorted(set(folds))
    for name, fold in (("test_fold", test_fold), ("val_fold", val_fold)):
        if fold not in known:
            raise ValueError(f"{name} {fold} is not a fold in the manifest; folds are {known}")
    if test_fold == val_fold:
        raise ValueError("test_fold and val_fold must differ")
    return {
        "train": [i for i, f in enumerate(folds) if f not in (test_fold, val_fold)],
        "val": [i for i, f in enumerate(folds) if f == val_fold],
        "test": [i for i, f in enumerate(folds) if f == test_fold],
    }


class VolumeManifestDataModule(VolumeDetectionDataModule):
    """Volumes and boxes listed in a :class:`VolumeManifest` JSON file.

    With ``split: predefined``, a manifest with a ``fold`` per volume tests on ``test_fold``
    and validates on ``val_fold`` (the next fold by default), as in LUNA16's ten-fold
    cross-validation; a manifest with a ``split`` per volume uses it as given. Otherwise use
    ``split: patient``.
    """

    def __init__(
        self,
        manifest: str,
        test_fold: int | None = None,
        val_fold: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.manifest = Path(manifest)
        if not self.manifest.is_file():
            raise FileNotFoundError(
                f"{manifest} does not exist; the dataset's script in scripts/datasets/ "
                "downloads the data and writes it"
            )
        self.spec = VolumeManifest.model_validate(json.loads(self.manifest.read_text()))
        self.test_fold = test_fold
        self.val_fold = val_fold
        self.info = DatasetInfo(
            label_mode="detection",
            class_names=tuple(self.spec.class_names),
            in_channels=len(self.spec.volumes[0].images),
            spatial_dims=3,
            patient_ids_available=self.spec.patient_ids_available,
            license=self.spec.license,
            citation=self.spec.citation,
        )

    def official_splits(self) -> dict[str, list[int]] | None:
        volumes = self.spec.volumes
        if all(v.fold is not None for v in volumes):
            folds = [int(v.fold) for v in volumes if v.fold is not None]
            if self.test_fold is None:
                if self.split == "predefined":
                    raise ValueError(f"{self.manifest} has folds; set test_fold")
                return None
            n = max(folds) + 1
            val = (self.test_fold + 1) % n if self.val_fold is None else self.val_fold
            return fold_splits(folds, self.test_fold, val)
        if all(v.split is not None for v in volumes):
            return {
                n: [i for i, v in enumerate(volumes) if v.split == n]
                for n in ("train", "val", "test")
            }
        return None

    def load(self) -> Source:
        base = self.manifest.parent
        volumes = self.spec.volumes
        return Source(
            dataset=VolumeFiles(
                [[str(base / p) for p in v.images] for v in volumes],
                [target(v.boxes, v.labels, v.ignore) for v in volumes],
            ),
            sample_ids=[v.id for v in volumes],
            patient_ids=[v.patient_id for v in volumes],
            strata=[int(bool(v.boxes)) for v in volumes],
            splits=self.official_splits(),
            fingerprint={
                "manifest": str(self.manifest),
                "dataset": self.spec.dataset,
                "volumes": len(volumes),
                "lesions": sum(len(v.boxes) for v in volumes),
                "test_fold": self.test_fold,
            },
        )
