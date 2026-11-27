"""The data contract every data module follows, and patient-level splitting.

A data module loads a :class:`Source`: an indexable dataset of ``(image, target)`` pairs with
one sample ID and one patient ID per item. The module splits it by patient, checks that no
patient spans two splits, and serves batches as dicts with ``image``, ``target``, ``id``, and
``patient_id``. Images are float tensors in ``[0, 1]`` shaped ``(C, H, W)`` or
``(C, D, H, W)`` before the configured transforms run.
"""

from __future__ import annotations

import logging
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

log = logging.getLogger("opendiagnostics")

SPLIT_NAMES = ("train", "val", "test")
LabelMode = Literal["binary", "multiclass", "multilabel", "detection"]
Transform = Callable[..., Any]


class LeakageError(RuntimeError):
    """Raised when one patient appears in more than one split."""


@dataclass(frozen=True)
class DatasetInfo:
    """What a dataset holds. The Lightning module reads it to size its metrics."""

    label_mode: LabelMode
    class_names: tuple[str, ...]
    in_channels: int
    spatial_dims: int = 2
    patient_ids_available: bool = True
    license: str = "unknown"
    citation: str = ""

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


class Items(Protocol):
    """Anything indexable that returns ``(image, target)`` pairs."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> tuple[Any, Any]: ...


@dataclass
class Source:
    dataset: Items
    sample_ids: list[str]
    patient_ids: list[str]
    strata: list[Hashable] | None = None
    splits: dict[str, list[int]] | None = None
    fingerprint: dict[str, Any] = field(default_factory=dict)


def kfold_split(
    pool: Sequence[int],
    test: Sequence[int],
    patient_ids: Sequence[str],
    strata: Sequence[Hashable] | None,
    folds: int,
    fold: int,
    seed: int = 0,
) -> dict[str, list[int]]:
    """Hold out fold ``fold`` of ``folds`` from ``pool`` as validation and train on the rest.

    Folds are grouped by patient, so no patient spans two folds, and stratified on ``strata``
    when given. ``test`` is passed through untouched.
    """
    if not 0 <= fold < folds:
        raise ValueError(f"fold must be in 0..{folds - 1}, got {fold}")
    pool = list(pool)
    groups = [patient_ids[i] for i in pool]
    labels = [strata[i] for i in pool] if strata is not None else [0] * len(pool)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    train, val = list(splitter.split(pool, labels, groups))[fold]
    return {
        "train": sorted(pool[i] for i in train),
        "val": sorted(pool[i] for i in val),
        "test": list(test),
    }


def patient_split(
    patient_ids: Sequence[str],
    fractions: Sequence[float] = (0.7, 0.15, 0.15),
    strata: Sequence[Hashable] | None = None,
    seed: int = 0,
) -> dict[str, list[int]]:
    """Split sample indices so that no patient spans two splits; fractions are of patients.

    With ``strata``, each patient takes its most common stratum and the split is stratified
    on it, so label prevalence stays close across splits.
    """
    if len(fractions) != 3 or any(f < 0 for f in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError(
            f"fractions must be three non-negative numbers that sum to 1, got {fractions}"
        )
    by_patient: dict[str, list[int]] = defaultdict(list)
    for index, patient in enumerate(patient_ids):
        by_patient[patient].append(index)
    patients = sorted(by_patient)
    labels = (
        [Counter(strata[i] for i in by_patient[p]).most_common(1)[0][0] for p in patients]
        if strata is not None
        else None
    )
    train_frac, val_frac, test_frac = fractions

    def cut(items: list[str], keys: list[Hashable] | None, size: float, salt: int):
        if size <= 0:
            return items, [], keys
        if size >= 1:
            return [], items, None
        try:
            parts = train_test_split(
                items,
                *([keys] if keys else []),
                test_size=size,
                random_state=seed + salt,
                stratify=keys,
            )
        except ValueError:
            warnings.warn(
                "too few patients per stratum; splitting without stratification", stacklevel=3
            )
            parts = train_test_split(items, test_size=size, random_state=seed + salt)
            return parts[0], parts[1], None
        if keys:
            return parts[0], parts[1], parts[2]
        return parts[0], parts[1], None

    rest, test, rest_keys = cut(patients, labels, test_frac, 0)
    held = val_frac / (train_frac + val_frac) if train_frac + val_frac else 0
    train, val, _ = cut(rest, rest_keys, held, 1)
    return {
        name: sorted(i for p in group for i in by_patient[p])
        for name, group in zip(SPLIT_NAMES, (train, val, test), strict=True)
    }


def check_disjoint(splits: Mapping[str, Sequence[int]], patient_ids: Sequence[str]) -> None:
    """Raise :class:`LeakageError` if a patient or a sample index is in two splits."""
    owner: dict[str, str] = {}
    seen: dict[int, str] = {}
    for name, indices in splits.items():
        for index in indices:
            if index in seen:
                raise LeakageError(f"sample {index} is in both {seen[index]!r} and {name!r}")
            seen[index] = name
            patient = patient_ids[index]
            if owner.setdefault(patient, name) != name:
                raise LeakageError(
                    f"patient {patient!r} is in both {owner[patient]!r} and {name!r}"
                )


def summarize(
    splits: Mapping[str, Sequence[int]],
    patient_ids: Sequence[str],
    strata: Sequence[Hashable] | None = None,
) -> dict[str, dict]:
    summary = {}
    for name, indices in splits.items():
        entry: dict = {"samples": len(indices), "patients": len({patient_ids[i] for i in indices})}
        if strata is not None:
            entry["strata"] = dict(sorted(Counter(str(strata[i]) for i in indices).items()))
        summary[name] = entry
    return summary


_to_float = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


def prepare(
    image: Any, target: Any, transform: Transform | None, mode: LabelMode
) -> tuple[Any, Any]:
    """Turn a raw ``(image, target)`` pair into tensors and apply the transform."""
    image = image.float() if isinstance(image, torch.Tensor) else _to_float(image)
    if mode == "detection":
        boxes = tv_tensors.BoundingBoxes(
            torch.as_tensor(target["boxes"], dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(int(image.shape[-2]), int(image.shape[-1])),
        )  # ty: ignore[no-matching-overload]
        target = {"boxes": boxes, "labels": torch.as_tensor(target["labels"], dtype=torch.int64)}
        if transform is not None:
            image, target = transform(image, target)
        return image, {
            "boxes": target["boxes"].as_subclass(torch.Tensor),
            "labels": target["labels"],
        }
    if transform is not None:
        image = transform(image)
    label = torch.as_tensor(np.asarray(target))
    if mode == "multilabel":
        return image, label.reshape(-1).float()
    return image, label.reshape(-1)[0].long()


def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack images and targets; detection images and box targets stay lists."""
    stack = isinstance(samples[0]["target"], torch.Tensor)
    return {
        "image": torch.stack([s["image"] for s in samples])
        if stack
        else [s["image"] for s in samples],
        "target": torch.stack([s["target"] for s in samples])
        if stack
        else [s["target"] for s in samples],
        "id": [s["id"] for s in samples],
        "patient_id": [s["patient_id"] for s in samples],
    }


class SplitDataset(Dataset):
    """A split of a source. Holds only picklable data, so Windows worker processes can load it."""

    def __init__(
        self, source: Source, indices: list[int], transform: Transform | None, mode: LabelMode
    ):
        self.source = source
        self.indices = indices
        self.transform = transform
        self.mode = mode

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        i = self.indices[index]
        image, target = prepare(*self.source.dataset[i], self.transform, self.mode)
        return {
            "image": image,
            "target": target,
            "id": self.source.sample_ids[i],
            "patient_id": self.source.patient_ids[i],
        }


class MedicalDataModule(LightningDataModule):
    """Base data module. Subclasses set ``info`` in ``__init__`` and implement :meth:`load`.

    ``split`` is ``predefined`` (the dataset's official split), ``patient`` (a random
    patient-level split with ``fractions``, stratified by label when ``stratify`` is true), or
    ``kfold``: the official train and validation splits pooled into ``folds`` patient-grouped
    folds, fold ``fold`` held out as validation, and the official test split untouched.
    ``train_transforms`` and ``eval_transforms`` are callables such as a
    ``torchvision.transforms.v2.Compose`` or a MONAI ``Compose``.
    """

    info: DatasetInfo

    def __init__(
        self,
        split: Literal["predefined", "patient", "kfold"] = "predefined",
        fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
        folds: int = 5,
        fold: int = 0,
        stratify: bool = True,
        split_seed: int = 0,
        train_transforms: Transform | None = None,
        eval_transforms: Transform | None = None,
        batch_size: int = 32,
        eval_batch_size: int | None = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.split = split
        self.fractions = fractions
        self.folds = folds
        self.fold = fold
        self.stratify = stratify
        self.split_seed = split_seed
        self.train_transforms = train_transforms
        self.eval_transforms = eval_transforms
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers
        self._source: Source | None = None
        self.splits: dict[str, list[int]] = {}

    def load(self) -> Source:
        raise NotImplementedError

    @property
    def source(self) -> Source:
        if self._source is None:
            raise RuntimeError("call setup() before reading the data")
        return self._source

    def setup(self, stage: str | None = None) -> None:
        if self._source is not None:
            return
        source = self.load()
        if self.split in ("predefined", "kfold"):
            if source.splits is None:
                raise ValueError(
                    f"{type(self).__name__} has no predefined split; use split: patient"
                )
            splits = source.splits
            if self.split == "kfold":
                splits = kfold_split(
                    [*splits["train"], *splits["val"]],
                    splits["test"],
                    source.patient_ids,
                    source.strata if self.stratify else None,
                    self.folds,
                    self.fold,
                    self.split_seed,
                )
        else:
            if not self.info.patient_ids_available:
                log.warning(
                    "this dataset has no patient IDs, so a patient split is a per-image split"
                )
            strata = source.strata if self.stratify else None
            splits = patient_split(source.patient_ids, self.fractions, strata, self.split_seed)
        if self.info.patient_ids_available:
            check_disjoint(splits, source.patient_ids)
        else:
            log.warning("patient IDs are unavailable; patient-level leakage cannot be checked")
        self._source = source
        self.splits = {name: list(splits[name]) for name in SPLIT_NAMES}

    def loader(self, split: str, train: bool = False) -> DataLoader:
        transform = self.train_transforms if train else self.eval_transforms
        return DataLoader(
            SplitDataset(self.source, self.splits[split], transform, self.info.label_mode),
            batch_size=self.batch_size if train else self.eval_batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self.loader("train", train=True)

    def val_dataloader(self) -> DataLoader:
        return self.loader("val")

    def test_dataloader(self) -> DataLoader:
        return self.loader("test")
