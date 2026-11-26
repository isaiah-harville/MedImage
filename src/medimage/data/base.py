"""The dataset contract shared by every task.

A dataset source returns *samples*: plain dicts with these keys.

``image``
    ``float32`` tensor shaped ``(C, H, W)`` for 2D or ``(C, D, H, W)`` for 3D.
``label``
    Classification only. ``int64`` scalar for binary and multiclass, ``float32`` vector of
    0/1 for multilabel.
``boxes`` and ``labels``
    Detection only. ``boxes`` is ``float32`` ``(N, 4)`` in pixel ``(x1, y1, x2, y2)`` order,
    where x runs along the last (width) axis. ``labels`` is ``int64`` ``(N,)`` holding
    foreground class indices starting at 0.
``id`` and ``patient_id``
    Strings. ``patient_id`` drives leakage-free splits and the patient-level bootstrap.

Sources are cheap to index: they expose ``patient_ids`` and ``strata`` without loading images,
so splits can be computed before any pixel is read.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch.utils.data import Dataset

Sample = dict[str, Any]
TaskType = Literal["classification", "detection"]
LabelMode = Literal["binary", "multiclass", "multilabel"]


@dataclass(frozen=True)
class DatasetInfo:
    """What a dataset holds. Tasks and models read this to size themselves."""

    task_type: TaskType
    spatial_dims: int
    in_channels: int
    class_names: tuple[str, ...]
    label_mode: LabelMode | None = None
    patient_ids_available: bool = True
    license: str = "unknown"
    citation: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


class DatasetSource(Dataset):
    """Base class for registered datasets. Subclasses fill ``info`` and the per-sample lists."""

    info: DatasetInfo

    def __len__(self) -> int:
        return len(self.sample_ids)

    @property
    def sample_ids(self) -> Sequence[str]:
        raise NotImplementedError

    @property
    def patient_ids(self) -> Sequence[str]:
        """One patient ID per sample. Defaults to the sample ID when patients are unknown."""
        return self.sample_ids

    @property
    def strata(self) -> Sequence[Any] | None:
        """A hashable label per sample for stratified splitting, or ``None``."""
        return None

    def predefined_splits(self) -> dict[str, list[int]] | None:
        """Official ``train``/``val``/``test`` indices, if the dataset ships them."""
        return None

    def fingerprint(self) -> dict[str, Any]:
        """Facts that identify the exact data used (file hashes, versions) for provenance."""
        return {}

    def __getitem__(self, index: int) -> Sample:
        raise NotImplementedError


class SplitDataset(Dataset):
    """A subset of a source with a transform applied to each sample."""

    def __init__(
        self,
        source: DatasetSource,
        indices: Sequence[int],
        transform: Callable[[Sample], Sample] | None = None,
    ) -> None:
        self.source = source
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Sample:
        sample = self.source[self.indices[index]]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def collate_classification(samples: list[Sample]) -> dict[str, Any]:
    return {
        "image": torch.stack([s["image"] for s in samples]),
        "label": torch.stack([torch.as_tensor(s["label"]) for s in samples]),
        "id": [s["id"] for s in samples],
        "patient_id": [s["patient_id"] for s in samples],
    }


def collate_detection(samples: list[Sample]) -> dict[str, Any]:
    """Detection images may differ in size, so they stay a list."""
    return {
        "image": [s["image"] for s in samples],
        "target": [{"boxes": s["boxes"], "labels": s["labels"]} for s in samples],
        "id": [s["id"] for s in samples],
        "patient_id": [s["patient_id"] for s in samples],
    }
