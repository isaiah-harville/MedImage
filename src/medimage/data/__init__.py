"""Dataset contract, splits, and collation."""

from medimage.data.base import (
    DatasetInfo,
    DatasetSource,
    Sample,
    SplitDataset,
    collate_classification,
    collate_detection,
)
from medimage.data.splits import LeakageError, check_disjoint, patient_split, summarize

__all__ = [
    "DatasetInfo",
    "DatasetSource",
    "LeakageError",
    "Sample",
    "SplitDataset",
    "check_disjoint",
    "collate_classification",
    "collate_detection",
    "patient_split",
    "summarize",
]
