"""Patient-level splitting.

Images from one patient are correlated: same anatomy, scanner, and often the same finding.
If they are split across train and test, test metrics measure memorization of the patient
rather than generalization. Every split in this package is made over patients, and
:func:`check_disjoint` verifies it before training starts.
"""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from collections.abc import Hashable, Sequence

import numpy as np

SPLIT_NAMES = ("train", "val", "test")


class LeakageError(RuntimeError):
    """Raised when one patient appears in more than one split."""


def patient_split(
    patient_ids: Sequence[str],
    fractions: Sequence[float] = (0.7, 0.15, 0.15),
    strata: Sequence[Hashable] | None = None,
    seed: int = 0,
) -> dict[str, list[int]]:
    """Split sample indices into train/val/test so that no patient spans two splits.

    With ``strata``, patients are grouped by their most common stratum and each group is
    divided in the requested proportions, so label prevalence stays close across splits.
    Fractions are of patients, not samples.
    """
    if len(fractions) != 3 or any(f < 0 for f in fractions):
        raise ValueError(f"fractions must be three non-negative numbers, got {fractions}")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {sum(fractions)}")
    if strata is not None and len(strata) != len(patient_ids):
        raise ValueError("strata and patient_ids must have the same length")

    by_patient: dict[str, list[int]] = defaultdict(list)
    for index, patient in enumerate(patient_ids):
        by_patient[patient].append(index)
    patients = sorted(by_patient)

    if strata is None:
        groups = {None: patients}
    else:
        patient_stratum = {
            p: Counter(strata[i] for i in by_patient[p]).most_common(1)[0][0] for p in patients
        }
        groups = defaultdict(list)
        for p in patients:
            groups[patient_stratum[p]].append(p)
        if len(groups) > len(patients) / 2:
            warnings.warn(
                f"{len(groups)} strata for {len(patients)} patients; splitting without "
                "stratification",
                stacklevel=2,
            )
            groups = {None: patients}

    rng = np.random.default_rng(seed)
    assigned: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for key in sorted(groups, key=repr):
        members = list(groups[key])
        rng.shuffle(members)
        cuts = np.round(np.cumsum(fractions) * len(members)).astype(int)
        start = 0
        for name, stop in zip(SPLIT_NAMES, cuts, strict=True):
            assigned[name].extend(members[start:stop])
            start = stop

    return {name: sorted(i for p in assigned[name] for i in by_patient[p]) for name in SPLIT_NAMES}


def check_disjoint(splits: dict[str, Sequence[int]], patient_ids: Sequence[str]) -> None:
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
    splits: dict[str, Sequence[int]],
    patient_ids: Sequence[str],
    strata: Sequence[Hashable] | None = None,
) -> dict[str, dict]:
    """Counts per split, for the run record."""
    summary = {}
    for name, indices in splits.items():
        entry: dict = {
            "samples": len(indices),
            "patients": len({patient_ids[i] for i in indices}),
        }
        if strata is not None:
            counts = Counter(str(strata[i]) for i in indices)
            entry["strata"] = dict(sorted(counts.items()))
        summary[name] = entry
    return summary
