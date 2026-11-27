"""FROC analysis and the LUNA16 competition performance metric (CPM).

The rules follow the LUNA16 evaluation script, ``noduleCADEvaluationLUNA16.py``:

* A detection hits a lesion when its center lies strictly inside the lesion's sphere. Our
  lesions are boxes, so the sphere generalizes to the ellipsoid inscribed in the box. A
  LUNA16 box is a cube whose side is the nodule diameter, so there the rule is exactly the
  script's ``distance < diameter / 2``.
* Each detected lesion counts once, at the highest score among the detections that hit it.
  The other detections that hit it are dropped, not counted as false positives.
* Detections that hit an ``ignore`` box (LUNA16's ``annotations_excluded.csv``) are dropped.
* Every other detection is a false positive.
* The curve is scikit-learn's ROC curve over the kept detections, rescaled to false
  positives per scan and to sensitivity over all lesions, missed ones included.

CPM is the mean sensitivity at 1/8, 1/4, 1/2, 1, 2, 4 and 8 false positives per scan.
The script reads those values off a 10,001-point grid; we interpolate at the rates
themselves. MONAI's ``compute_froc_curve_data`` is not used because it drops the operating
point that keeps every detection.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import roc_curve

from opendiagnostics.metrics.bootstrap import Estimate, bootstrap

FP_RATES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
BOX_COLUMNS = ("x1", "y1", "z1", "x2", "y2", "z2")


@dataclass(frozen=True)
class ScanHits:
    """One scan's detections after matching: the score of each detected lesion, the scores
    of the false positives, and how many lesions the scan has."""

    lesion_scores: np.ndarray
    false_positive_scores: np.ndarray
    lesions: int


@dataclass(frozen=True)
class Scan:
    """Boxes are ``(x1, y1, z1, x2, y2, z2)`` in one shared world frame, in millimetres."""

    id: str
    patient_id: str
    boxes: np.ndarray
    scores: np.ndarray
    truth: np.ndarray
    ignore: np.ndarray


def _boxes(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1, 6)


def inside(points: ArrayLike, boxes: ArrayLike) -> np.ndarray:
    """``(P, B)``: whether each point lies strictly inside each box's inscribed ellipsoid."""
    points, boxes = np.asarray(points, np.float64).reshape(-1, 3), _boxes(boxes)
    center = (boxes[:, :3] + boxes[:, 3:]) / 2
    half = (boxes[:, 3:] - boxes[:, :3]) / 2
    offset = (points[:, None, :] - center[None]) / half[None]
    return (offset**2).sum(-1) < 1


def match_scan(boxes, scores, truth, ignore) -> ScanHits:
    boxes, scores = _boxes(boxes), np.asarray(scores, np.float64).reshape(-1)
    truth, ignore = _boxes(truth), _boxes(ignore)
    centers = (boxes[:, :3] + boxes[:, 3:]) / 2
    on_truth, on_ignore = inside(centers, truth), inside(centers, ignore)
    detected = on_truth.any(0)
    best = np.array([scores[on_truth[:, j]].max() for j in np.flatnonzero(detected)])
    false = ~(on_truth.any(1) | on_ignore.any(1))
    return ScanHits(best.reshape(-1), scores[false], len(truth))


def froc_curve(scans: Sequence[ScanHits]) -> tuple[np.ndarray, np.ndarray]:
    """False positives per scan and sensitivity at each operating point, FP rate ascending."""
    lesions = sum(s.lesions for s in scans)
    hits = np.concatenate([s.lesion_scores for s in scans])
    false = np.concatenate([s.false_positive_scores for s in scans])
    if not lesions:
        raise ValueError("FROC needs at least one lesion")
    if not len(hits):
        return np.array([0.0, len(false) / len(scans)]), np.zeros(2)
    if not len(false):
        return np.zeros(2), np.array([0.0, len(hits) / lesions])
    target = np.r_[np.ones(len(hits)), np.zeros(len(false))]
    fpr, tpr, _ = roc_curve(target, np.r_[hits, false])
    return fpr * len(false) / len(scans), tpr * len(hits) / lesions


def froc_metrics(scans: Sequence[ScanHits]) -> dict[str, float]:
    """Sensitivity at each LUNA16 false-positive rate and their mean, the CPM. NaN when no
    scan has a lesion."""
    if not sum(s.lesions for s in scans):
        return {"cpm": float("nan")} | {f"sensitivity_at_{r:g}_fp": float("nan") for r in FP_RATES}
    fps, sensitivity = froc_curve(scans)
    values = np.interp(FP_RATES, fps, sensitivity)
    return {"cpm": float(values.mean())} | {
        f"sensitivity_at_{r:g}_fp": float(v) for r, v in zip(FP_RATES, values, strict=True)
    }


def scan_rows(scan: Scan, class_name: str) -> list[dict]:
    """The rows of a predictions CSV for one scan: one ``scan`` row, then its boxes."""
    blank = {"label": "", "score": ""} | dict.fromkeys(BOX_COLUMNS, "")
    rows: list[dict] = [{"id": scan.id, "patient_id": scan.patient_id, "kind": "scan"} | blank]
    kinds = (("truth", scan.truth, None), ("ignore", scan.ignore, None))
    for kind, boxes, scores in (*kinds, ("prediction", scan.boxes, scan.scores)):
        for n, box in enumerate(_boxes(boxes)):
            rows.append(
                {
                    "id": scan.id,
                    "patient_id": scan.patient_id,
                    "kind": kind,
                    "label": class_name,
                    "score": "" if scores is None else float(scores[n]),
                }
                | {c: float(v) for c, v in zip(BOX_COLUMNS, box, strict=True)}
            )
    return rows


def read_scans(path: Path) -> list[Scan]:
    """Read a ``predictions_<split>.csv`` written by a ``VolumeDetector`` run."""
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "kind" not in rows[0]:
        raise ValueError(f"{path} is not a 3D detection predictions file")
    order: dict[str, dict] = {}
    for row in rows:
        entry = order.setdefault(
            row["id"],
            {"patient_id": row["patient_id"], "truth": [], "ignore": [], "boxes": [], "scores": []},
        )
        if row["kind"] == "scan":
            continue
        box = [float(row[c]) for c in BOX_COLUMNS]
        if row["kind"] == "prediction":
            entry["boxes"].append(box)
            entry["scores"].append(float(row["score"]))
        else:
            entry[row["kind"]].append(box)
    return [
        Scan(
            id=sample,
            patient_id=e["patient_id"],
            boxes=_boxes(e["boxes"]),
            scores=np.asarray(e["scores"], np.float64),
            truth=_boxes(e["truth"]),
            ignore=_boxes(e["ignore"]),
        )
        for sample, e in order.items()
    ]


def scan_metrics(
    scans: Sequence[Scan], replicates: int = 1000, confidence: float = 0.95, seed: int = 0
) -> dict[str, Estimate]:
    """FROC sensitivities and CPM with percentile intervals that resample whole patients."""
    hits = [match_scan(s.boxes, s.scores, s.truth, s.ignore) for s in scans]
    return bootstrap(
        lambda index: froc_metrics([hits[i] for i in index]),
        len(hits),
        groups=[s.patient_id for s in scans],
        replicates=replicates,
        confidence=confidence,
        seed=seed,
    )


def pool(paths: Sequence[Path]) -> list[Scan]:
    """The scans of several prediction files, such as the test folds of a cross-validation.
    A scan may appear in only one file."""
    scans: list[Scan] = []
    owner: dict[str, Path] = {}
    for path in paths:
        for scan in read_scans(path):
            if scan.id in owner:
                raise ValueError(f"scan {scan.id} is in both {owner[scan.id]} and {path}")
            owner[scan.id] = path
            scans.append(scan)
    return scans
