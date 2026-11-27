"""Download the CBIS-DDSM images the manifest needs from TCIA and write its box manifest.

    uv run python scripts/datasets/cbis_ddsm.py [--root data/cbis-ddsm] [--workers 4] [--delete-archives]

The four case-description CSVs give the official train and test split and, per lesion, the
full mammogram and the ROI series. Only those series are fetched, one ZIP each, through
TCIA's public NBIA REST API (no account), so the cropped-lesion series are skipped where TCIA
stores them apart. Each lesion's box is the bounding box of its ROI mask, the ROI image with
the mammogram's size. The patient is the participant, the ``P_00001`` part of the subject ID.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pydicom
import torch
from common import download, extract, write_manifest
from torchvision.ops import masks_to_boxes

from opendiagnostics.data.manifest import read_image

CSV_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/"
CASES = {
    ("mass", "train"): "mass_case_description_train_set.csv",
    ("calcification", "train"): "calc_case_description_train_set.csv",
    ("mass", "test"): "mass_case_description_test_set.csv",
    ("calcification", "test"): "calc_case_description_test_set.csv",
}
NBIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"
PARTICIPANT = re.compile(r"P_\d{5}")


@dataclass(frozen=True)
class Lesion:
    label: str
    split: str
    participant: str
    side: str
    view: str
    pathology: str
    image_subject: str
    roi_subject: str


@dataclass
class Mammogram:
    participant: str
    split: str
    side: str
    view: str
    boxes: list[tuple[str, str, list[float]]] = field(default_factory=list)


def _key(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", header.strip().lower()).strip("_")


def _subject(path: str) -> str:
    return path.strip().strip('"').split("/")[0]


def read_cases(path: Path, label: str, split: str) -> list[Lesion]:
    """Lesions from one case-description CSV, its headers matched case- and space-insensitively."""
    with path.open(newline="") as handle:
        rows = [{_key(k): v for k, v in row.items()} for row in csv.DictReader(handle)]
    needed = {"patient_id", "left_or_right_breast", "image_view", "pathology"}
    needed |= {"image_file_path", "roi_mask_file_path"}
    missing = needed - set(rows[0]) if rows else needed
    if missing:
        raise SystemExit(f"{path} lacks columns {sorted(missing)}")
    lesions = []
    for row in rows:
        participant = PARTICIPANT.search(row["patient_id"])
        if participant is None:
            raise SystemExit(f"{path}: patient_id {row['patient_id']!r} has no P_xxxxx part")
        lesions.append(
            Lesion(
                label=label,
                split=split,
                participant=participant.group(0),
                side=row["left_or_right_breast"].strip(),
                view=row["image_view"].strip(),
                pathology=row["pathology"].strip(),
                image_subject=_subject(row["image_file_path"]),
                roi_subject=_subject(row["roi_mask_file_path"]),
            )
        )
    return lesions


def series_to_fetch(lesions: list[Lesion], series: list[dict[str, Any]]) -> dict[str, list[str]]:
    """The series UIDs each subject needs: all of a mammogram's, and of an ROI subject only the
    mask series when TCIA lists the masks apart from the crops."""
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in series:
        by_subject[entry["PatientID"]].append(entry)
    needed: dict[str, list[str]] = {}
    for subject in sorted({lesion.image_subject for lesion in lesions}):
        needed[subject] = [e["SeriesInstanceUID"] for e in by_subject[subject]]
    for subject in sorted({lesion.roi_subject for lesion in lesions}):
        entries = by_subject[subject]
        masks = [e for e in entries if "mask" in str(e.get("SeriesDescription", "")).lower()]
        needed[subject] = [e["SeriesInstanceUID"] for e in masks or entries]
    absent = sorted(s for s, uids in needed.items() if not uids)
    if absent:
        print(f"{len(absent)} subjects are not in TCIA's series list, e.g. {absent[0]}")
    return needed


def fetch_series(
    needed: dict[str, list[str]],
    root: Path,
    workers: int,
    delete_archives: bool,
    fetch: Callable[[str, Path], Path] = download,
) -> None:
    def one(subject: str, uid: str) -> None:
        dest = root / "dicom" / subject / uid
        if (dest / ".complete").exists():
            return
        archive = fetch(f"{NBIA}getImage?SeriesInstanceUID={uid}", root / "archives" / f"{uid}.zip")
        extract(archive, dest)
        if delete_archives:
            archive.unlink(missing_ok=True)

    jobs = [(subject, uid) for subject, uids in needed.items() for uid in uids]
    with ThreadPoolExecutor(workers) as pool:
        for _ in pool.map(lambda job: one(*job), jobs):
            pass


def _dicoms(folder: Path) -> list[Path]:
    return sorted(folder.rglob("*.dcm"))


def _shape(path: Path) -> tuple[int, int]:
    header = pydicom.dcmread(path, stop_before_pixels=True)
    return int(header.Rows), int(header.Columns)


def mask_box(roi_files: list[Path], shape: tuple[int, int]) -> list[float] | None:
    """The box around the first binary ROI image with the mammogram's shape, ``x2`` and ``y2``
    one past the last mask pixel."""
    for path in roi_files:
        if _shape(path) != shape:
            continue
        image = read_image(path)[0]
        if set(image.flat) - {0.0, 1.0} or not image.any():
            continue
        x1, y1, x2, y2 = masks_to_boxes(torch.from_numpy(image > 0)[None])[0].tolist()
        return [x1, y1, x2 + 1, y2 + 1]
    return None


def _pixels(path: Path) -> str:
    return hashlib.sha256(pydicom.dcmread(path).PixelData).hexdigest()


def mammograms(lesions: list[Lesion], dicom: Path) -> tuple[dict[Path, Mammogram], Counter]:
    """Boxes per mammogram file. A film listed under both mass and calcification cases is
    kept once with both kinds of box when the two files hold the same pixels."""
    images: dict[Path, Mammogram] = {}
    skipped: Counter = Counter()
    for lesion in lesions:
        files = _dicoms(dicom / lesion.image_subject)
        if len(files) != 1:
            skipped["mammogram missing or not one file"] += 1
            continue
        box = mask_box(_dicoms(dicom / lesion.roi_subject), _shape(files[0]))
        if box is None:
            skipped["no ROI mask with the mammogram's size"] += 1
            continue
        image = images.setdefault(
            files[0], Mammogram(lesion.participant, lesion.split, lesion.side, lesion.view)
        )
        image.boxes.append((lesion.label, lesion.pathology, box))
    groups: dict[tuple[str, ...], list[Path]] = defaultdict(list)
    for path, image in images.items():
        groups[(image.participant, image.split, image.side, image.view)].append(path)
    for paths in groups.values():
        kept: dict[str, Path] = {}
        for path in paths:
            digest = _pixels(path) if len(paths) > 1 else ""
            if digest in kept:
                images[kept[digest]].boxes += images.pop(path).boxes
                skipped["duplicate film merged"] += 1
            else:
                kept[digest] = path
    return images, skipped


def drop_test_participants_from_train(images: dict[Path, Mammogram]) -> int:
    """Mass and calcification cases were split separately, so a participant can be in both
    splits. The official test split stays whole; their training images are dropped."""
    tested = {image.participant for image in images.values() if image.split == "test"}
    leaked = [
        p for p, image in images.items() if image.split == "train" and image.participant in tested
    ]
    for path in leaked:
        del images[path]
    return len(leaked)


def manifest_rows(images: dict[Path, Mammogram], root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "patient_id": image.participant,
            "split": image.split,
            "label": label,
            "x1": box[0],
            "y1": box[1],
            "x2": box[2],
            "y2": box[3],
            "pathology": pathology,
        }
        for path, image in sorted(images.items())
        for label, pathology, box in image.boxes
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path("data/cbis-ddsm"))
    parser.add_argument("--workers", type=int, default=4, help="series downloaded at once")
    parser.add_argument(
        "--delete-archives", action="store_true", help="delete each series ZIP once unpacked"
    )
    args = parser.parse_args(argv)
    root: Path = args.root
    lesions = []
    for (label, split), name in CASES.items():
        lesions += read_cases(download(CSV_URL + name, root / "csv" / name), label, split)
    listing = download(f"{NBIA}getSeries?Collection=CBIS-DDSM", root / "csv" / "series.json")
    needed = series_to_fetch(lesions, json.loads(listing.read_text()))
    fetch_series(needed, root, args.workers, args.delete_archives)
    images, skipped = mammograms(lesions, root / "dicom")
    leaked = drop_test_participants_from_train(images)
    path = write_manifest(root / "manifest.csv", manifest_rows(images, root))
    print(
        f"wrote {path}: {len(images)} mammograms, "
        f"{sum(len(i.boxes) for i in images.values())} boxes from {len(lesions)} lesions; "
        f"{leaked} training mammograms dropped because their participant is in test; {dict(skipped)}"
    )


if __name__ == "__main__":
    main()
