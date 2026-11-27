"""Download the RSNA Pneumonia Detection Challenge (2018) and write its box manifest.

    uv run python scripts/datasets/rsna_pneumonia.py [--root data/rsna-pneumonia] [--delete-archives]

The files are RSNA's open release of the images and adjudicated annotations used in the
challenge, and the mapping to the original NIH image IDs
(https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018).
No Kaggle account is needed. Every box becomes one ``opacity`` row; an annotated image with
no box is a negative. The patient ID is the NIH patient ID, the first eight digits of the NIH
image name, because one NIH patient can have several RSNA images.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pydicom
from common import download, extract, write_manifest

BASE = "https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/"
IMAGES = BASE + "pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip"
ANNOTATIONS = BASE + "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json"
MAPPING = BASE + "pneumonia-challenge-dataset-mappings_2018.json"
LABEL = "opacity"
UID_FIELDS = ("SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID")
NIH_IMAGE = re.compile(r"(?<!\d)(\d{8})_\d{3}(?!\d)")

Box = tuple[float, float, float, float]


def index_dicoms(images: Path) -> dict[str, Path]:
    """Each DICOM file under ``images`` by every identifier it carries: its UIDs, its
    PatientID tag and its file name. An identifier shared by two files is dropped."""
    owners: dict[str, set[Path]] = defaultdict(set)
    for path in sorted(images.rglob("*.dcm")):
        header = pydicom.dcmread(path, stop_before_pixels=True)
        keys = {path.stem} | {str(header.get(f, "")) for f in (*UID_FIELDS, "PatientID")}
        for key in keys - {""}:
            owners[key].add(path)
    return {key: paths.pop() for key, paths in owners.items() if len(paths) == 1}


def annotated_boxes(export: dict[str, Any], index: dict[str, Path]) -> dict[Path, set[Box]]:
    """Boxes per image file from an MD.ai export: ``datasets[].annotations[]``, each with the
    image's UIDs and, for a box, ``data: {x, y, width, height}`` in pixels."""
    images: dict[Path, set[Box]] = {}
    unmatched = 0
    for dataset in export["datasets"]:
        found = Counter()
        for annotation in dataset["annotations"]:
            uids = (annotation.get(f) for f in UID_FIELDS)
            path = next((index[u] for u in uids if u in index), None)
            if path is None:
                unmatched += 1
                continue
            boxes = images.setdefault(path, set())
            data = annotation.get("data") or {}
            if all(k in data for k in ("x", "y", "width", "height")):
                x, y = float(data["x"]), float(data["y"])
                boxes.add((x, y, x + float(data["width"]), y + float(data["height"])))
                found["boxes"] += 1
            found["annotations"] += 1
        print(f"annotations in {dataset.get('name', dataset.get('id'))!r}: {dict(found)}")
    if unmatched:
        print(f"{unmatched} annotations name an image that is not in the download")
    return images


def _string_records(node: Any) -> Iterator[list[str]]:
    if isinstance(node, dict):
        yield [str(v) for v in node.values() if isinstance(v, str | int)]
        for key, value in node.items():
            if isinstance(value, str):
                yield [key, value]
            else:
                yield from _string_records(value)
    elif isinstance(node, list):
        for value in node:
            yield from _string_records(value)


def nih_patients(mapping: Any, index: dict[str, Path]) -> dict[Path, str]:
    """The NIH patient ID of each image, from mapping records that name exactly one NIH image
    (``00000013_005.png``) and exactly one RSNA image by any identifier it carries."""
    patients: dict[Path, str] = {}
    for values in _string_records(mapping):
        nih = {m.group(1) for v in values if (m := NIH_IMAGE.search(v))}
        paths = {index[v] for v in values if v in index}
        if len(nih) == 1 and len(paths) == 1:
            patients[paths.pop()] = nih.pop()
    return patients


def manifest_rows(
    boxes: dict[Path, set[Box]], patients: dict[Path, str], root: Path
) -> list[dict[str, Any]]:
    unmapped = sorted(set(boxes) - set(patients))
    if unmapped:
        raise SystemExit(
            f"{len(unmapped)} of {len(boxes)} annotated images have no NIH patient ID in "
            f"{MAPPING}, for example {unmapped[0].name}. The mapping format may differ from the "
            "records this script reads: an NIH image name such as 00000013_005.png beside a "
            "DICOM UID, PatientID, or file name of the RSNA image."
        )
    rows = []
    for path in sorted(boxes):
        common = {"path": path.relative_to(root).as_posix(), "patient_id": patients[path]}
        if not boxes[path]:
            rows.append(common | {"label": "", "x1": "", "y1": "", "x2": "", "y2": ""})
        for x1, y1, x2, y2 in sorted(boxes[path]):
            rows.append(common | {"label": LABEL, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path("data/rsna-pneumonia"))
    parser.add_argument(
        "--delete-archives", action="store_true", help="delete the image ZIP once unpacked"
    )
    args = parser.parse_args(argv)
    root: Path = args.root
    archives = root / "archives"
    images = root / "images"
    if not (images / ".complete").exists():
        extract(download(IMAGES, archives / IMAGES.rsplit("/", 1)[1]), images, flatten=True)
    if args.delete_archives:
        (archives / IMAGES.rsplit("/", 1)[1]).unlink(missing_ok=True)
    export = json.loads(download(ANNOTATIONS, archives / ANNOTATIONS.rsplit("/", 1)[1]).read_text())
    mapping = json.loads(download(MAPPING, archives / MAPPING.rsplit("/", 1)[1]).read_text())
    print("indexing DICOM headers")
    index = index_dicoms(images)
    boxes = annotated_boxes(export, index)
    rows = manifest_rows(boxes, nih_patients(mapping, index), root)
    path = write_manifest(root / "manifest.csv", rows)
    patients = len({r["patient_id"] for r in rows})
    positives = sum(1 for b in boxes.values() if b)
    print(f"wrote {path}: {len(boxes)} images ({positives} with boxes) from {patients} patients")


if __name__ == "__main__":
    main()
