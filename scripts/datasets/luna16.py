# /// script
# requires-python = ">=3.11"
# dependencies = ["SimpleITK>=2.3"]
# ///
"""Download LUNA16 from Zenodo and write data/luna16/manifest.json.

    uv run scripts/datasets/luna16.py [--root data/luna16] [--delete-archives]

About 66 GB of zips (records 3723295 and 4121926, CC BY 4.0), each MD5-checked. Every
MetaImage scan becomes ``images/<seriesuid>.nii.gz``. Rerunning resumes: finished downloads
and converted scans are skipped, and a partial download continues where it stopped.
``--delete-archives`` deletes each subset zip once its scans are converted.

Each scan's ``fold`` is its official subset (0-9), for the ten-fold protocol. Nodules in
``annotations.csv`` become cubes whose side is the diameter, centered on the nodule, in
world LPS millimetres (LUNA16's coordinates are ITK physical points, which are LPS).
``annotations_excluded.csv`` from the evaluation script becomes ``ignore`` boxes, with the
script's 10 mm diameter where none is given. Patient IDs come from TCIA's LIDC-IDRI series
list, so a patient's two scans can be checked for leakage across folds.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _volumes import (
    cube,
    download,
    extract,
    get_json,
    log,
    tcia_series,
    write_manifest,
    write_nifti,
)

RECORDS = ("3723295", "4121926")
CITATION = (
    "Setio AAA et al. Validation, comparison, and combination of algorithms for automatic "
    "detection of pulmonary nodules in computed tomography images: the LUNA16 challenge. "
    "Medical Image Analysis 42, 2017. doi:10.1016/j.media.2017.06.015"
)
UNSIZED_DIAMETER = 10.0


def zenodo_files() -> dict[str, dict]:
    files = {}
    for record in RECORDS:
        for f in get_json(f"https://zenodo.org/api/records/{record}")["files"]:
            files[f["key"]] = {
                "url": f["links"]["self"],
                "md5": f["checksum"].removeprefix("md5:"),
                "size": f["size"],
            }
    return files


def read_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def boxes_by_series(rows: list[dict[str, str]]) -> dict[str, list[list[float]]]:
    """Cubes from LUNA16 annotation rows, keyed by series UID."""
    boxes: dict[str, list[list[float]]] = {}
    for row in rows:
        diameter = float(row["diameter_mm"])
        center = [float(row[k]) for k in ("coordX", "coordY", "coordZ")]
        boxes.setdefault(row["seriesuid"], []).append(
            cube(center, diameter if diameter > 0 else UNSIZED_DIAMETER)
        )
    return boxes


def manifest(
    folds: dict[str, int],
    patients: dict[str, str],
    annotations: list[dict[str, str]],
    excluded: list[dict[str, str]],
) -> dict:
    nodules, ignore = boxes_by_series(annotations), boxes_by_series(excluded)
    missing = sorted(uid for uid in folds if uid not in patients)
    if missing:
        log(f"{len(missing)} scans have no TCIA patient ID; leakage cannot be checked")
    volumes = []
    for uid in sorted(folds, key=lambda u: (folds[u], u)):
        found = nodules.get(uid, [])
        volumes.append(
            {
                "id": uid,
                "patient_id": patients.get(uid, uid),
                "images": [f"images/{uid}.nii.gz"],
                "fold": folds[uid],
                "boxes": found,
                "labels": [0] * len(found),
                "ignore": ignore.get(uid, []),
            }
        )
    return {
        "dataset": "LUNA16",
        "license": "CC BY 4.0",
        "citation": CITATION,
        "class_names": ["nodule"],
        "coordinates": "world-lps-mm",
        "patient_ids_available": not missing,
        "volumes": volumes,
    }


def cross_fold_patients(folds: dict[str, int], patients: dict[str, str]) -> list[str]:
    seen: dict[str, set[int]] = {}
    for uid, fold in folds.items():
        if uid in patients:
            seen.setdefault(patients[uid], set()).add(fold)
    return sorted(p for p, f in seen.items() if len(f) > 1)


def convert_subset(archive: Path, root: Path) -> list[str]:
    """Convert every scan in one subset zip; return the series UIDs it holds."""
    images = root / "images"
    with zipfile.ZipFile(archive) as handle:
        headers = [n for n in handle.namelist() if n.endswith(".mhd")]
    uids = []
    for header in headers:
        uid = Path(header).stem
        uids.append(uid)
        dest = images / f"{uid}.nii.gz"
        if dest.exists():
            continue
        scratch = root / "extract"
        extract(archive, scratch, [header, header.removesuffix(".mhd") + ".raw"])
        write_nifti(scratch / header, dest, integer=True)
        shutil.rmtree(scratch)
    return uids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path("data/luna16"))
    parser.add_argument("--delete-archives", action="store_true")
    parser.add_argument("--subsets", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args(argv)
    root, archives = args.root, args.root / "archives"
    files = zenodo_files()

    def fetch(name: str) -> Path:
        f = files[name]
        log(f"{name}: {f['size'] / 1e9:.2f} GB")
        return download(f["url"], archives / name, md5=f["md5"], size=f["size"])

    annotations = read_rows(fetch("annotations.csv").read_text())
    with zipfile.ZipFile(fetch("evaluationScript.zip")) as handle:
        name = next(n for n in handle.namelist() if n.endswith("annotations_excluded.csv"))
        excluded = read_rows(handle.read(name).decode())

    folds_file = root / "folds.json"
    folds: dict[str, int] = json.loads(folds_file.read_text()) if folds_file.exists() else {}
    for subset in args.subsets:
        if subset in folds.values() and not (archives / f"subset{subset}.zip").exists():
            continue
        archive = fetch(f"subset{subset}.zip")
        for uid in convert_subset(archive, root):
            folds[uid] = subset
        folds_file.write_text(json.dumps(folds, indent=1) + "\n")
        if args.delete_archives:
            archive.unlink()

    patients = {s["SeriesInstanceUID"]: s["PatientID"] for s in tcia_series("LIDC-IDRI", "CT")}
    if spanning := cross_fold_patients(folds, patients):
        log(f"patients with scans in two subsets (setup() will refuse this split): {spanning}")
    write_manifest(root / "manifest.json", manifest(folds, patients, annotations, excluded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
