"""Download CMMD from TCIA and write a patient-level manifest of labelled mammograms.

    uv run python scripts/datasets/cmmd.py data/cmmd [--delete-archives]

Source: The Chinese Mammography Database on TCIA, doi:10.7937/tcia.eqde-4b16 (CC BY 4.0),
through TCIA's public NBIA REST API, which needs no account. Each of the 1,775 series (one per
patient, 5,202 DICOM images, about 23 GB) is a separate zip. TCIA publishes each series' image
count, which is checked after extraction; it publishes no archive size or checksum.

Labels come from the clinical table, one row per breast with a biopsy-confirmed benign or
malignant lesion. A DICOM image is labelled by its patient ID and laterality; images of a
breast with no row in the table are left out. Patients are split into train, val and test
(70/15/15, stratified by label).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import pydicom
from _download import (
    arguments,
    assign_splits,
    download,
    extract,
    find_files,
    relative,
    write_manifest,
)
from openpyxl import load_workbook

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
CLINICAL = (
    "https://www.cancerimagingarchive.net/wp-content/uploads/CMMD_clinicaldata_revision.xlsx",
    71_936,
)
LABELS = {"Benign": "benign", "Malignant": "malignant"}


def read_clinical(path: Path) -> dict[tuple[str, str], str]:
    """Map (patient ID, ``L`` or ``R``) to the breast's label."""
    sheet = load_workbook(path, read_only=True).worksheets[0]
    rows = sheet.iter_rows(values_only=True)
    header = [str(c).strip() for c in next(rows)]
    missing = {"ID1", "LeftRight", "classification"} - set(header)
    if missing:
        raise RuntimeError(f"{path} lacks the columns {sorted(missing)}; has its format changed?")
    column = {name: header.index(name) for name in ("ID1", "LeftRight", "classification")}
    labels: dict[tuple[str, str], str] = {}
    for row in rows:
        if row[column["ID1"]] is None:
            continue
        key = (str(row[column["ID1"]]).strip(), str(row[column["LeftRight"]]).strip())
        label = LABELS[str(row[column["classification"]]).strip()]
        if labels.setdefault(key, label) != label:
            raise RuntimeError(f"{path} gives breast {key} two labels")
    return labels


def laterality(path: Path) -> tuple[str, str]:
    header = pydicom.dcmread(path, stop_before_pixels=True)
    side = header.get("ImageLaterality") or header.get("Laterality")
    if side not in ("L", "R"):
        raise RuntimeError(f"{path} has no image laterality, so it cannot be matched to a label")
    return str(header.PatientID), side


def manifest_rows(
    images: list[tuple[Path, str, str]], labels: dict[tuple[str, str], str], root: Path
) -> list[dict[str, str]]:
    """``images`` holds (path, patient ID, laterality) per DICOM file."""
    kept = [
        (p, patient, labels[(patient, side)])
        for p, patient, side in images
        if (patient, side) in labels
    ]
    print(f"{len(kept)} of {len(images)} images belong to a labelled breast")
    splits = assign_splits([patient for _, patient, _ in kept], [label for *_, label in kept])
    return [
        {"path": relative(p, root), "patient_id": patient, "label": label, "split": split}
        for (p, patient, label), split in zip(kept, splits, strict=True)
    ]


def series_list(root: Path) -> list[dict]:
    path = root / "series.json"
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{API}/getSeries?Collection=CMMD") as response:
            path.write_bytes(response.read())
    return json.loads(path.read_text())


def fetch_series(series: dict, root: Path, delete_archive: bool) -> Path:
    uid = series["SeriesInstanceUID"]
    into = root / "dicom" / uid
    archive = root / "archives" / f"{uid}.zip"
    if not (into / f".{archive.name}.extracted").exists():
        query = urllib.parse.urlencode({"SeriesInstanceUID": uid})
        download(f"{API}/getImage?{query}", archive)
    extract(archive, into, delete_archive)
    count = len(find_files(into, [".dcm"]))
    if count != int(series["ImageCount"]):
        raise RuntimeError(
            f"series {uid} has {count} DICOM files but TCIA lists {series['ImageCount']}; "
            f"delete {into} and {archive} and rerun"
        )
    return into


def main() -> None:
    args = arguments(__doc__.splitlines()[0])
    root: Path = args.root
    series = series_list(root)
    for number, entry in enumerate(series, 1):
        print(f"series {number}/{len(series)}")
        fetch_series(entry, root, args.delete_archives)
    labels = read_clinical(
        download(CLINICAL[0], root / "CMMD_clinicaldata_revision.xlsx", CLINICAL[1])
    )
    files = sorted(p for p in (root / "dicom").rglob("*.dcm"))
    images = [(p, *laterality(p)) for p in files]
    write_manifest(root, manifest_rows(images, labels, root))


if __name__ == "__main__":
    main()
