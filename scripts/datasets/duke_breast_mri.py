# /// script
# requires-python = ">=3.11"
# dependencies = ["SimpleITK>=2.3", "pydicom>=3.0", "openpyxl>=3.1"]
# ///
"""Download the two DCE-MRI phases the tumor task needs from Duke-Breast-Cancer-MRI.

    uv run scripts/datasets/duke_breast_mri.py [--root data/duke-breast-mri]
        [--delete-archives] [--limit N]

The collection is about 368 GB (CC BY-NC 4.0, TCIA's public REST API, no account). This
fetches only the fat-saturated pre-contrast and first post-contrast series of each patient,
about 151 GB for 917 of the 922 patients, as ``images/<series uid>.nii.gz``; the model sees
them as two channels. Rerunning skips converted series. ``--delete-archives`` deletes each
series' zip once it is converted.

Series are picked from their TCIA SeriesDescription: the first post-contrast phase is
``Ph1/...``, ``... 1st pass`` or ``1st ...``; the pre-contrast phase says ``pre`` or is the
unprefixed series of a ``PhN/`` family. Non-fat-saturated T1 and segmentation series are
never picked. Patients without exactly one of each are skipped and listed.

``Annotation_Boxes.xlsx`` gives one tumor box per patient as start and end row, column and
slice. Following the collection page, rows and columns are read as 1-based pixel indices
and slice numbers as 1-based positions in ascending height (ImagePositionPatient along the
slice normal). The box is mapped to world LPS millimetres through the first post-contrast
series' geometry, covering whole pixels and half a slice gap past the end slices.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from _volumes import (
    by_height,
    download,
    log,
    pixel_box,
    slice_gap,
    tcia_series,
    tcia_volume,
    write_manifest,
)

BOXES_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/Annotation_Boxes.xlsx"
CITATION = (
    "Saha A et al. A machine learning approach to radiogenomics of breast cancer: a study of "
    "922 subjects and 529 DCE-MRI features. British Journal of Cancer 119(4), 2018. "
    "doi:10.1038/s41416-018-0185-8. Data: doi:10.7937/TCIA.e3sv-re93"
)
BOX_COLUMNS = ("Start Row", "End Row", "Start Column", "End Column", "Start Slice", "End Slice")


def _name(series: dict[str, Any]) -> str:
    return " ".join(str(series.get("SeriesDescription") or "").lower().split())


def _excluded(name: str) -> bool:
    return (
        name == "segmentation"
        or name.startswith("ax t1")
        or "non fs" in name
        or "t1 bilateral" in name
        or "ideal" in name
    )


def choose_series(series: list[dict[str, Any]]) -> tuple[dict, dict] | str:
    """The (pre-contrast, first post-contrast) series of one patient, or why there is none."""
    names = [_name(s) for s in series]
    families = {n.split("/", 1)[1] for n in names if n.startswith("ph") and "/" in n}
    pre, post = [], []
    for s, n in zip(series, names, strict=True):
        if _excluded(n):
            continue
        if n.startswith("ph1/") or "1st pas" in n or n.startswith("1st "):
            post.append(s)
        elif n.split()[-1].endswith("pre") or n in families:
            pre.append(s)
    if len(pre) != 1 or len(post) != 1:
        return (
            f"{len(pre)} pre-contrast and {len(post)} first post-contrast series in {sorted(names)}"
        )
    return pre[0], post[0]


def tumor_box(row: dict[str, Any], geometry: dict[str, dict[str, list[float]]]) -> list[float]:
    """The world LPS box of one Annotation_Boxes.xlsx row on its series' geometry."""
    slices = by_height(geometry.values())
    top_row, bottom_row, left, right, first, last = (int(row[k]) for k in BOX_COLUMNS)
    if not 1 <= first <= last <= len(slices):
        raise ValueError(f"slices {first}-{last} are outside the series' {len(slices)} slices")
    points = [
        (slices[s - 1], column - 1, pixel_row - 1)
        for s in (first, last)
        for column, pixel_row in ((left, top_row), (right, bottom_row))
    ]
    return pixel_box(points, slice_gap(slices))


def same_grid(a: dict[str, dict[str, list[float]]], b: dict[str, dict[str, list[float]]]) -> bool:
    def grid(g: dict[str, dict[str, list[float]]]) -> list[list[float]]:
        return sorted(s["position"] + s["spacing"] for s in g.values())

    ga, gb = grid(a), grid(b)
    return len(ga) == len(gb) and all(
        abs(x - y) < 1e-2 for p, q in zip(ga, gb, strict=True) for x, y in zip(p, q, strict=True)
    )


def read_boxes(path: Path) -> dict[str, dict[str, Any]]:
    import openpyxl  # ty: ignore[import-not-found]

    sheet = openpyxl.load_workbook(path, read_only=True).active
    if sheet is None:
        raise ValueError(f"{path} has no worksheet")
    rows = sheet.iter_rows(values_only=True)
    header = [str(v).strip() for v in next(rows)]
    missing = {"Patient ID", *BOX_COLUMNS} - set(header)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}; its header is {header}")
    records = (dict(zip(header, values, strict=False)) for values in rows)
    return {str(r["Patient ID"]): r for r in records if r.get("Patient ID")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path("data/duke-breast-mri"))
    parser.add_argument("--delete-archives", action="store_true")
    parser.add_argument("--limit", type=int, help="convert only the first N patients")
    args = parser.parse_args(argv)
    root = args.root
    boxes = read_boxes(download(BOXES_URL, root / "archives" / "Annotation_Boxes.xlsx"))
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in tcia_series("Duke-Breast-Cancer-MRI"):
        by_patient[s["PatientID"]].append(s)
    chosen, skipped = {}, {}
    for patient in sorted(by_patient):
        picked = choose_series(by_patient[patient])
        if isinstance(picked, str):
            skipped[patient] = picked
        elif patient not in boxes:
            skipped[patient] = "no row in Annotation_Boxes.xlsx"
        else:
            chosen[patient] = picked
    patients = sorted(chosen)[: args.limit] if args.limit else sorted(chosen)
    size = sum(s["FileSize"] for p in patients for s in chosen[p])
    log(f"{len(patients)} patients, {size / 1e9:.1f} GB to download")
    volumes = []
    for n, patient in enumerate(patients, start=1):
        log(f"[{n}/{len(patients)}] {patient}")
        pre, post = chosen[patient]
        grids = [
            tcia_volume(s, root, integer=False, delete_archive=args.delete_archives)
            for s in (pre, post)
        ]
        if not same_grid(*grids):
            skipped[patient] = "pre- and post-contrast series are on different voxel grids"
            continue
        volumes.append(
            {
                "id": patient,
                "patient_id": patient,
                "images": [f"images/{s['SeriesInstanceUID']}.nii.gz" for s in (pre, post)],
                "boxes": [tumor_box(boxes[patient], grids[1])],
                "labels": [0],
            }
        )
    for patient, reason in skipped.items():
        log(f"skipped {patient}: {reason}")
    write_manifest(
        root / "manifest.json",
        {
            "dataset": "Duke-Breast-Cancer-MRI, pre-contrast and first post-contrast phases",
            "license": "CC BY-NC 4.0",
            "citation": CITATION,
            "class_names": ["tumor"],
            "coordinates": "world-lps-mm",
            "volumes": volumes,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
