# /// script
# requires-python = ">=3.11"
# dependencies = ["SimpleITK>=2.3", "pydicom>=3.0"]
# ///
"""Download the LIDC-IDRI CT scans from TCIA and write data/lidc-idri/manifest.json.

    uv run scripts/datasets/lidc_idri.py [--root data/lidc-idri] [--min-readers 1]
        [--delete-archives] [--limit N]

All 1,018 CT series, about 128 GB, through TCIA's public REST API without an account
(CC BY 3.0), plus the reader XML in LIDC-XML-only.zip. Each series becomes
``images/<series uid>.nii.gz`` and ``geometry/<series uid>.json`` (every slice's position,
orientation and pixel spacing). Rerunning skips converted series. ``--delete-archives``
deletes each series' zip once it is converted. The manifest is rebuilt from the XML and the
geometry on every run, so ``--min-readers`` can change without downloading again.

Nodule boxes, from up to four readers who did not have to agree:

1. A reader's mark of a nodule of 3 mm or more is its outline (``inclusion`` true). Its box
   covers every outlined pixel and half a slice gap above and below the outer slices.
2. Marks from different readers are one nodule when their boxes overlap (single linkage).
3. A nodule's box is the per-coordinate median of its readers' boxes.
4. A nodule marked by at least ``--min-readers`` readers is a target. The rest, and every
   mark of a nodule under 3 mm or of a non-nodule (a 10 mm cube around the point, as the
   LUNA16 evaluation does for marks without a diameter), are ``ignore`` boxes: detections
   there are neither hits nor false positives.

Pixel coordinates in the XML are read as 0-based column (x) and row (y) indices.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _volumes import (
    cube,
    download,
    log,
    pixel_box,
    slice_gap,
    tcia_series,
    tcia_volume,
    write_manifest,
)

XML_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/LIDC-XML-only.zip"
CITATION = (
    "Armato SG III et al. The Lung Image Database Consortium (LIDC) and Image Database "
    "Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans. "
    "Medical Physics 38(2), 2011. doi:10.1118/1.3528204. Data: doi:10.7937/K9/TCIA.2015.LO9QL9SX"
)
POINT_DIAMETER = 10.0


@dataclass(frozen=True)
class Mark:
    """One reader's mark. ``points`` are ``(SOP Instance UID, column, row)``."""

    reader: str
    kind: str
    points: tuple[tuple[str, float, float], ...]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [e for e in element if _local(e.tag) == name]


def _text(element: ET.Element, name: str) -> str | None:
    found = _children(element, name)
    return found[0].text.strip() if found and found[0].text else None


def parse_reads(text: bytes | str) -> tuple[str | None, list[Mark]]:
    """The series UID and every reader's marks in one LIDC XML file."""
    root = ET.fromstring(text)
    header = _children(root, "ResponseHeader")
    series = _text(header[0], "SeriesInstanceUid") if header else None
    marks = []
    for n, session in enumerate(_children(root, "readingSession")):
        reader = _text(session, "servicingRadiologistID") or f"session-{n}"
        for nodule in _children(session, "unblindedReadNodule"):
            rois = [r for r in _children(nodule, "roi") if _text(r, "inclusion") != "FALSE"]
            points = tuple(
                (
                    _text(roi, "imageSOP_UID") or "",
                    float(_text(e, "xCoord") or 0),
                    float(_text(e, "yCoord") or 0),
                )
                for roi in rois
                for e in _children(roi, "edgeMap")
            )
            outlined = any(len(_children(r, "edgeMap")) > 1 for r in rois)
            if points:
                marks.append(Mark(reader, "nodule" if outlined else "small", points))
        for other in _children(session, "nonNodule"):
            sop = _text(other, "imageSOP_UID") or ""
            points = tuple(
                (sop, float(_text(e, "xCoord") or 0), float(_text(e, "yCoord") or 0))
                for e in _children(other, "locus")
            )
            if points:
                marks.append(Mark(reader, "non_nodule", points))
    return series, marks


def _overlap(a: list[float], b: list[float]) -> bool:
    return all(a[k] < b[k + 3] and b[k] < a[k + 3] for k in range(3))


def cluster(boxes: list[list[float]], readers: list[str]) -> list[tuple[list[float], int]]:
    """Group overlapping boxes of different readers; return each group's median box and
    its number of distinct readers."""
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if readers[i] != readers[j] and _overlap(boxes[i], boxes[j]):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    return [
        (
            [statistics.median(boxes[i][k] for i in members) for k in range(6)],
            len({readers[i] for i in members}),
        )
        for members in groups.values()
    ]


def series_boxes(
    marks: list[Mark], geometry: dict[str, dict[str, list[float]]], min_readers: int
) -> tuple[list[list[float]], list[list[float]]]:
    """Target and ignore boxes for one series from its readers' marks."""
    gap = slice_gap(geometry.values())
    outlines, readers, ignore = [], [], []
    for mark in marks:
        points = [(geometry[sop], x, y) for sop, x, y in mark.points if sop in geometry]
        if not points:
            continue
        box = pixel_box(points, gap)
        if mark.kind == "nodule":
            outlines.append(box)
            readers.append(mark.reader)
        else:
            center = [(box[k] + box[k + 3]) / 2 for k in range(3)]
            ignore.append(cube(center, POINT_DIAMETER))
    targets = []
    for box, count in cluster(outlines, readers):
        (targets if count >= min_readers else ignore).append(box)
    return targets, ignore


def read_xml(archive: Path) -> dict[str, list[Mark]]:
    marks: dict[str, list[Mark]] = {}
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            if name.lower().endswith(".xml"):
                series, found = parse_reads(handle.read(name))
                if series:
                    marks.setdefault(series, []).extend(found)
    return marks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path("data/lidc-idri"))
    parser.add_argument("--min-readers", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--delete-archives", action="store_true")
    parser.add_argument("--limit", type=int, help="convert only the first N series")
    args = parser.parse_args(argv)
    root = args.root
    marks = read_xml(download(XML_URL, root / "archives" / "LIDC-XML-only.zip"))
    series = sorted(tcia_series("LIDC-IDRI", "CT"), key=lambda s: s["SeriesInstanceUID"])
    series = series[: args.limit] if args.limit else series
    log(f"{len(series)} CT series, {sum(s['FileSize'] for s in series) / 1e9:.1f} GB")
    volumes = []
    for n, s in enumerate(series, start=1):
        uid = s["SeriesInstanceUID"]
        log(f"[{n}/{len(series)}] {s['PatientID']} {uid}")
        geometry = tcia_volume(s, root, integer=True, delete_archive=args.delete_archives)
        targets, ignore = series_boxes(marks.get(uid, []), geometry, args.min_readers)
        volumes.append(
            {
                "id": uid,
                "patient_id": s["PatientID"],
                "images": [f"images/{uid}.nii.gz"],
                "boxes": targets,
                "labels": [0] * len(targets),
                "ignore": ignore,
            }
        )
    write_manifest(
        root / "manifest.json",
        {
            "dataset": f"LIDC-IDRI, nodules >= 3 mm marked by >= {args.min_readers} reader(s)",
            "license": "CC BY 3.0",
            "citation": CITATION,
            "class_names": ["nodule"],
            "coordinates": "world-lps-mm",
            "volumes": volumes,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
