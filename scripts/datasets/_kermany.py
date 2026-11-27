"""The Kermany OCT and pediatric chest X-ray sets, both in one Mendeley Data archive.

Kermany D, Zhang K, Goldbaum M. Labeled Optical Coherence Tomography (OCT) and Chest X-Ray
Images for Classification. Mendeley Data v3, doi:10.17632/rscbjbr9sj.3, CC BY 4.0. The one
file in version 3, ZhangLabData.zip (8.44 GB), holds both sets under CellData/; Mendeley
publishes its size and SHA-256. Images are named ``<CLASS>-<patient ID>-<image number>.jpeg``
and sit in ``train/<label>/`` or ``test/<label>/``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from _download import Opener, fetch_and_extract, parser, relative, write_manifest

URL = (
    "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/"
    "810b2ce2-11c3-4424-996e-3bef36600907/file_downloaded"
)
ARCHIVE = "ZhangLabData.zip"
SIZE = 8_441_154_111
SHA256 = "acbc42869440eec719361942d0ba560f132742230262f22e78bfb93eb5ea4563"
SPLITS = ("train", "val", "test")
FILENAME = re.compile(r"[A-Z]+-(\d+)-\d+\.jpeg")


def member_name(prefix: str, name: str) -> str | None:
    """Where a zip member lands under the dataset directory, or None to skip it."""
    if not name.startswith(prefix) or not name.endswith(".jpeg"):
        return None
    return name.removeprefix(prefix)


def manifest_rows(paths: Iterable[str], labels: tuple[str, ...]) -> list[dict[str, str]]:
    """Rows for paths such as ``train/CNV/CNV-81630-1.jpeg``, relative to the dataset dir."""
    rows = []
    for path in sorted(paths):
        split, label, filename = path.split("/")
        match = FILENAME.fullmatch(filename)
        if split not in SPLITS or label not in labels or match is None:
            raise RuntimeError(
                f"unexpected file {path}; expected <split>/<label>/<CLASS>-<patient>-<n>.jpeg "
                f"with a split in {SPLITS} and a label in {labels}"
            )
        rows.append({"path": path, "patient_id": match[1], "label": label, "split": split})
    return rows


def main(
    doc: str,
    subset: str,
    labels: tuple[str, ...],
    argv: list[str] | None = None,
    opener: Opener | None = None,
) -> Path:
    cli = parser(doc.splitlines()[0])
    cli.add_argument(
        "--archive",
        type=Path,
        help=f"where to keep {ARCHIVE}; point both Kermany scripts at one copy to download it once",
    )
    args = cli.parse_args(argv)
    root: Path = args.root
    prefix = f"CellData/{subset}/"
    fetch_and_extract(
        URL,
        args.archive or root / ARCHIVE,
        root,
        SIZE,
        delete_archive=args.delete_archives,
        opener=opener,
        sha256=SHA256,
        members=lambda name: member_name(prefix, name),
    )
    paths = [relative(p, root) for p in root.glob("*/*/*.jpeg")]
    return write_manifest(root, manifest_rows(paths, labels))
