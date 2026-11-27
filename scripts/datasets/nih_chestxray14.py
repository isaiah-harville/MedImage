"""Download NIH ChestX-ray14 from the NIH Clinical Center's Box folder and write manifest.csv.

    uv run python scripts/datasets/nih_chestxray14.py data/nih_chestxray14 [--delete-archives]

Twelve image archives (45.08 GB) and three metadata files (11 MB) come from
https://nihcc.app.box.com/v/ChestXray-NIHCC, the folder NIH's own batch_download_zips.py
points at. Box publishes no checksums, so each file is checked against the size the folder
lists. The archives unpack 112,120 PNGs into images/, and with --delete-archives each archive
is deleted once unpacked.

manifest.csv has one row per image: ``images/<Image Index>``, the ``Patient ID`` from
Data_Entry_2017_v2020.csv, the findings joined by ``|`` (empty for "No Finding"), and
``test`` for images in NIH's test_list.txt or ``train`` for train_val_list.txt. NIH gives no
validation list; the model config holds out patients from ``train`` as validation.
"""

from __future__ import annotations

import csv
import urllib.request
from collections.abc import Iterable
from pathlib import Path

from _download import DownloadError, Opener, fetch, fetch_and_extract, parser, write_manifest

BOX = (
    "https://nihcc.app.box.com/index.php?rm=box_download_shared_file"
    "&vanity_name=ChestXray-NIHCC&file_id=f_{}"
)
METADATA = {
    "Data_Entry_2017_v2020.csv": (219760887468, 9_003_496),
    "train_val_list.txt": (256056636701, 1_470_907),
    "test_list.txt": (256055473534, 435_131),
}
ARCHIVES = {
    "images_001.tar.gz": (219764235225, 2_008_470_987),
    "images_002.tar.gz": (219767703471, 3_952_623_504),
    "images_003.tar.gz": (219770039352, 3_929_234_850),
    "images_004.tar.gz": (221185642661, 3_838_903_983),
    "images_005.tar.gz": (219776556743, 3_935_496_531),
    "images_006.tar.gz": (219777758783, 3_986_301_172),
    "images_007.tar.gz": (220610700915, 4_016_328_426),
    "images_008.tar.gz": (219776273384, 4_018_347_353),
    "images_009.tar.gz": (219782291318, 4_111_327_929),
    "images_010.tar.gz": (219781375034, 4_181_556_296),
    "images_011.tar.gz": (219777519815, 4_187_084_020),
    "images_012.tar.gz": (219778785923, 2_914_187_733),
}
FINDINGS = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)
NO_FINDING = "No Finding"


def manifest_rows(
    entries: Iterable[dict[str, str]], train_val: set[str], test: set[str]
) -> list[dict[str, str]]:
    """One manifest row per Data_Entry row, labelled with its official split."""
    if overlap := train_val & test:
        raise DownloadError(f"{len(overlap)} images are in both NIH lists, e.g. {min(overlap)}")
    rows = []
    for entry in entries:
        image = entry["Image Index"]
        findings = [f for f in entry["Finding Labels"].split("|") if f != NO_FINDING]
        if unknown := sorted(set(findings) - set(FINDINGS)):
            raise DownloadError(f"{image} has findings outside the 14 NIH labels: {unknown}")
        split = "test" if image in test else "train" if image in train_val else None
        if split is None:
            raise DownloadError(f"{image} is in neither train_val_list.txt nor test_list.txt")
        rows.append(
            {
                "path": f"images/{image}",
                "patient_id": entry["Patient ID"],
                "label": "|".join(findings),
                "split": split,
            }
        )
    if len(rows) != len(train_val) + len(test):
        raise DownloadError(
            f"Data_Entry has {len(rows)} images, the NIH lists {len(train_val) + len(test)}"
        )
    return rows


def _names(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def main(argv: list[str] | None = None, opener: Opener = urllib.request.urlopen) -> Path:
    args = parser(__doc__.splitlines()[0]).parse_args(argv)
    out: Path = args.out
    for name, (file_id, size) in METADATA.items():
        fetch(BOX.format(file_id), out / name, size, opener=opener)
    for name, (file_id, size) in ARCHIVES.items():
        fetch_and_extract(
            BOX.format(file_id), out / name, size, out, args.delete_archives, opener=opener
        )
    with (out / "Data_Entry_2017_v2020.csv").open(newline="") as handle:
        rows = manifest_rows(
            csv.DictReader(handle),
            _names(out / "train_val_list.txt"),
            _names(out / "test_list.txt"),
        )
    if missing := [r["path"] for r in rows if not (out / r["path"]).exists()]:
        raise DownloadError(f"{len(missing)} listed images are not on disk, e.g. {missing[0]}")
    count = write_manifest(out / "manifest.csv", rows)
    print(f"wrote {out / 'manifest.csv'} with {count} images")
    return out / "manifest.csv"


if __name__ == "__main__":
    main()
