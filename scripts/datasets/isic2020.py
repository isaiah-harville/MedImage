"""Download the ISIC 2020 training data and write a patient-level manifest.

    uv run python scripts/datasets/isic2020.py data/isic2020 [--delete-archives]

Source: the ISIC challenge data page, https://challenge.isic-archive.com/data/ (CC BY-NC 4.0).
The official test labels were never released, so the manifest splits the 33,126 training
images by ``patient_id`` into train, val and test (70/15/15, stratified by melanoma). Images
on ISIC's duplicate list are dropped first: one copy is kept when both copies share a label,
and neither when they disagree.

The JPEG archive is 24.7 GB. Sizes and MD5s are from the S3 objects' headers (for a
single-part upload the ETag is the MD5).
"""

from __future__ import annotations

import csv
from pathlib import Path

from _download import (
    arguments,
    assign_splits,
    download,
    fetch_and_extract,
    find_files,
    read_csv,
    relative,
    write_manifest,
)

BASE = "https://isic-archive.s3.amazonaws.com/challenges/2020"
IMAGES = ("ISIC_2020_Training_JPEG.zip", 24_707_698_022, None)
GROUND_TRUTH = ("ISIC_2020_Training_GroundTruth.csv", 2_056_020, "86657c7009b003543b377b62a8745b4c")
DUPLICATES = ("ISIC_2020_Training_Duplicates.csv", 11_500, "62992a2252b492095a54dbc5ce0d5b3e")
LABELS = {"0": "benign", "1": "melanoma"}


def duplicates_to_drop(groups: list[list[str]], label: dict[str, str]) -> set[str]:
    """Keep the first image of each duplicate group, or none when the copies disagree."""
    drop: set[str] = set()
    for group in groups:
        names = sorted(n for n in group if n in label)
        if len({label[n] for n in names}) > 1:
            drop.update(names)
        else:
            drop.update(names[1:])
    return drop


def read_duplicates(path: Path, known: set[str]) -> list[list[str]]:
    """Each row of the duplicate list names images that are copies of one another."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    cells = ([c.strip().removesuffix(".jpg") for c in row] for row in rows[1:])
    groups = [[c for c in row if c in known] for row in cells]
    groups = [g for g in groups if len(g) > 1]
    if not groups:
        raise RuntimeError(f"no row of {path} names two training images; has its format changed?")
    return groups


def manifest_rows(
    truth: list[dict[str, str]], duplicates: list[list[str]], images: dict[str, Path], root: Path
) -> list[dict[str, str]]:
    label = {r["image_name"]: LABELS[r["target"]] for r in truth}
    drop = duplicates_to_drop(duplicates, label)
    kept = [r for r in truth if r["image_name"] not in drop]
    missing = [r["image_name"] for r in kept if r["image_name"] not in images]
    if missing:
        raise RuntimeError(
            f"{len(missing)} images in the ground truth are not on disk: {missing[:5]}"
        )
    splits = assign_splits([r["patient_id"] for r in kept], [label[r["image_name"]] for r in kept])
    return [
        {
            "path": relative(images[r["image_name"]], root),
            "patient_id": r["patient_id"],
            "label": label[r["image_name"]],
            "split": split,
        }
        for r, split in zip(kept, splits, strict=True)
    ]


def main() -> None:
    args = arguments(__doc__.splitlines()[0])
    root: Path = args.root
    name, size, md5 = IMAGES
    fetch_and_extract(
        f"{BASE}/{name}", root / name, root / "images", size, md5, args.delete_archives
    )
    truth_path = download(f"{BASE}/{GROUND_TRUTH[0]}", root / GROUND_TRUTH[0], *GROUND_TRUTH[1:])
    dup_path = download(f"{BASE}/{DUPLICATES[0]}", root / DUPLICATES[0], *DUPLICATES[1:])
    truth = read_csv(truth_path, ["image_name", "patient_id", "target"])
    duplicates = read_duplicates(dup_path, {r["image_name"] for r in truth})
    images = find_files(root / "images", [".jpg"])
    write_manifest(root, manifest_rows(truth, duplicates, images, root))


if __name__ == "__main__":
    main()
