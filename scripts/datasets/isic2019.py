"""Download the ISIC 2019 training data and write a lesion-level manifest.

    uv run python scripts/datasets/isic2019.py data/isic2019 [--delete-archives]

Source: the ISIC challenge data page, https://challenge.isic-archive.com/data/ (CC BY-NC 4.0).
Only the 25,331 training images are used, with their 8 classes. ISIC 2019 publishes no patient
ID, so the manifest's ``patient_id`` column holds the metadata's ``lesion_id``: images of one
lesion stay in one split. An image without a lesion ID is its own group. Lesions are split
into train, val and test (70/15/15, stratified by diagnosis).

The image archive is 9.8 GB. Sizes and MD5s are from the S3 objects' headers (for a
single-part upload the ETag is the MD5).
"""

from __future__ import annotations

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

BASE = "https://isic-archive.s3.amazonaws.com/challenges/2019"
IMAGES = ("ISIC_2019_Training_Input.zip", 9_771_618_190, None)
GROUND_TRUTH = ("ISIC_2019_Training_GroundTruth.csv", 1_291_479, "2c02bdcc6e7f36d355f4f86b210595ae")
METADATA = ("ISIC_2019_Training_Metadata.csv", 1_214_351, "8ca5fdef200ffd8579f48c823b504f7e")
CLASSES = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")


def diagnosis(row: dict[str, str]) -> str:
    labels = [c for c in CLASSES if float(row[c]) == 1.0]
    if len(labels) != 1:
        raise RuntimeError(f"{row['image']} has {len(labels)} training diagnoses, expected one")
    return labels[0]


def manifest_rows(
    truth: list[dict[str, str]],
    metadata: list[dict[str, str]],
    images: dict[str, Path],
    root: Path,
) -> list[dict[str, str]]:
    lesion = {r["image"]: r["lesion_id"].strip() for r in metadata}
    groups = [lesion.get(r["image"]) or f"image:{r['image']}" for r in truth]
    labels = [diagnosis(r) for r in truth]
    missing = [r["image"] for r in truth if r["image"] not in images]
    if missing:
        raise RuntimeError(
            f"{len(missing)} images in the ground truth are not on disk: {missing[:5]}"
        )
    splits = assign_splits(groups, labels)
    return [
        {
            "path": relative(images[r["image"]], root),
            "patient_id": group,
            "label": label,
            "split": split,
        }
        for r, group, label, split in zip(truth, groups, labels, splits, strict=True)
    ]


def main() -> None:
    args = arguments(__doc__.splitlines()[0])
    root: Path = args.root
    name, size, md5 = IMAGES
    fetch_and_extract(
        f"{BASE}/{name}", root / name, root / "images", size, md5, args.delete_archives
    )
    truth_path = download(f"{BASE}/{GROUND_TRUTH[0]}", root / GROUND_TRUTH[0], *GROUND_TRUTH[1:])
    meta_path = download(f"{BASE}/{METADATA[0]}", root / METADATA[0], *METADATA[1:])
    truth = read_csv(truth_path, ["image", *CLASSES])
    metadata = read_csv(meta_path, ["image", "lesion_id"])
    images = find_files(root / "images", [".jpg"])
    write_manifest(root, manifest_rows(truth, metadata, images, root))


if __name__ == "__main__":
    main()
