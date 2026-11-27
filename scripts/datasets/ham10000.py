"""Download HAM10000 and the ISIC 2018 Task 3 test set, and write a lesion-level manifest.

    uv run python scripts/datasets/ham10000.py data/ham10000 [--delete-archives]

Source: Harvard Dataverse, doi:10.7910/DVN/DBW86T (CC BY-NC 4.0), which also hosts the
ISIC 2018 Task 3 test images and their ground truth. Dataverse publishes each file's size and
MD5; the two CSVs are fetched in their original format, whose MD5 Dataverse does not list, so
only their size is checked.

The 10,015 HAM10000 images become train and val (85/15). HAM10000 publishes no patient ID, so
the manifest's ``patient_id`` column holds ``lesion_id``: images of one lesion stay in one
split, stratified by diagnosis. The ISIC 2018 Task 3 test images are the test split, which
makes test metrics comparable with that challenge's leaderboard. They carry no lesion ID, so
each is its own group.
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

ACCESS = "https://dataverse.harvard.edu/api/access/datafile"
# name: (Dataverse file id, size in bytes, MD5 or None)
ARCHIVES = {
    "HAM10000_images_part_1.zip": (3172585, 1_366_522_108, "4639bfa73ab251610530a97c898e6e46"),
    "HAM10000_images_part_2.zip": (3172584, 1_403_566_547, "da43d6cc50f6613013be07e8986b384b"),
    "ISIC2018_Task3_Test_Images.zip": (3855824, 421_046_860, "0488b6f65849e310857d9207daa55a21"),
}
METADATA = ("HAM10000_metadata.csv", 4338392, 690_218)
TEST_TRUTH = ("ISIC2018_Task3_Test_GroundTruth.csv", 6924466, 108_383)
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


def manifest_rows(
    metadata: list[dict[str, str]],
    test_truth: list[dict[str, str]],
    images: dict[str, Path],
    root: Path,
) -> list[dict[str, str]]:
    names = [r["image_id"] for r in [*metadata, *test_truth]]
    missing = [n for n in names if n not in images]
    if missing:
        raise RuntimeError(f"{len(missing)} labelled images are not on disk: {missing[:5]}")
    unknown = {r["dx"] for r in [*metadata, *test_truth]} - set(CLASSES)
    if unknown:
        raise RuntimeError(f"the label files have unexpected diagnoses {sorted(unknown)}")
    splits = assign_splits(
        [r["lesion_id"] for r in metadata], [r["dx"] for r in metadata], (0.85, 0.15, 0.0)
    )
    rows = [
        {
            "path": relative(images[r["image_id"]], root),
            "patient_id": r["lesion_id"],
            "label": r["dx"],
            "split": split,
        }
        for r, split in zip(metadata, splits, strict=True)
    ]
    rows += [
        {
            "path": relative(images[r["image_id"]], root),
            "patient_id": r["lesion_id"],
            "label": r["dx"],
            "split": "test",
        }
        for r in test_truth
    ]
    return rows


def main() -> None:
    args = arguments(__doc__.splitlines()[0])
    root: Path = args.root
    for name, (file_id, size, md5) in ARCHIVES.items():
        fetch_and_extract(
            f"{ACCESS}/{file_id}", root / name, root / "images", size, md5, args.delete_archives
        )
    tables = {}
    for name, file_id, size in (METADATA, TEST_TRUTH):
        tables[name] = download(f"{ACCESS}/{file_id}?format=original", root / name, size)
    metadata = read_csv(tables[METADATA[0]], ["lesion_id", "image_id", "dx"])
    # Dataverse serves the ISIC 2018 test labels in the metadata layout, one row per image
    # with its diagnosis in dx and a HAMTEST_ lesion ID, not as the challenge's one-hot CSV.
    test_truth = read_csv(tables[TEST_TRUTH[0]], ["lesion_id", "image_id", "dx"])
    images = find_files(root / "images", [".jpg"])
    write_manifest(root, manifest_rows(metadata, test_truth, images, root))


if __name__ == "__main__":
    main()
