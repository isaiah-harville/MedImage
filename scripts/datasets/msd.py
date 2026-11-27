"""Download Medical Segmentation Decathlon tasks and write one manifest per task.

    uv run python scripts/datasets/msd.py data/msd [--task Task09_Spleen ...]
        [--delete-archives]

Each task is one tar from the AWS mirror that MONAI's ``DecathlonDataset`` uses, checked
against the MD5 MONAI records for it. Sizes: BrainTumour 7.6 GB, Heart 0.46 GB, Liver
28.9 GB, Hippocampus 0.03 GB, Prostate 0.24 GB, Lung 9.2 GB, Pancreas 12.3 GB,
HepaticVessel 9.3 GB, Spleen 1.6 GB, Colon 6.2 GB (85.6 GB in all).

The test images' labels were never released, so each task's labelled training cases are
split 70/10/20 into train, validation and test. MSD publishes no patient IDs, so the case
ID stands in for the patient and leakage between cases of one patient cannot be checked.

License: CC BY-SA 4.0 (http://medicaldecathlon.com/).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _download import assign_splits, download, extract, write_manifest
from monai.apps import DecathlonDataset

TASKS = sorted(DecathlonDataset.resource)
COLUMNS = ("id", "image", "label", "patient_id", "split")


def prepare(task: str, root: Path, delete: bool) -> None:
    archive = root / f"{task}.tar"
    if not (root / f".{archive.name}.extracted").exists():
        download(DecathlonDataset.resource[task], archive, md5=DecathlonDataset.md5[task])
    folder = extract(
        archive,
        root,
        delete,
        members=lambda name: None if name.rsplit("/", 1)[-1].startswith("._") else name,
    )
    task_dir = folder / task
    listing = json.loads((task_dir / "dataset.json").read_text())
    rows = []
    for entry in listing["training"]:
        case = entry["label"].rsplit("/", 1)[-1].removesuffix(".nii.gz")
        rows.append(
            {
                "id": case,
                "image": entry["image"].removeprefix("./"),
                "label": entry["label"].removeprefix("./"),
                "patient_id": case,
                "split": "",
            }
        )
    splits = assign_splits([r["id"] for r in rows], ["all"] * len(rows), (0.7, 0.1, 0.2))
    for row, split in zip(rows, splits, strict=True):
        row["split"] = split
    write_manifest(task_dir, rows, COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="dataset directory, for example data/msd")
    parser.add_argument("--task", action="append", choices=TASKS, help="task to fetch (repeatable)")
    parser.add_argument(
        "--delete-archives", action="store_true", help="delete each archive once it is extracted"
    )
    options = parser.parse_args()
    for task in options.task or TASKS:
        prepare(task, options.root, options.delete_archives)


if __name__ == "__main__":
    main()
