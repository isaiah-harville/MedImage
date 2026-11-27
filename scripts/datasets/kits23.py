"""Download KiTS23 (kidney, tumor and cyst in CT) and write its manifest.

    uv run python scripts/datasets/kits23.py data/kits23 [--delete-archives]

The 489 training CTs come from the Hugging Face dataset the official ``kits23`` package
downloads from (about 42 GB, each file checked against its SHA-256 in the repository's
file listing). The masks come from the official GitHub repository at a pinned commit. The
110 test cases' masks are not public, so the 489 labelled cases are split by patient
(one case per patient) into 70% train, 10% validation and 20% test.

License: CC BY-NC-SA 4.0 (https://kits-challenge.org/kits23/).
"""

from __future__ import annotations

import shutil

from _download import arguments, assign_splits, download, extract, write_manifest
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

IMAGES_REPO = "neheller/KiTS-Challenge-Imaging"
IMAGES_REVISION = "65f1f295873a326230153c7e1de0c7dba10f0b29"
LABELS_COMMIT = "c1088353084c17b8882a11db71429e7c022b7785"
LABELS_URL = f"https://codeload.github.com/neheller/kits23/tar.gz/{LABELS_COMMIT}"
TRAINING_CASES = [*range(300), *range(400, 589)]
COLUMNS = ("id", "image", "label", "patient_id", "split")


def main() -> None:
    args = arguments(__doc__.split("\n\n")[0])
    root = args.root
    files = {
        f.path: f
        for f in HfApi().list_repo_tree(
            IMAGES_REPO, path_in_repo="images", repo_type="dataset", revision=IMAGES_REVISION
        )
        if isinstance(f, RepoFile)
    }
    archive = root / f"kits23-{LABELS_COMMIT[:12]}.tar.gz"
    if not (root / "repo" / f".{archive.name}.extracted").exists():
        download(LABELS_URL, archive)
    extract(
        archive,
        root / "repo",
        args.delete_archives,
        keep=lambda name: name.endswith("/segmentation.nii.gz"),
    )
    (labels,) = (root / "repo").glob("kits23-*/dataset")
    rows = []
    for number in TRAINING_CASES:
        case = f"case_{number:05d}"
        info = files[f"images/{case}.nii.gz"]
        download(
            f"https://huggingface.co/datasets/{IMAGES_REPO}/resolve/{IMAGES_REVISION}/images/{case}.nii.gz",
            root / "images" / f"{case}.nii.gz",
            size=info.size,
            sha256=info.lfs.sha256 if info.lfs else None,
        )
        label = root / "labels" / f"{case}.nii.gz"
        if not label.exists():
            label.parent.mkdir(exist_ok=True)
            shutil.copyfile(labels / case / "segmentation.nii.gz", label)
        rows.append(
            {
                "id": case,
                "image": f"images/{case}.nii.gz",
                "label": f"labels/{case}.nii.gz",
                "patient_id": case,
                "split": "",
            }
        )
    splits = assign_splits([r["id"] for r in rows], ["all"] * len(rows), (0.7, 0.1, 0.2))
    for row, split in zip(rows, splits, strict=True):
        row["split"] = split
    write_manifest(root, rows, COLUMNS)


if __name__ == "__main__":
    main()
