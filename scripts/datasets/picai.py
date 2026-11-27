"""Download PI-CAI (biparametric prostate MRI, csPCa lesions) and write its manifest.

    uv run --with SimpleITK python scripts/datasets/picai.py data/picai [--delete-archives]

The 1,500 public studies come as five zips on Zenodo (record 6624726, 26.9 GB), each
checked against its MD5. The lesion masks and marksheet come from the official
``picai_labels`` repository at a pinned commit: the human-expert masks for 1,295 studies,
resampled to the T2W grid, and the expert masks of Pooch et al. (2025) for the other 205.

The MHA files are converted to NIfTI with SimpleITK, which only this script needs (hence
``--with SimpleITK``). ADC and high-b-value DWI are resampled onto the T2W grid, because the
sequences are not co-registered, and so is the mask (nearest neighbour). Masks grade
lesions by ISUP; every value above 0 is a clinically significant lesion, so the masks are
made binary.

The hidden test set's labels are not public, so the studies are split by patient (some
patients have two studies) 70/10/20, stratified by the marksheet's ``case_csPCa``.

License: CC BY-NC 4.0 (https://zenodo.org/records/6624726). Cite Saha et al. 2024 and,
for the 205 new masks, Pooch et al. 2025.
"""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
from typing import Any

from _download import arguments, assign_splits, download, extract, write_manifest

RECORD = "https://zenodo.org/api/records/6624726/files"
FOLDS = {
    0: (5_368_893_021, "b0cdd8f5fb7733ca585dca3541586a34"),
    1: (5_164_062_415, "d7c8cd0efc0e7f264df8d5acf860662f"),
    2: (5_270_275_720, "5ac0caf27cfe94dfc04bb408a971b351"),
    3: (5_502_916_242, "6602bce3d0882d7b9c2153d880c97e96"),
    4: (5_562_839_222, "8941720d8a8ee66571691cfb10dbee66"),
}
LABELS_COMMIT = "ce4a4723d7c46d882a6cbaacb40ed6c4be86282f"
LABELS_URL = f"https://codeload.github.com/DIAGNijmegen/picai_labels/tar.gz/{LABELS_COMMIT}"
SEQUENCES = ("t2w", "adc", "hbv")
COLUMNS = ("id", *(f"image_{s}" for s in SEQUENCES), "label", "patient_id", "case_csPCa", "split")
MASKS = (
    "csPCa_lesion_delineations/human_expert/resampled",
    "csPCa_lesion_delineations/human_expert/Pooch25",
)


def simpleitk() -> Any:
    try:
        return importlib.import_module("SimpleITK")
    except ModuleNotFoundError:
        raise SystemExit(
            "PI-CAI ships MHA files, which this script converts with SimpleITK. Run it as "
            "`uv run --with SimpleITK python scripts/datasets/picai.py`."
        ) from None


def convert(sitk: Any, study: str, images: dict[str, Path], mask: Path, out: Path) -> None:
    """Write the three sequences and a binary mask, all on the T2W grid, as NIfTI."""
    if (out / f"{study}_label.nii.gz").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    t2w = sitk.ReadImage(str(images["t2w"]))
    for sequence in SEQUENCES:
        image = t2w if sequence == "t2w" else sitk.ReadImage(str(images[sequence]))
        if sequence != "t2w":
            image = sitk.Resample(image, t2w, sitk.Transform(), sitk.sitkLinear, 0.0)
        sitk.WriteImage(sitk.Cast(image, sitk.sitkFloat32), str(out / f"{study}_{sequence}.nii.gz"))
    label = sitk.Resample(
        sitk.ReadImage(str(mask)), t2w, sitk.Transform(), sitk.sitkNearestNeighbor, 0
    )
    label = sitk.Cast(label > 0, sitk.sitkUInt8)
    partial = out / f"{study}_label.part.nii.gz"
    sitk.WriteImage(label, str(partial))
    partial.replace(out / f"{study}_label.nii.gz")


def main() -> None:
    args = arguments(__doc__.split("\n\n")[0])
    root, sitk = args.root, simpleitk()
    archive = root / f"picai_labels-{LABELS_COMMIT[:12]}.tar.gz"
    if not (root / "labels" / f".{archive.name}.extracted").exists():
        download(LABELS_URL, archive)
    labels = extract(
        archive,
        root / "labels",
        args.delete_archives,
        members=lambda name: (
            name if any(m in name for m in (*MASKS, "clinical_information/marksheet.csv")) else None
        ),
    )
    for fold, (size, md5) in FOLDS.items():
        archive = root / f"picai_public_images_fold{fold}.zip"
        if not (root / "images" / f".{archive.name}.extracted").exists():
            download(f"{RECORD}/{archive.name}/content", archive, size=size, md5=md5)
        extract(archive, root / "images", args.delete_archives)
    found = {p.name: p for p in (root / "images").rglob("*.mha")}
    masks = {p.name: p for m in MASKS for p in labels.rglob(f"{m}/*.nii.gz")}
    (marksheet,) = labels.rglob("marksheet.csv")
    with marksheet.open(newline="") as handle:
        studies = list(csv.DictReader(handle))
    rows = []
    for entry in studies:
        study = f"{entry['patient_id']}_{entry['study_id']}"
        wanted = [f"{study}_{s}.mha" for s in SEQUENCES]
        missing = [n for n in [*wanted, f"{study}.nii.gz"] if n not in found and n not in masks]
        if missing:
            raise RuntimeError(f"study {study} lacks {missing}; check the archives and labels")
        images = {s: found[n] for s, n in zip(SEQUENCES, wanted, strict=True)}
        mask = masks[f"{study}.nii.gz"]
        convert(sitk, study, images, mask, root / "nifti" / entry["patient_id"])
        rows.append(
            {
                "id": study,
                **{
                    f"image_{s}": f"nifti/{entry['patient_id']}/{study}_{s}.nii.gz"
                    for s in SEQUENCES
                },
                "label": f"nifti/{entry['patient_id']}/{study}_label.nii.gz",
                "patient_id": entry["patient_id"],
                "case_csPCa": entry["case_csPCa"],
                "split": "",
            }
        )
    splits = assign_splits(
        [r["patient_id"] for r in rows], [r["case_csPCa"] for r in rows], (0.7, 0.1, 0.2)
    )
    for row, split in zip(rows, splits, strict=True):
        row["split"] = split
    write_manifest(root, rows, COLUMNS)


if __name__ == "__main__":
    main()
