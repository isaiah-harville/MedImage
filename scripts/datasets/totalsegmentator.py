"""Download TotalSegmentator v2.0.1 (117 CT structures) and write its manifest.

    uv run python scripts/datasets/totalsegmentator.py data/totalsegmentator [--delete-archives]

The dataset is one 23.6 GB zip on Zenodo (record 10047292), checked against its MD5. Each
case ships one binary mask per structure; this script merges them into one label map per
case (``label.nii.gz``, values 1..117 in the order of ``CLASSES``, the order of the
official ``total`` task). Where two masks overlap, the later class wins and the script
prints how many voxels that touched.

The split is the dataset's own (``meta.csv``: train, val, test). The dataset publishes no
patient IDs, so the case ID stands in for the patient.

License: CC BY 4.0 (https://zenodo.org/records/10047292).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from _download import arguments, download, extract, write_manifest

URL = "https://zenodo.org/api/records/10047292/files/Totalsegmentator_dataset_v201.zip/content"
SIZE = 23_581_218_285
MD5 = "fe250e5718e0a3b5df4c4ea9d58a62fe"
COLUMNS = ("id", "image", "label", "patient_id", "split")

# totalsegmentator/map_to_binary.py, class_map["total"], in label order 1..117.
CLASSES = (
    "spleen kidney_right kidney_left gallbladder liver stomach pancreas adrenal_gland_right "
    "adrenal_gland_left lung_upper_lobe_left lung_lower_lobe_left lung_upper_lobe_right "
    "lung_middle_lobe_right lung_lower_lobe_right esophagus trachea thyroid_gland small_bowel "
    "duodenum colon urinary_bladder prostate kidney_cyst_left kidney_cyst_right sacrum "
    "vertebrae_S1 vertebrae_L5 vertebrae_L4 vertebrae_L3 vertebrae_L2 vertebrae_L1 "
    "vertebrae_T12 vertebrae_T11 vertebrae_T10 vertebrae_T9 vertebrae_T8 vertebrae_T7 "
    "vertebrae_T6 vertebrae_T5 vertebrae_T4 vertebrae_T3 vertebrae_T2 vertebrae_T1 "
    "vertebrae_C7 vertebrae_C6 vertebrae_C5 vertebrae_C4 vertebrae_C3 vertebrae_C2 "
    "vertebrae_C1 heart aorta pulmonary_vein brachiocephalic_trunk subclavian_artery_right "
    "subclavian_artery_left common_carotid_artery_right common_carotid_artery_left "
    "brachiocephalic_vein_left brachiocephalic_vein_right atrial_appendage_left "
    "superior_vena_cava inferior_vena_cava portal_vein_and_splenic_vein iliac_artery_left "
    "iliac_artery_right iliac_vena_left iliac_vena_right humerus_left humerus_right "
    "scapula_left scapula_right clavicula_left clavicula_right femur_left femur_right "
    "hip_left hip_right spinal_cord gluteus_maximus_left gluteus_maximus_right "
    "gluteus_medius_left gluteus_medius_right gluteus_minimus_left gluteus_minimus_right "
    "autochthon_left autochthon_right iliopsoas_left iliopsoas_right brain skull "
    + " ".join(f"rib_left_{k}" for k in range(1, 13))
    + " "
    + " ".join(f"rib_right_{k}" for k in range(1, 13))
    + " sternum costal_cartilages"
).split()


def merge(case: Path) -> int:
    """Write ``case/label.nii.gz`` from its per-structure masks; return overlapping voxels."""
    target = case / "label.nii.gz"
    if target.exists():
        return 0
    ct = nib.Nifti1Image.load(case / "ct.nii.gz")
    label = np.zeros(ct.shape[:3], np.uint8)
    overlap = 0
    for value, name in enumerate(CLASSES, start=1):
        mask = (
            np.asarray(nib.Nifti1Image.load(case / "segmentations" / f"{name}.nii.gz").dataobj) > 0
        )
        if mask.shape != label.shape:
            raise RuntimeError(f"{case.name}/{name} is {mask.shape}, the CT is {label.shape}")
        overlap += int(np.count_nonzero(mask & (label > 0)))
        label[mask] = value
    partial = case / "label.part.nii.gz"
    nib.save(nib.Nifti1Image(label, ct.affine), partial)
    partial.replace(target)
    return overlap


def main() -> None:
    args = arguments(__doc__.split("\n\n")[0])
    root = args.root
    archive = root / "Totalsegmentator_dataset_v201.zip"
    if not (root / "v201" / f".{archive.name}.extracted").exists():
        download(URL, archive, size=SIZE, md5=MD5)
    folder = extract(archive, root / "v201", args.delete_archives)
    (meta,) = folder.rglob("meta.csv")
    with meta.open(newline="") as handle:
        dialect = csv.Sniffer().sniff(handle.read(4096), delimiters=",;")
        handle.seek(0)
        cases = list(csv.DictReader(handle, dialect=dialect))
    rows = []
    for entry in cases:
        case = meta.parent / entry["image_id"]
        overlap = merge(case)
        if overlap:
            print(f"{case.name}: {overlap} voxels in two masks", file=sys.stderr)
        rows.append(
            {
                "id": case.name,
                "image": f"{case.relative_to(root).as_posix()}/ct.nii.gz",
                "label": f"{case.relative_to(root).as_posix()}/label.nii.gz",
                "patient_id": case.name,
                "split": entry["split"],
            }
        )
    write_manifest(root, rows, COLUMNS)


if __name__ == "__main__":
    main()
