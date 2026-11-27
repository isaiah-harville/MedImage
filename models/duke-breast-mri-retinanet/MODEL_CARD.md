# Model card: duke-breast-mri-retinanet

## Summary
A MONAI 3D RetinaNet (ResNet-50 backbone, three-level FPN) that finds breast tumors in
DCE-MRI from the pre-contrast and first post-contrast phases. Scored by FROC and CPM on a
patient-level test split. No run has been published yet.

## Intended use
Research only. Not a medical device. It must not be used to find, rule out, or stage
breast cancer in patients.

## Data
- Duke-Breast-Cancer-MRI (Saha et al., British Journal of Cancer 119(4), 2018,
  doi:10.1038/s41416-018-0185-8; data doi:10.7937/TCIA.e3sv-re93), from TCIA without an
  account. The collection is about 368 GB; the script fetches only the two phases used,
  about 151 GB for 917 of 922 patients.
- License CC BY-NC 4.0: non-commercial use, with attribution.
- `scripts/datasets/duke_breast_mri.py` writes `data/duke-breast-mri/manifest.json`. It
  picks the phases from each series description and maps the one box per patient in
  `Annotation_Boxes.xlsx` to world LPS millimetres, reading slice numbers in ascending
  height as the collection page describes.
- Split: random by patient, 70/15/15.

## Training
`config.yaml`: two channels, resampled to 1 x 1 x 1.5 mm and scaled per channel between the
0.5th and 99.5th percentiles; two 160 x 160 x 64 crops per patient; left-right and
head-foot flips (not front-back, since patients lie prone); bf16, 100 epochs. Not yet run
on the target RTX 4070 Ti (12 GB).

## Results
None yet. Report test CPM and the seven sensitivities with 95% patient-level intervals from
the run's `metrics.json`.

## Limitations and biases
One institution, 2000 to 2014, 1.5 T and 3 T scanners. Each patient has one box (the
biopsied or largest biopsied lesion), so other real lesions count as false positives and
FROC understates the detector. The phase choice from series descriptions and the slice
order of the boxes are inferred from metadata and the collection page; check a few
patients visually before trusting a run.

## Reproduce
```bash
uv run scripts/datasets/duke_breast_mri.py
uv run opendiagnostics fit -c models/duke-breast-mri-retinanet/config.yaml
```
