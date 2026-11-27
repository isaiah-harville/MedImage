# Model card: lidc-idri-retinanet

## Summary
A MONAI 3D RetinaNet (ResNet-50 backbone, two-level FPN) that finds lung nodules of 3 mm or
more in all LIDC-IDRI CT scans, with boxes from the readers' outlines. Scored by FROC and
CPM on a patient-level test split. No run has been published yet.

## Intended use
Research only. Not a medical device. It must not be used to find, rule out, or follow
nodules in patients.

## Data
- LIDC-IDRI (Armato et al., Medical Physics 38(2), 2011, doi:10.1118/1.3528204; data
  doi:10.7937/K9/TCIA.2015.LO9QL9SX), from TCIA without an account: 1,018 CT series from
  1,010 patients, about 128 GB.
- License CC BY 3.0, so weights, predictions and example images may be published with
  attribution.
- `scripts/datasets/lidc_idri.py` writes `data/lidc-idri/manifest.json`. Box rule: each
  reader's outline of a nodule of 3 mm or more is a box over its outlined pixels; outlines
  from different readers that overlap are one nodule, boxed by the per-coordinate median;
  a nodule marked by at least `--min-readers` readers (default 1) is a target. Weaker
  nodules, marks under 3 mm, and non-nodules are ignored in scoring.
- Split: random by patient, 70/15/15, stratified on whether a scan has a nodule. The eight
  patients with two scans stay on one side.

## Training
Same recipe as `luna16-retinanet`: 0.703 x 0.703 x 1.25 mm, -1024..300 HU, 192 x 192 x 80
crops, bf16, 100 epochs. Not yet run on the target RTX 4070 Ti (12 GB).

## Results
None yet. Report test CPM and the seven sensitivities with 95% patient-level intervals from
the run's `metrics.json`.

## Limitations and biases
Readers did not have to agree and saw each other's marks in the second read, so labels
are noisy and correlated. The box rule is ours; results under it are not comparable with
LUNA16 or with papers that use other rules. Do not pretrain here and test on LUNA16: the
scans overlap.

## Reproduce
```bash
uv run scripts/datasets/lidc_idri.py
uv run opendiagnostics fit -c models/lidc-idri-retinanet/config.yaml
```
