# Model card: luna16-retinanet

## Summary
A MONAI 3D RetinaNet (ResNet-50 backbone, two-level FPN) that finds lung nodules in chest
CT and outputs boxes with scores. Scored by FROC and the LUNA16 CPM over the ten official
folds. No run has been published yet.

## Intended use
Research only. Not a medical device. It must not be used to find, rule out, or follow
nodules in patients.

## Data
- LUNA16 (Setio et al., Medical Image Analysis 42, 2017, doi:10.1016/j.media.2017.06.015),
  from Zenodo records 3723295 and 4121926: 888 CT scans from LIDC-IDRI, 1,186 nodules of
  3 mm or more accepted by at least three of four radiologists.
- License CC BY 4.0, so weights, predictions and example images may be published with
  attribution.
- `scripts/datasets/luna16.py` downloads about 66 GB and writes `data/luna16/manifest.json`.
  Each nodule is a cube whose side is its diameter, in world LPS millimetres;
  `annotations_excluded.csv` findings are ignored in scoring.
- Split: the official subsets. Run `k` tests on subset `k`, selects the checkpoint on
  subset `k + 1` (mod 10), and trains on the other eight. Patient IDs come from TCIA's
  LIDC-IDRI series list; `setup()` refuses a fold split that puts one patient on both sides.

## Training
`config.yaml`: scans resampled to 0.703 x 0.703 x 1.25 mm and windowed to -1024..300 HU;
two 192 x 192 x 80 crops per scan, half centered on a nodule, two scans per optimizer step
(gradient accumulation); flips, 90 degree rotations and intensity jitter; SGD with Nesterov
momentum and a cosine schedule over 100 epochs in bf16; validation every 5 epochs with a
256 x 256 x 128 sliding window. The recipe follows MONAI's LUNA16 detection tutorial.
Not yet run on the target RTX 4070 Ti (12 GB): if it runs out of memory, lower
`crops_per_volume` to 1 and raise `accumulate_grad_batches` to 4.

## Results
None yet. Report the pooled CPM over the ten test folds
(`uv run opendiagnostics froc runs/luna16-retinanet/*/predictions_test.csv`), with its 95%
patient-level bootstrap interval, and the seven sensitivities.

## Limitations and biases
LIDC-IDRI comes from seven US academic sites with 2000s-era scanners. Nodules below 3 mm
and those fewer than three readers accepted are ignored, not scored. There is no
false-positive reduction stage. Folds are fixed, so the ten fold models are not one model.

## Reproduce
```bash
uv run scripts/datasets/luna16.py
for k in 0 1 2 3 4 5 6 7 8 9; do
  uv run opendiagnostics fit -c models/luna16-retinanet/config.yaml --data.init_args.test_fold=$k
done
uv run opendiagnostics froc runs/luna16-retinanet/*/predictions_test.csv
```
