# Model card: nih-chestxray14-convnext

## Summary
ConvNeXt-S, pre-trained on ImageNet-22k, fine-tuned at 448 px to predict 14 findings from a
frontal chest X-ray. No run is published yet; the headline metric will be mean AUROC over
the 14 findings on NIH's official test list, with a patient-level 95% bootstrap interval.

## Intended use
Research only. Not a medical device. It must not be used to diagnose, triage or rule out
disease in any patient.

## Data
- NIH ChestX-ray14 (ChestX-ray8 expanded), from the NIH Clinical Center:
  https://nihcc.app.box.com/v/ChestXray-NIHCC. 112,120 frontal PNGs at 1024 x 1024 from
  30,805 patients. Cite Wang et al., CVPR 2017 (arXiv:1705.02315).
- Terms: NIH places no restrictions on use, and asks for a link to the download site, the
  CVPR 2017 citation, and an acknowledgement of the NIH Clinical Center. NIH names no license,
  so `opendiagnostics publish` treats it as not open and shares neither example images nor
  weights.
- `scripts/datasets/nih_chestxray14.py` downloads the archives and writes `manifest.csv`.
  Patient IDs come from `Patient ID` in `Data_Entry_2017_v2020.csv`.
- Split: NIH's patient-level `test_list.txt` (25,596 images) is the test split. Validation is
  fold 0 of 8 patient-grouped folds of `train_val_list.txt` (86,524 images). Per-split counts
  are in each run's `splits.json`.
- Labels are text-mined from radiology reports by NLP, with expected accuracy above 90%.
  "No Finding" is the all-zero target.

## Training
Config: `config.yaml`. AdamW, cosine schedule over 10 epochs, bf16 mixed precision, EMA weights,
batches of 16 with four accumulated per optimizer step. Augmentation is a random resized crop
and small affine jitter, with no flips. Commit, hardware and training time are in each run's
`run.json`.

## Results
No published run yet.

| Metric | Value | 95% CI |
|---|---|---|

## Limitations and biases
- One hospital system's images, so expect site and scanner shift elsewhere.
- NLP labels are noisy, and the reports were never released.
- Many patients have many follow-up images; the bootstrap resamples patients for that reason.
- Published baselines pre-trained on chest X-ray collections that include this dataset's
  training images; this model starts from ImageNet only.

## Reproduce
```bash
uv run python scripts/datasets/nih_chestxray14.py data/nih_chestxray14
uv run opendiagnostics fit -c models/nih-chestxray14-convnext/config.yaml
```
