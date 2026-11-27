# Model card: isic2020-convnext

## Summary
A ConvNeXt-S classifier that scores a dermoscopic image for melanoma (binary: benign vs
melanoma), trained on the ISIC 2020 training data at 384 px. No run has been published yet.

## Intended use
Research only. Not a medical device. It must not be used to diagnose, triage or reassure
patients.

## Data
- ISIC 2020 Challenge training data (SIIM-ISIC Melanoma Classification): 33,126 dermoscopic
  JPEGs from 2,056 patients, with melanoma in about 1.8% of images. Rotemberg V, et al.,
  *Sci Data* 8:34 (2021), doi:10.1038/s41597-021-00815-z. Dataset DOI 10.34970/2020-ds01.
- Downloaded from the ISIC challenge data page (https://challenge.isic-archive.com/data/) by
  `scripts/datasets/isic2020.py`: `ISIC_2020_Training_JPEG.zip` (24.7 GB),
  `ISIC_2020_Training_GroundTruth.csv` and `ISIC_2020_Training_Duplicates.csv`.
- License: CC BY-NC 4.0. Weights, metrics, predictions and a few example images can be
  published for non-commercial use with attribution.
- **Split.** The official test labels were never released, so there is no comparable published
  test number. The script splits the training data by `patient_id` into train, val and test
  (70/15/15, stratified by each patient's majority label, seed 0). Images on ISIC's duplicate
  list are removed first: one copy is kept when both copies share a label, neither when they
  disagree. Per-split counts are in each run's `splits.json`.
- Labels: the challenge's `target` (malignant means melanoma), from histopathology, expert
  agreement or follow-up as recorded in the source metadata.

## Training
Config: `config.yaml`. `convnext_small.fb_in22k_ft_in1k_384` from timm, fine-tuned end to end at
384x384 with AdamW, a cosine schedule, bf16 mixed precision and EMA weights, batch 24,
15 epochs. Augmentation: horizontal and vertical flips, rotation by any angle with 0.9-1.1
scaling, and mild colour jitter; dermoscopic images have no canonical orientation, so these
keep the label. The checkpoint is chosen by validation AUROC; the decision threshold is
Youden's J on validation. Batch size was sized for a 12 GB GPU from an estimate of saved
activations, not measured on one.

## Results
No published run yet. Results go here from the run's `metrics.json`, with 95% patient-level
bootstrap intervals.

| Metric | Value | 95% CI |
|---|---|---|

## Limitations and biases
- Our test split comes from the same sources as training, so it says nothing about other
  clinics or cameras.
- Melanoma is rare (about 1.8%), so AUPRC and calibration matter as much as AUROC.
- The images come from centres in Spain, Austria, the USA, Australia and Greece; other populations are not represented.

## Reproduce
```bash
uv run python scripts/datasets/isic2020.py data/isic2020
uv run opendiagnostics fit -c models/isic2020-convnext/config.yaml
```
