# Model card: cmmd-convnext

## Summary
A ConvNeXt-S classifier that scores a single mammogram view for a malignant vs benign lesion,
trained on CMMD at 1024x832 px. No run has been published yet.

## Intended use
Research only. Not a medical device. It must not be used for screening or diagnosis. Every
CMMD breast has a biopsied lesion, so the model has never seen a normal screen.

## Data
- CMMD (The Chinese Mammography Database): 1,775 patients, 5,202 DICOM images in TCIA's
  listing. Cui C, et al., The Cancer Imaging Archive (2021), doi:10.7937/tcia.eqde-4b16.
- Downloaded from TCIA's public NBIA API, without an account, by `scripts/datasets/cmmd.py`,
  one zip per series; each series' image count is checked against TCIA's listing.
- License: CC BY 4.0.
- Labels: `classification` (Benign or Malignant) from `CMMD_clinicaldata_revision.xlsx`, one
  row per breast, confirmed by biopsy. Each image is matched to its breast by patient ID and the
  DICOM image laterality. Images of a breast with no row are left out; the run's `splits.json`
  records how many images remain.
- **Split.** Patients split 70/15/15, stratified by majority label, seed 0. Subset D2 is all
  malignant, so splitting by subset would leak the label; we do not.

## Training
Config: `config.yaml`. `convnext_small.fb_in22k_ft_in1k` from timm with one input channel, at
1024x832 (close to the images' aspect), AdamW, a cosine schedule, bf16, EMA weights, batch 4
with 8-step gradient accumulation (effective batch 32), 30 epochs. Augmentation: horizontal flip
(left and right breasts are mirror images, so it keeps the label) and mild brightness and
contrast jitter. No crops, since a crop can cut out the lesion; no vertical flips or rotations,
since those produce views a mammogram never shows. The checkpoint is chosen by validation AUROC;
the threshold is Youden's J on validation. Batch size was sized for a 12 GB GPU from an
estimate, not measured on one.

## Results
No published run yet.

| Metric | Value | 95% CI |
|---|---|---|

## Limitations and biases
- One site's population (Chinese women) and one vendor (GE Senographe DS).
- Each image is scored alone; the model does not combine CC and MLO views.
- Downsampling to 1024x832 can lose calcifications.

## Reproduce
```bash
uv run python scripts/datasets/cmmd.py data/cmmd
uv run opendiagnostics fit -c models/cmmd-convnext/config.yaml
```
