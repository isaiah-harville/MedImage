# Model card: isic2019-convnext

## Summary
A ConvNeXt-S classifier for 8 dermoscopic diagnoses (MEL, NV, BCC, AK, BKL, DF, VASC, SCC),
trained on the ISIC 2019 training data at 384 px. No run has been published yet.

## Intended use
Research only. Not a medical device. It must not be used to diagnose, triage or reassure
patients.

## Data
- ISIC 2019 Challenge training data: 25,331 dermoscopic JPEGs pooled from HAM10000 (Medical
  University of Vienna), BCN_20000 (Hospital Clinic de Barcelona) and MSK. Citations:
  Tschandl P, et al., *Sci Data* 5:180161 (2018); Codella NCF, et al., arXiv:1710.05006;
  Hernández-Pérez C, et al., *Sci Data* 11:641 (2024).
- Downloaded from the ISIC challenge data page (https://challenge.isic-archive.com/data/) by
  `scripts/datasets/isic2019.py`: `ISIC_2019_Training_Input.zip` (9.8 GB),
  `ISIC_2019_Training_GroundTruth.csv` and `ISIC_2019_Training_Metadata.csv`.
- License: CC BY-NC 4.0.
- **Split.** Training data only. ISIC 2019 gives **no patient ID**, only a lesion ID, so the
  manifest's `patient_id` column holds `lesion_id`. Images of one lesion stay in one split, but
  one patient with several lesions can appear in more than one split, so the split is
  lesion-level, not patient-level. Images without a lesion ID are each their own group. Lesions
  are split 70/15/15, stratified by diagnosis, seed 0. Bootstrap intervals resample lesions.
- Labels: the challenge's one-hot ground truth. The UNK column is empty in training.

## Training
Config: `config.yaml`. `convnext_small.fb_in22k_ft_in1k_384` from timm at 384x384 (images are
resized without cropping), AdamW, a cosine schedule, bf16, EMA weights, label smoothing 0.1,
batch 24, 20 epochs. Augmentation: flips, rotation by any angle with 0.9-1.1 scaling, mild
colour jitter. The checkpoint is chosen by validation balanced accuracy, the challenge metric.
Batch size was sized for a 12 GB GPU from an estimate, not measured on one.

## Results
No published run yet.

| Metric | Value | 95% CI |
|---|---|---|

## Limitations and biases
- No published number uses this protocol; see `baselines.yaml`.
- Lesion-level grouping cannot rule out one patient's lesions in both train and test.
- HAM10000 is inside this training set; never test this model on HAM10000.
- The classes are imbalanced (NV dominates), and there is no out-of-distribution class.

## Reproduce
```bash
uv run python scripts/datasets/isic2019.py data/isic2019
uv run opendiagnostics fit -c models/isic2019-convnext/config.yaml
```
