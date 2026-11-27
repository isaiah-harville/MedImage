# Model card: ham10000-convnext

## Summary
A ConvNeXt-S classifier for the 7 HAM10000 diagnoses (akiec, bcc, bkl, df, mel, nv, vasc),
trained on HAM10000 at 384x512 px and tested on the official ISIC 2018 Task 3 test set. No run
has been published yet.

## Intended use
Research only. Not a medical device. It must not be used to diagnose, triage or reassure
patients.

## Data
- HAM10000: 10,015 dermoscopic JPEGs (600x450). Tschandl P, Rosendahl C, Kittler H,
  *Sci Data* 5:180161 (2018), doi:10.1038/sdata.2018.161. Harvard Dataverse
  doi:10.7910/DVN/DBW86T, which also hosts the ISIC 2018 Task 3 test images and ground truth.
- Downloaded by `scripts/datasets/ham10000.py`, MD5-checked against Dataverse's published
  checksums (the two CSVs by size).
- License: CC BY-NC 4.0 (the Dataverse terms of use), metadata CC0.
- **Split.**
    - Train and val: the HAM10000 images, split 85/15 by `lesion_id`, stratified by diagnosis,
      seed 0. HAM10000 gives **no patient ID**, so the manifest's `patient_id` column holds
      `lesion_id`: images of one lesion stay together, but one patient's different lesions can
      fall in both splits.
    - Test: the ISIC 2018 Task 3 test images, which makes test numbers comparable with that
      challenge's leaderboard. They have no lesion ID, so each image is its own group, and
      overlap of lesions between HAM10000 and this test set cannot be checked by ID.
- Labels: `dx`. More than half are confirmed by histopathology; the rest by follow-up, expert
  consensus or confocal microscopy.

## Training
Config: `config.yaml`. `convnext_small.fb_in22k_ft_in1k_384` from timm at 384x512 (the images'
3:4 aspect), AdamW, a cosine schedule, bf16, EMA weights, label smoothing 0.1, batch 16,
30 epochs. Augmentation: flips, rotation by any angle with 0.9-1.1 scaling, mild colour jitter.
The checkpoint is chosen by validation balanced accuracy, the challenge metric. Batch size was
sized for a 12 GB GPU from an estimate, not measured on one.

## Results
No published run yet. Compare test balanced accuracy, with its 95% interval, against
`baselines.yaml`.

| Metric | Value | 95% CI |
|---|---|---|

## Limitations and biases
- About two thirds of the images are nevi; df and vasc have 115 and 142 images.
- The training images come from two sites (Vienna and Queensland); part of the test set comes
  from other sites.
- HAM10000 is also inside the ISIC 2019 training set.

## Reproduce
```bash
uv run python scripts/datasets/ham10000.py data/ham10000
uv run opendiagnostics fit -c models/ham10000-convnext/config.yaml
```
