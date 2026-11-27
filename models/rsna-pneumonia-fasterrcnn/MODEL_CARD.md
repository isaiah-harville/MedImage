# Model card: rsna-pneumonia-fasterrcnn

## Summary
A torchvision Faster R-CNN (ResNet-50 FPN v2, ImageNet backbone) that draws boxes around
pneumonia-like lung opacity on frontal chest radiographs. No run is published yet.

## Intended use
Research only. Not a medical device. It must not be used to diagnose or triage patients.

## Data
- RSNA Pneumonia Detection Challenge (2018), the adjudicated images and annotations used in
  the challenge, downloaded from RSNA's open links by `scripts/datasets/rsna_pneumonia.py`.
  Shih et al., Radiology: Artificial Intelligence 1(1):e180041, 2019.
- License: RSNA's Pneumonia Detection Challenge Terms of Use and Attribution (academic,
  commercial and non-commercial use with attribution; no re-identification). It is not a
  named open license, so `publish` uploads no checkpoint and copies no example images.
- Split: patient-level, 70/15/15 of NIH patients, stratified by whether an image has a box.
  The RSNA images are a subset of NIH ChestX-ray14, and one NIH patient can have several
  images, so the patient is the NIH patient ID from RSNA's mapping file. The script prints the
  annotation sets it finds; whether the open release holds Kaggle's stage 2 test labels is not
  verified, and the patient split does not rely on it.
- Labels: boxes drawn by radiologists and adjudicated by RSNA. Images with no box are
  negatives. Exact duplicate boxes are merged.

## Training
Config: `config.yaml`. 1024 px (the native size), batch 4, 12 epochs of SGD with cosine decay,
horizontal flips. Sized for a 12 GB GPU; peak memory has not been measured.

## Results
Headline metric: test mAP@0.5 with a 95% patient-level bootstrap interval. Not comparable
with the Kaggle leaderboard; see `baselines.yaml`.

| Metric | Value | 95% CI |
|---|---|---|

## Limitations and biases
- The ImageNet backbone was not pretrained on NIH images, but do not swap in one that was:
  it would have seen these test images.
- Boxes mark opacity consistent with pneumonia, not confirmed infection.
- NIH images come from one hospital; site and scanner shift are untested.

## Reproduce
```bash
uv run python scripts/datasets/rsna_pneumonia.py --delete-archives
uv run opendiagnostics fit -c models/rsna-pneumonia-fasterrcnn/config.yaml
```
