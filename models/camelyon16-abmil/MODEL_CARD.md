# Model card: camelyon16-abmil

## Summary
Predicts whether a whole-slide image of a sentinel lymph node section contains breast cancer
metastasis (tumor vs normal). Attention-based multiple-instance learning (MONAI `MILModel`,
`mil_mode: att`) over frozen ImageNet ResNet-50 layer-3 features of 256 px tissue tiles at
level 1 (about 20x). No published run yet.

## Intended use
Research only. Not a medical device. Do not use it to diagnose patients or to triage slides.

## Data
- CAMELYON16 (Ehteshami Bejnordi et al., JAMA 2017; Litjens et al., GigaScience 2018).
  399 H&E slides from Radboud UMC and UMC Utrecht, from the AWS open data bucket
  `s3://camelyon-dataset/CAMELYON16/`.
- License CC0. Slides, weights and per-slide predictions may be shared.
- Split: the official one. The 270 training slides (`normal_*`, `tumor_*`) lose 15% to
  validation, stratified by metastasis class and centre; the 129 official test slides
  (`test_*`, labels in `evaluation/reference.csv`) are scored once.
- CAMELYON16 publishes no patient IDs, so each slide is its own patient, intervals resample
  slides, and leakage between slides of one patient cannot be checked.
- Labels are slide-level, from the challenge's pathologist reference standard. The README lists
  20 tumor slides whose annotations are not exhaustive; slide labels are unaffected.

## Training
Two stages, both from `config.yaml`:

1. `uv run opendiagnostics prepare -c models/camelyon16-abmil/config.yaml` finds tissue with an
   Otsu threshold on saturation at the level nearest 32x downsampling, keeps 256 px level-1
   tiles that are at least 25% tissue, and writes one feature file per slide to
   `features_dir`. It resumes where it stopped.
2. `uv run opendiagnostics fit -c models/camelyon16-abmil/config.yaml` trains the attention
   head on the cached features, one slide per step, for 40 epochs, keeping the epoch with the
   best validation AUROC.

Commit, hardware and training time: from the run's `run.json`.

## Results
Test split, best-validation checkpoint, 95% slide-level bootstrap CIs from the run's
`metrics.json`.

| Metric | Value | 95% CI |
|---|---|---|

The decision threshold is Youden's J on validation.

## Limitations and biases
- Two centres and two scanners; other labs, stains and scanners are untested.
- ImageNet features are not pathology features; pathology foundation models usually do
  better but most need a gated account to download.
- Validation holds about 40 slides, so checkpoint selection is noisy.

## Reproduce
```bash
uv run python scripts/datasets/camelyon16.py --out data/camelyon16
uv run opendiagnostics prepare -c models/camelyon16-abmil/config.yaml
uv run opendiagnostics fit -c models/camelyon16-abmil/config.yaml
```
