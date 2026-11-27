# Model card: camelyon17-abmil

## Summary
Classifies a whole-slide image of a sentinel lymph node section by the largest metastasis it
contains: negative, isolated tumour cells (ITC), micro-metastasis or macro-metastasis.
Attention-based multiple-instance learning (MONAI `MILModel`, `mil_mode: att`) over frozen
ImageNet ResNet-50 layer-3 features of 256 px tissue tiles at level 1 (about 20x). No
published run yet.

## Intended use
Research only. Not a medical device. Do not use it to stage patients.

## Data
- CAMELYON17 (Bandi et al., IEEE TMI 2019; Litjens et al., GigaScience 2018). Slides from
  five Dutch centres, from the AWS open data bucket `s3://camelyon-dataset/CAMELYON17/`.
- License CC0. Slides, weights and per-slide predictions may be shared.
- Only the 100 training patients (500 slides, five lymph nodes each) have labels, in
  `stages.csv`. The challenge never released labels for its 100 test patients, so those
  slides are not used.
- Split: by patient, 60/20/20, stratified by pN stage. No patient spans two splits.
  The paper calls these patients artificial: slides were grouped into patients after
  labelling, each within one centre.
- Class counts in `stages.csv`: 318 negative, 36 ITC, 59 micro, 87 macro.

## Training
As for camelyon16-abmil: `uv run opendiagnostics prepare` streams the 500 labelled slides
(1.46 TB in all; peak disk about two slides, 16 GB at most, plus the features) and caches
their features, then `uv run opendiagnostics fit` trains the attention head for 40 epochs and
keeps the epoch with the best validation macro AUROC. Commit, hardware and training time:
from the run's `run.json`.

## Results
Test split, best-validation checkpoint, 95% patient-level bootstrap CIs from the run's
`metrics.json`.

| Metric | Value | 95% CI |
|---|---|---|

## Limitations and biases
- Not the challenge's benchmark: no pN stage or kappa, and our test split holds 20 patients.
- ITC slides are few and their tumour cells are tiny at 20x, so ITC is the hardest class.
- Centres differ in scanner and stain; report per-centre results before claiming robustness.

## Reproduce
```bash
uv run opendiagnostics prepare -c models/camelyon17-abmil/config.yaml
uv run opendiagnostics fit -c models/camelyon17-abmil/config.yaml
```
