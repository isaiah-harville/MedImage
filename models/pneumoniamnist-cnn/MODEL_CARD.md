# Model card: pneumoniamnist-cnn

## Summary

This is a 294k-parameter convolutional network that classifies 64×64 pediatric chest X-rays
as pneumonia or normal. It is the worked example that exercises the whole pipeline on open
data, trained on a laptop CPU in under two minutes. It is a baseline to beat, not a
state-of-the-art model.

On the official PneumoniaMNIST test split, run `20261126-011615-8a199f44` reaches AUROC
**0.9616** (95% CI 0.9442 to 0.9770). That ties the MedMNIST v2 paper's ResNet-50 at
224×224 (0.962). It is clearly below the published best: 0.991 for Google AutoML Vision in
the same paper, 0.995 for MedViT-S, and 0.997 for the C-Mixer preprint (see
`baselines.yaml` and docs/benchmarks.md).

## Intended use

Research and pipeline benchmarking only. This is not a medical device. Do not use it on
patients or to inform any clinical decision.

## Data

* **Dataset.** PneumoniaMNIST from MedMNIST v2, at 64×64. Yang et al., *Scientific Data*
  10:41 (2023), https://doi.org/10.1038/s41597-022-01721-8. Its source is Kermany et al.,
  *Cell* 172(5), 2018.
* **License.** CC BY 4.0, both as MedMNIST declares it and as the source deposit states it.
* **Split.** The official one, which has no patient IDs, so patient-level leakage cannot be
  checked:
  * train: 4,708 images (3,494 pneumonia);
  * validation: 524 (389 pneumonia);
  * test: 624 (390 pneumonia).
* **Where the splits come from.** MedMNIST splits the source training set 9:1 into train
  and validation, and uses the source's own validation set as test. So validation and
  test come from different parts of the source dataset.

## Training

* **Config:** `config.yaml`.
* **Model:** a VGG-style CNN (`simple_cnn`, width 16, depth 4, dropout 0.2).
* **Optimizer:** AdamW at 2e-3 with a cosine schedule and 1 warmup epoch, 10 epochs,
  batch size 64.
* **Augmentation:** horizontal flips and small intensity jitter.
* **Run:** seed 0, deterministic mode `strict`, on an Apple M-series CPU, about 111
  seconds including evaluation. Commit `8a199f4`, on a clean tree.

## Results

All numbers are on the test split. The checkpoint is from epoch 9 (the best validation
AUROC). Intervals are 2,000-replicate percentile bootstrap; with no patient IDs, each image
is resampled on its own.

| Metric | Test | 95% CI |
|---|---|---|
| AUROC | 0.9616 | 0.9442 to 0.9770 |
| AUPRC | 0.9672 | 0.9476 to 0.9839 |
| Accuracy | 0.8846 | 0.8605 to 0.9087 |
| Sensitivity | 0.9897 | 0.9785 to 0.9975 |
| Specificity | 0.7094 | 0.6545 to 0.7661 |
| Brier score | 0.1000 | 0.0806 to 0.1209 |
| ECE (15 bins) | 0.1191 | 0.0985 to 0.1433 |

The decision threshold, 0.5698, was chosen by Youden's J on validation.

## Limitations

* **Validation does not predict test here.**
  * AUROC: 0.9971 on validation, 0.9616 on test.
  * Expected calibration error: 0.0195 on validation, 0.1191 on test.
  * Specificity at the validation-chosen threshold: 0.9704 on validation, 0.7094 on test.
  * Measured: the model over-calls pneumonia on test.
  * Inferred: the source-train/source-validation difference described under Data is a
    likely cause. Two things to test first in the next round: model selection that is
    robust to this shift, and recalibration.
* **Low resolution.** At 64×64 most diagnostic detail is gone.
* **No leakage check.** Without patient IDs, the split cannot be checked for leakage.
* **One run.** This is one seed. The paper's baselines are means of at least 3 trials.
* **Pediatric only.** All images are pediatric, so nothing here says how the model does
  on adults or at other sites.

## Reproduce

```bash
git checkout 8a199f4
uv sync --locked --extra cpu
uv run medimage train models/pneumoniamnist-cnn/config.yaml
```

The project was called `medimage` at that commit; on later commits the command is
`uv run opendiagnostics train`.

The run is deterministic on CPU: two runs of this code gave identical metrics to the last
digit.
