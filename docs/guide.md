# Using the pipeline

This guide takes one experiment from config to a published, comparable result. The
[code reference](reference/index.md) documents every module.

## Install

```bash
uv sync --locked --extra cpu
```

Use `--extra cu132` on an NVIDIA machine, and add `--extra monai` for MONAI's 3D networks.

## Write an experiment

An experiment is one YAML file. Everything not given takes the default shown in the
[config reference](reference/config.md). Unknown keys and wrong types are errors.

```yaml
name: pneumoniamnist-cnn
seed: 0
deterministic: strict          # strict | warn | off
task:
  name: classification
  params: {threshold: youden}  # binary threshold chosen on validation
data:
  dataset:
    name: medmnist             # any registered dataset; see `opendiagnostics list`
    params: {subset: pneumoniamnist, size: 64, download: true}
  split: {method: predefined}  # or: {method: patient, fractions: [0.7, 0.15, 0.15]}
  train_transforms:
    - {name: random_flip, params: {axes: [-1]}}
    - {name: normalize, params: {mean: [0.5], std: [0.25]}}
  eval_transforms:
    - {name: normalize, params: {mean: [0.5], std: [0.25]}}
  batch_size: 64
model:
  name: timm                   # simple_cnn | timm | monai | fasterrcnn | retinanet | fcos
  params: {model_name: resnet18, pretrained: true}
optim: {name: adamw, lr: 1.0e-3, scheduler: cosine, warmup_epochs: 1}
train: {epochs: 10, device: auto}
eval: {bootstrap_samples: 2000, bootstrap_unit: patient}
```

The task derives `in_channels`, `spatial_dims`, and `num_classes` from the dataset, so
`model.params` never sets them.

## Train

```bash
uv run opendiagnostics train models/pneumoniamnist-cnn/config.yaml
```

Override any value from the command line with `key.path=value`, for example
`train.epochs=2` or `model.params.model_name=convnext_tiny`. Each run writes
`runs/<name>/<timestamp>-<commit>/`:

| File | Holds |
|---|---|
| `config.yaml` | the fully resolved config |
| `run.json` | seed, git commit and dirty flag, package versions, hardware, data fingerprint |
| `splits.json` | the sample IDs in each split, with patient and label counts |
| `history.jsonl` | per epoch: training loss, validation metrics, learning rate |
| `metrics.json` | validation and test metrics with bootstrap intervals |
| `predictions_<split>.csv` | per-sample outputs |
| `checkpoint.pt` | weights from the best validation epoch |

## Compare

To claim that one run beats another, compare them on the same test samples:

```bash
uv run opendiagnostics compare runs/a/<run>/predictions_test.csv runs/b/<run>/predictions_test.csv
```

This reports B minus A with a paired patient-level bootstrap interval and a p-value. An
interval that excludes zero is a difference; overlapping per-run intervals do not settle
the question either way.

## Publish

Commit first, then train, so that the commit reproduces the run. Then publish the run into
its model project and rebuild the notebook:

```bash
uv run opendiagnostics publish runs/pneumoniamnist-cnn/<run> models/pneumoniamnist-cnn
uv run opendiagnostics notebook
uv run zensical serve
```

The project page ranks the run against the published baselines in `baselines.yaml`.
See [design](design.md#beating-published-results) for the rules a comparison has to meet.

## Extend

New datasets, models, tasks, and transforms are registered builders. See
[design](design.md#extension-points) and the [registry reference](reference/registry.md).
