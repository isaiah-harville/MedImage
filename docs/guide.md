# Using the pipeline

This guide takes one experiment from config to a published, comparable result. The
[code reference](reference/index.md) documents every module.

## Install

```bash
uv sync --locked --extra cpu
```

Use `--extra cu132` on an NVIDIA machine.

## Write an experiment

An experiment is one `LightningCLI` YAML file. Components are named by import path
(`class_path`) with their arguments (`init_args`). Unknown keys and wrong types are errors.
`uv run opendiagnostics fit --help` lists every option, and
`uv run opendiagnostics fit --print_config` prints a full config with the defaults.

```yaml
name: pneumoniamnist-convnext
seed_everything: 0
trainer:
  precision: bf16-mixed
  deterministic: warn            # true on CPU; some CUDA kernels have no deterministic version
  max_epochs: 40
  callbacks:
    - class_path: lightning.pytorch.callbacks.EMAWeightAveraging
      init_args: {decay: 0.998}
model:
  class_path: opendiagnostics.modules.ImageClassifier
  init_args:
    network:
      class_path: timm.create_model
      init_args: {model_name: convnext_small.fb_in22k_ft_in1k, pretrained: true}
      dict_kwargs: {num_classes: 1, in_chans: 1}   # binary: 1 logit
    threshold: youden            # binary threshold chosen on validation
data:
  class_path: opendiagnostics.data.MedMNISTDataModule
  init_args:
    flag: pneumoniamnist
    size: 224
    split: predefined            # or patient, with fractions: [0.7, 0.15, 0.15]
    batch_size: 64
    train_transforms:
      class_path: torchvision.transforms.v2.RandomResizedCrop
      init_args: {size: 224, scale: [0.7, 1.0]}
optimizer:
  class_path: torch.optim.AdamW
  init_args: {lr: 1.0e-4, weight_decay: 0.05}
lr_scheduler:
  class_path: torch.optim.lr_scheduler.CosineAnnealingLR
  init_args: {T_max: 40}
checkpoint: {monitor: val/auroc, mode: max}
eval: {bootstrap_samples: 2000, bootstrap_unit: patient}
```

The network's output size must match the task: one logit for binary, one per class for
multiclass, one per label for multilabel. A mismatch stops the run at the first batch.

## Train

```bash
uv run opendiagnostics fit -c models/pneumoniamnist-convnext/config.yaml
```

Override any value on the command line, for example `--trainer.max_epochs=2` or
`--data.init_args.batch_size=32`. Each run writes `runs/<name>/<timestamp>-<commit>/`:

| File | Holds |
|---|---|
| `config.yaml` | the fully resolved config |
| `run.json` | seed, git commit and dirty flag, package versions, hardware, data fingerprint |
| `splits.json` | the sample IDs in each split, with patient and label counts |
| `metrics.csv` | per epoch: training loss, validation metrics, learning rate |
| `metrics.json` | validation and test metrics with bootstrap intervals |
| `predictions_<split>.csv` | per-sample outputs |
| `checkpoint.ckpt` | weights from the best validation epoch |
| `examples/` | test images beside the model's attribution map or boxes, for the notebook |

## Queue several runs

`scripts/run_queue.py` runs a list of jobs on one GPU while their datasets download in
parallel, so the GPU never waits on a download it could have overlapped:

```yaml
# queue.yaml
jobs:
  - name: organmnist3d
    config: models/organmnist3d-densenet/config.yaml
  - name: nih-chestxray14
    config: models/nih-chestxray14-convnext/config.yaml
    prepare: [uv, run, python, scripts/datasets/nih_chestxray14.py, data/nih-chestxray14]
```

```bash
uv run python scripts/run_queue.py queue.yaml --downloads 3
```

Every job's data step starts at once, up to `--downloads` at a time: `prepare` if given,
otherwise `opendiagnostics prepare -c <config>`, which downloads and unpacks the config's
dataset without training. The GPU then trains the jobs in order, each as soon as its own
data is ready. A failed job is recorded and the queue moves on. Status is in
`queue-logs/queue-status.json`, and rerunning the queue skips jobs that already succeeded.
A job can also be a plain command (`run: [...]`) that runs in the GPU lane in order.

## A threshold from cross-validation

A binary model's decision threshold is fitted on the validation split. When that split is
small, the threshold is noisy. Train `k` runs that each hold out one fold of the pooled
official train and validation splits, then fit the threshold on their out-of-fold
predictions:

```bash
for k in 0 1 2 3 4; do
  uv run opendiagnostics fit -c models/<name>/config.yaml --name=<name>-fold$k \
    --data.init_args.split=kfold --data.init_args.folds=5 --data.init_args.fold=$k
done
uv run opendiagnostics evaluate runs/<name>/<run> --threshold-from runs/<name>-fold*/*
```

Folds are grouped by patient and stratified by label, and the official test split is never
used. `evaluate` checks that the folds' held-out sets are disjoint, cover the run's train and
validation samples, and contain no test sample, then rescores the run with that threshold.
Publish the run again to update its project.

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
uv run opendiagnostics publish runs/pneumoniamnist-convnext/<run> models/pneumoniamnist-convnext
uv run opendiagnostics notebook
uv run zensical serve
```

Publishing uploads the checkpoint to the public weights bucket when the dataset's license
allows it; see [model weights](weights.md). The project page ranks the run against the
published baselines in `baselines.yaml`.
See [design](design.md#beating-published-results) for the rules a comparison has to meet.

## Extend

Networks and transforms need no code: name any importable class in the config. A new
dataset is a `MedicalDataModule` subclass, and a new task is a `DiagnosticModule` subclass.
See [design](design.md#extension-points).
