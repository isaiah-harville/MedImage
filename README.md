# OpenDiagnostics

Open research models for medical image diagnosis. Not a medical device.

An open-source monorepo of PyTorch models for medical image classification and object
detection. The goal is models that beat published results, shown with the statistics to
back the claim. All of them are built on one generic, reproducible pipeline.

* **Built on Lightning.** Experiments are `LightningCLI` configs. Networks come from timm,
  torchvision, and MONAI by import path; data modules, transforms (`torchvision.transforms.v2`,
  MONAI), metrics (`torchmetrics`), checkpointing, mixed precision, and EMA are the libraries'
  own.
* **Sound evaluation.**
  * Patient-level splits, with a leakage check that runs before training.
  * AUROC, AUPRC, and calibration for classification; COCO mAP for detection.
  * A patient-level bootstrap 95% CI on every metric, and a paired bootstrap to compare
    two models on the same test samples.
  * Decision thresholds chosen on validation data only.
* **Reproducible runs.** Each run records its seed, resolved config, git commit (and
  whether the tree was dirty), package versions, hardware, and data fingerprint. With
  `trainer.deterministic: true`, two CPU runs give bit-identical metrics.

Read [docs/design.md](docs/design.md) for the design and its reasoning, and
[docs/datasets.md](docs/datasets.md) for a survey of public datasets: their tasks, sizes,
licenses, access terms, and a ranked shortlist for the first models.

## Install

This repository uses [uv](https://docs.astral.sh/uv/). Pick the torch build for your
hardware, CPU or CUDA 13.2:

```bash
uv sync --extra cpu
```

```bash
uv sync --extra cu132
```

On a CUDA machine, pass the extra to every `uv run` as well (`uv run --extra cu132 ...`), or
uv re-syncs the CPU build of torch.

Outside uv, `pip install .` takes torch from PyPI as usual.

## Train

Every MedMNIST 2D subset has a project under `models/` with a recipe for one 12 GB GPU:
ConvNeXt-S pretrained on ImageNet-22k, fine-tuned at 224 px with bf16 and EMA. MedMNIST is
open and needs no account; the first run downloads the subset from Zenodo and checks its
MD5. At 224 px the twelve subsets need about 50 GB of disk once unpacked.

```bash
uv run opendiagnostics fit -c models/pneumoniamnist-convnext/config.yaml
```

Override any value from the command line:

```bash
uv run opendiagnostics fit -c models/pneumoniamnist-convnext/config.yaml --trainer.max_epochs=2
```

Each run writes `runs/<name>/<timestamp>-<commit>/`, which holds `metrics.json` (with
CIs), `run.json` (provenance), `splits.json`, `metrics.csv`, per-sample predictions, example
figures, and the best checkpoint. To re-score a finished run:

```bash
uv run opendiagnostics evaluate runs/pneumoniamnist-convnext/<run>
```

The recipes are generated: edit `scripts/medmnist_recipes.py`, then run
`uv run python scripts/medmnist_recipes.py`.

## Layout

```
src/opendiagnostics/     LightningCLI, data modules, Lightning modules, evaluation, metrics, notebook
models/<name>/           one project per model: config.yaml, baselines.yaml, MODEL_CARD.md, runs/
configs/smoke/           tiny configs on generated data, run in CI
scripts/                 generators, such as the MedMNIST recipe configs
docs/                    design.md, datasets.md, guide.md
tests/                   pytest suite
```

To add a model, see [models/README.md](models/README.md).

## Develop

```bash
uv sync --locked --extra cpu && uv run ruff format --check && uv run ruff check && uv run ty check && uv run pytest
```

CI runs the same checks on Ubuntu with Python 3.12, builds the docs, and smoke-trains every
task from the command line.

## Status and scope

These models are research artifacts and are not medical devices. Results on benchmark
datasets such as MedMNIST measure the pipeline, not clinical performance.

## License

The code is licensed under [Apache-2.0](LICENSE). Datasets keep their own licenses, and
weights or predictions derived from a dataset are bound by its terms; see
[docs/datasets.md](docs/datasets.md) before publishing either.
