# OpenDiagnostics

Open research models for medical image diagnosis. Not a medical device.

An open-source monorepo of PyTorch models for medical image classification and object
detection. The goal is models that beat published results, shown with the statistics to
back the claim. All of them are built on one generic, reproducible pipeline.

* **One loop, many tasks.** Classification (binary, multiclass, multilabel) and 2D
  detection share one training and evaluation loop. Datasets, models, tasks, and transforms
  are registries, so new ones plug in without touching the loop.
* **Medical images.** DICOM (with rescale to Hounsfield units, `MONOCHROME1`, and series
  stacking), NIfTI (canonical orientation), and PNG/JPEG/TIFF. Transforms work in 2D and
  3D and keep boxes aligned.
* **Sound evaluation.**
  * Patient-level splits, with a leakage check that runs before training.
  * AUROC, AUPRC, and calibration for classification; COCO mAP and FROC for detection.
    Our values match scikit-learn and pycocotools.
  * A patient-level bootstrap 95% CI on every metric.
  * Decision thresholds chosen on validation data only.
* **Reproducible runs.** Each run records its seed, resolved config, git commit (and
  whether the tree was dirty), package versions, hardware, and data fingerprint. With
  `deterministic: strict`, two CPU runs give bit-identical metrics.

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

Add `--extra monai` for MONAI's 3D networks. Outside uv, `pip install .` takes torch from
PyPI as usual.

## Train the worked example

PneumoniaMNIST is open (CC BY 4.0) and needs no account. The first run downloads a 21 MB
file from Zenodo and checks its MD5. On a laptop CPU, training takes about three minutes:

```bash
uv run opendiagnostics train models/pneumoniamnist-cnn/config.yaml
```

For a quick check, override any config value from the command line:

```bash
uv run opendiagnostics train models/pneumoniamnist-cnn/config.yaml train.epochs=2
```

Each run writes `runs/<name>/<timestamp>-<commit>/`, which holds `metrics.json` (with
CIs), `run.json` (provenance), `splits.json`, `history.jsonl`, per-sample predictions, and
the best checkpoint. To re-score a finished run:

```bash
uv run opendiagnostics evaluate runs/pneumoniamnist-cnn/<run>
```

`uv run opendiagnostics list` shows every registered dataset, model, task, and transform.

## Layout

```
src/opendiagnostics/     shared core: registry, config, io, transforms, datasets, models, tasks, metrics, engine
models/<name>/    one project per trained model: config.yaml, MODEL_CARD.md, results/
configs/smoke/    tiny configs on generated data, run in CI
docs/             design.md, datasets.md
tests/            pytest suite
```

To add a model, see [models/README.md](models/README.md).

## Develop

```bash
uv sync --locked --extra cpu --extra monai && uv run ruff format --check && uv run ruff check && uv run ty check && uv run pytest
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
