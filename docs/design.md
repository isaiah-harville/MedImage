# Design

This document records how the `medimage` pipeline is put together and why. Read it before
changing behaviour. When a decision changes, update this file in the same change.

## Goals

1. **Models that beat published results, shown with the statistics to back the claim.**
   Every result is compared with published baselines under the benchmark's own protocol,
   and a claim that one model beats another rests on a paired test on the same test
   samples, not on two numbers side by side.
2. **One pipeline for many models.** Classification and detection, 2D and 3D, X-ray, CT,
   MRI and other modalities all run through one training and evaluation loop. A new
   dataset or model is a registered builder plus a config file, with no fork of the loop.
3. **Results that hold up to scrutiny.** No patient-level leakage, confidence intervals on
   every headline metric, decision thresholds chosen on validation data only, and enough
   provenance to rerun any result.
4. **Open and reproducible.** Every artifact in a run directory can be regenerated from its
   `config.yaml` and its git commit, and the lab notebook shows every published run.

## Repository layout

```
src/medimage/        the shared core package
  registry.py        DATASETS, MODELS, TASKS, TRANSFORMS registries and plugin loading
  config.py          the experiment schema (strict Pydantic models) and the YAML loader
  data/              the sample contract, patient-level splits, collation
  io/                DICOM, DICOM series, NIfTI, PNG/JPEG/TIFF, and NPY loaders
  transforms/        2D/3D-aware sample transforms that keep boxes aligned
  datasets/          built-in datasets (MedMNIST, CSV manifest, COCO manifest, synthetic)
  models/            built-in model builders (simple CNN, timm, torchvision detectors, MONAI)
  tasks/             classification and detection: loss, prediction, metrics
  metrics/           AUROC, AUPRC, calibration, mAP, FROC, cluster bootstrap, paired comparison
  engine/            the training and evaluation loop and the run directory
  publish.py         copy a finished run into a model project
  notebook/          generate the lab notebook pages from model projects
  repro.py           seeding, deterministic mode, provenance
  cli.py             `medimage train | evaluate | compare | publish | notebook | list`
models/<name>/       one project per trained model: config.yaml, MODEL_CARD.md,
                     baselines.yaml, runs/<run-id>/
configs/smoke/       seconds-long configs on generated data, run by CI
docs/                the docs site: this file, the dataset survey, the generated notebook
zensical.toml        the docs site config
tests/               pytest suite
```

The core is one installable package. Per-model projects under `models/` hold configs, model
cards, and committed results, plus an optional `plugin.py` for code that applies to one
dataset only (a custom label parser, say). Code that more than one project needs moves into
the core.

## Extension points

Every extensible piece is a `Registry` that maps a name to a builder. Builders register when
their module is imported:

```python
from medimage.registry import MODELS


@MODELS.register("my_net")
def my_net(in_channels: int, num_classes: int, spatial_dims: int = 2, depth: int = 3): ...
```

Plugins come from three places: the built-in modules; installed packages that declare a
`medimage.plugins` entry point; and modules or `.py` files listed under `plugins:` in a
config. Importing a plugin runs its code, so configs should list only trusted plugins.

The **task** is the seam that keeps the loop generic. A `Task`:

* sizes the model from the dataset (`model_kwargs`: channels, spatial dims, outputs);
* collates samples into batches;
* computes the training loss;
* turns model output into per-sample predictions;
* fits any calibration on validation predictions (the binary threshold);
* scores predictions, with bootstrap intervals when asked.

The engine never branches on the task type. Segmentation, regression, or survival can be
added as new tasks without touching the loop.

## The sample contract

Datasets return plain dicts (see `src/medimage/data/base.py`):

| key | type | notes |
|---|---|---|
| `image` | `float32 (C, H, W)` or `(C, D, H, W)` | channel-first, 2D or 3D |
| `label` | `int64` scalar, or `float32 (K,)` for multilabel | classification |
| `boxes` | `float32 (N, 4)`, pixel `(x1, y1, x2, y2)` | detection, 2D |
| `labels` | `int64 (N,)`, foreground classes from 0 | detection |
| `id`, `patient_id` | `str` | `patient_id` drives splits and the bootstrap |

Sources also expose `patient_ids`, `strata`, and `predefined_splits()` **without loading
pixels**, so splits are computed and checked before training starts. A `DatasetInfo` says
what the dataset holds (task type, dims, channels, class names, label mode, license,
citation) and whether patient IDs exist.

## Framework decision: torchvision, timm, and MONAI

We weighed three libraries. All three are permissively licensed (torchvision BSD-3-Clause;
timm and MONAI Apache-2.0) and actively maintained.

| | torchvision | timm | MONAI |
|---|---|---|---|
| What it offers | Detection models (Faster R-CNN, RetinaNet, FCOS), box ops, natural-image backbones | About a thousand 2D classification backbones with pretrained weights; `in_chans` adapts first-layer weights for grayscale input | Medical-specific: 3D networks, a large dictionary transform library, loaders, detection (RetinaNet 3D), and its own training engines |
| 3D support | None | None | Extensive |
| Dependencies | torch, numpy, pillow | torch, torchvision, huggingface-hub, safetensors, pyyaml | torch, numpy (many features need further optional packages) |
| Abstraction cost | Low; plain `nn.Module`s | Low; plain `nn.Module`s | Higher; `MetaTensor`, its own transform and engine conventions |

**Decision.**

* **torchvision is a core dependency.** It supplies the 2D detectors. Writing our own
  Faster R-CNN would add risk and no value.
* **timm is a core dependency.** It is the broadest, best-maintained source of 2D
  classification backbones and pretrained weights, and its models are ordinary modules.
  The `timm` model builder takes any `model_name`.
* **MONAI is an optional extra (`medimage[monai]`).** It is the right source for 3D
  networks. The `monai` model builder wraps any `monai.networks.nets` class and passes
  sizing arguments when the constructor accepts them. MONAI stays out of the core
  because adopting its transforms, `MetaTensor`, and engines would make it a second
  framework the pipeline has to follow: the sample contract, the training loop, and the
  metrics with patient-level bootstrap intervals (which MONAI's metrics do not provide)
  are owned here, so what a run does stays auditable in one small codebase. A plugin is
  free to call MONAI transforms inside a dataset or transform.

We wrote our own loaders and transforms on top of pydicom and nibabel, not MONAI's,
because the set we need is small (load, window, normalize, resize, flip) and box-aware
spatial transforms must follow our box convention. If transform needs outgrow this set,
revisit: a MONAI adapter transform is the likely next step.

## Data handling

### Loading

* **DICOM**: the modality LUT (rescale slope and intercept) is always applied, so CT is in
  Hounsfield units. The VOI LUT (display windowing) is applied only on request, because
  windowing discards information. `MONOCHROME1` is inverted so that higher always means
  brighter. A directory is read as one series, slices sorted along the slice normal, and
  unevenly spaced slices are an error instead of a silent distortion.
* **NIfTI**: reoriented to the closest canonical RAS+ orientation, then transposed to
  `(z, y, x)`.
* DICOM (LPS) and NIfTI (RAS) still use different patient coordinate systems. Do not mix
  the two in one dataset without resampling to a common orientation.

### Splits and leakage

Images of one patient are correlated. Splitting them across train and test inflates test
metrics. So:

* `split.method: patient` splits **patients**, not images, stratified by each patient's
  majority label.
* `split.method: predefined` uses the dataset's official split.
* Either way, `check_disjoint` runs before training and raises `LeakageError` if a patient
  appears in two splits.
* `splits.json` in each run records the sample IDs in every split.

Some datasets publish no patient IDs (MedMNIST, VinDr-CXR). Leakage then cannot be checked.
The run logs a warning and `run.json` records `patient_ids_available: false`.

## Training

* One loop for every task (`src/medimage/engine/__init__.py`). AdamW, Adam, or SGD. No
  weight decay on biases and normalization parameters. Cosine or step schedule with linear
  warmup. Optional AMP on CUDA and gradient clipping. A non-finite loss stops the run.
* **Model selection** uses only the validation split: the checkpoint with the best
  `train.monitor` (AUROC for classification, mAP@0.5 for detection, by default). There is
  optional early stopping.
* The test split is scored once, after selection.

## Evaluation and metrics

* **Classification.**
  * Binary: AUROC, AUPRC (average precision), Brier score, expected calibration error,
    and sensitivity, specificity, PPV, NPV, accuracy and F1 at a decision threshold.
  * Multiclass: macro one-vs-rest AUROC and AUPRC, accuracy, balanced accuracy, and
    per-class values.
  * Multilabel: per-label and macro AUROC and AUPRC.
  * AUROC and AUPRC match scikit-learn exactly, including ties. The tests check this.
* **Threshold.** The binary threshold is fitted on validation predictions (Youden's J by
  default, or a fixed value) and applied unchanged to test. Choosing a threshold on the
  test set is a common source of optimistic sensitivity and specificity.
* **Detection.**
  * COCO-protocol mAP (averaged over IoU 0.50 to 0.95), mAP@0.5, mAP@0.75 and per-class
    AP@0.5. These match `pycocotools` to 1e-6 on randomized scenes; the tests check this.
  * Lesion-level FROC: sensitivity at 1/8 to 8 false positives per image, and the CPM
    (their mean), as in LUNA16. Repeat hits on an already-found lesion are ignored.
* **Confidence intervals.**
  * Every metric gets a percentile bootstrap interval, 1,000 replicates by default.
  * The default unit is the **patient** (a cluster bootstrap): resampling images when
    patients contribute several makes intervals falsely narrow.
  * Replicates where a metric is undefined (a resample with no positives) are dropped,
    and `valid_replicates` records how many remained.
* Per-sample predictions are written to `predictions_<split>.csv`, so any further analysis
  (subgroups, other metrics, DeLong tests) can run without the model.

## Reproducibility

* `seed` seeds Python, NumPy and torch. DataLoader workers get derived seeds, and shuffling
  uses a seeded generator.
* `deterministic` sets the policy for nondeterministic operations:
  * `strict` (the default) errors on them;
  * `warn` only warns. Some CUDA backward passes used by the detectors have no
    deterministic implementation, so detection on a GPU may need this;
  * `off` allows them.
* The tests check that two CPU runs of one config give bit-identical metrics.
* `run.json` records:
  * the git commit, branch, and whether the tree had uncommitted changes (and warns if so);
  * Python and package versions, platform, and GPU;
  * the dataset's license, citation and fingerprint (for MedMNIST, the verified MD5);
  * the command line and start and end times.
* `config.yaml` in the run is the fully resolved config. Config parsing is strict: unknown
  keys and wrong types are errors, so a typo cannot silently fall back to a default.

## Beating published results

A claim that we outperform a published method has to survive three questions: is it the
same benchmark, is the difference bigger than the noise, and would it hold on a rerun.

* **Same benchmark.** Use the benchmark's official split and its metric definition. Each
  project's `baselines.yaml` records the benchmark, the protocol, and every published
  number with its source. Numbers reported under other protocols (image-level splits,
  extra training data, a different label set) do not go in the file.
* **Bigger than the noise.** The leaderboard on each project page places every published
  number against our best run's 95% interval: above it, inside it, or below it. A published
  number inside our interval is a tie, whatever the point estimates say. Between our own
  models, `medimage compare` runs a paired patient-level bootstrap on the same test
  samples and reports the difference, its interval, and a p-value. When a baseline's code
  is public, reproduce it in this pipeline and compare paired, because a paired test is
  far more sensitive than comparing two intervals.
* **Would it hold on a rerun.** Publish results from clean commits only, so the commit
  reproduces the run. Where a paper reports a mean over seeds, report ours the same way
  before claiming a win.
* **Test once.** Model and threshold selection use validation data only. Repeatedly
  checking the test set while tuning turns it into a validation set, so tune on
  validation and look at test when a run is final.

## Lab notebook and artifact hosting

The docs site, built with [Zensical](https://zensical.org), is the project's lab notebook.
`medimage notebook` generates `docs/notebook/` from `models/` before each build:

* a home page listing every project and its latest result;
* a page per project: the leaderboard against published baselines, the runs, and the
  model card;
* a page per run: metrics with intervals, training curves, ROC and precision-recall
  curves for binary tasks, the data splits, provenance, the resolved config, and
  downloads.

The generated pages are not committed. They are rebuilt from the committed run files, so
they cannot go stale.

### Where artifacts live

| Artifact | Size | Where | Why |
|---|---|---|---|
| Config, run record, metrics, history, split summary, predictions | KB to a few MB | git, under `models/<project>/runs/<run-id>/`; served by the docs site | Reviewed with the code, versioned with it, and rendered and downloadable from the run page. `medimage publish` refuses files over 20 MB. |
| Checkpoints | MB to GB | A Hugging Face Hub model repo per project, one folder per run | Built for model weights. The run page links the file and shows its SHA-256 from `artifacts.yaml`. |
| Citable snapshots | the repository at a release | Zenodo, through its GitHub release integration | A DOI per version plus a concept DOI for all versions, for papers that cite our results. |

We weighed four hosts for checkpoints, against limits taken from each host's documentation
on 2026-11-26:

* **Hugging Face Hub (chosen).**
  * No per-repo size limit. Files up to 500 GB (under 200 GB recommended).
  * The README renders as a model card, with license, datasets, and evaluation results
    in its metadata.
  * Downloads use stable URLs: `https://huggingface.co/<org>/<repo>/resolve/<revision>/<path>`.
  * `huggingface_hub` is already installed through timm.
  * Free public storage is "best-effort" beyond the first few gigabytes, so publish
    checkpoints for runs worth keeping, not every experiment.
* **GitHub Releases.**
  * Each asset must be under 2 GiB, with up to 1,000 assets per release and no total
    size or bandwidth limit.
  * A good fallback, but releases are tied to code tags rather than runs, are flat lists
    with no folders, and are not discoverable as models.
* **Zenodo.**
  * 50 GB and 100 files per record, with DOIs for every version.
  * Right for archival snapshots that papers cite, too heavy for every run.
* **Git LFS on GitHub Pages (rejected).**
  * GitHub's documentation says Git LFS cannot be used with GitHub Pages sites.
  * A Pages site is capped at 1 GB.
  * The free LFS quota is 10 GiB of storage and 10 GiB of bandwidth a month.

Uploading is a separate, deliberate step because it publishes data. Upload the checkpoint
to the project's Hugging Face repo, put its URL in the run's `artifacts.yaml`, and rebuild
the notebook. Rerunning `medimage publish` keeps that URL while the checkpoint's SHA-256 is
unchanged. Check the dataset's license before uploading weights or predictions; for
credentialed data, publish metrics only (`medimage publish --no-predictions`) unless the
agreement allows more.

## Roadmap

* A segmentation task (Dice and HD95 with bootstrap intervals), since many leading
  datasets (BraTS, KiTS, MSD) are segmentation-first.
* 3D detection: boxes in `(x1, y1, z1, x2, y2, z2)`, through MONAI's 3D RetinaNet or our
  own head.
* Resampling to a common voxel spacing and orientation as a transform.
* Training features for state-of-the-art results: weight EMA, test-time augmentation,
  higher-resolution pretrained backbones, and GPU recipes per benchmark.
* Multi-seed runs aggregated on the project page, to match papers that report means.
* An upload command for checkpoints, once the Hugging Face organization exists.
* Subgroup reports (sex, age, site or scanner) when the manifest provides those columns.
* Dataset adapters for the starter datasets in [datasets.md](datasets.md).
