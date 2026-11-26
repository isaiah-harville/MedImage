# Design

This document records how the `opendiagnostics` pipeline is put together and why. Read it before
changing behaviour. When a decision changes, update this file in the same change.

## Goals

1. **Models that beat published results, shown with the statistics to back the claim.**
   Every result is compared with published baselines under the benchmark's own protocol,
   and a claim that one model beats another rests on a paired test on the same test
   samples, not on two numbers side by side.
2. **One pipeline for many models.** Classification and detection, 2D and 3D, X-ray, CT,
   MRI and other modalities all train with Lightning through one `LightningCLI`. A new
   network or transform is a `class_path` in a config file; a new dataset is a
   `LightningDataModule` subclass.
3. **Results that hold up to scrutiny.** No patient-level leakage, confidence intervals on
   every headline metric, decision thresholds chosen on validation data only, and enough
   provenance to rerun any result.
4. **Open and reproducible.** Every artifact in a run directory can be regenerated from its
   `config.yaml` and its git commit, and the lab notebook shows every published run.

## Repository layout

```
src/opendiagnostics/
  cli.py             `opendiagnostics fit | evaluate | compare | publish | notebook`; the LightningCLI
  data/              the data module base class, patient-level splits, MedMNIST, CSV manifests,
                     synthetic smoke data
  modules/           Lightning modules: ImageClassifier, ObjectDetector, and their base
  evaluation.py      final evaluation of the selected checkpoint with bootstrap intervals
  metrics/           torchmetrics-backed classification metrics, cluster bootstrap, paired compare
  provenance.py      git state and package versions for run.json
  publish.py         copy a finished run into a model project
  notebook/          generate the lab notebook pages from model projects
models/<name>/       one project per trained model: config.yaml, MODEL_CARD.md,
                     baselines.yaml, runs/<run-id>/
configs/smoke/       seconds-long configs on generated data, run by CI
scripts/             generators, such as the MedMNIST recipe configs
docs/                the docs site: this file, the dataset survey, the generated notebook
tests/               pytest suite
```

## Libraries, and what stays ours

The pipeline is Lightning and existing libraries wherever one does the job:

| Concern | Library |
|---|---|
| Config, CLI, strict parsing | `LightningCLI` (jsonargparse) |
| Training loop, precision, gradient clipping, determinism | `lightning.Trainer` |
| Optimizer and schedule | `torch.optim` through LightningCLI's `optimizer:` and `lr_scheduler:` |
| Checkpoint selection, early stopping, EMA | `ModelCheckpoint`, `EarlyStopping`, `EMAWeightAveraging` |
| Per-epoch logs | `CSVLogger` |
| 2D networks and pretrained weights | timm (`timm.create_model`) |
| Detectors | torchvision (`torchvision.models.detection`) |
| 3D networks, medical image reading | MONAI (`monai.networks.nets`, `LoadImage`) |
| 2D transforms, keeping boxes aligned | `torchvision.transforms.v2` |
| MedMNIST metadata and archives | `medmnist`, torchvision's `download_url` |
| Patient-level split | scikit-learn `train_test_split` over patients |
| Metrics | `torchmetrics` (functional AUROC, AP, calibration, ROC; `MeanAveragePrecision`) |

What no library does stays here, and is small:

* **The patient-level cluster bootstrap and the paired comparison.** `torchmetrics`'
  `BootStrapper` resamples single images, which makes intervals too narrow when a patient
  has several images. Each replicate calls `torchmetrics`, so no metric is computed by hand.
* **The data contract and leakage check.** Every data module serves the same batch shape
  and refuses a split in which a patient appears twice.
* **Run provenance, publishing, and the lab notebook.**

## Extension points

Configs name components by import path, so any importable class or factory works without
registration:

```yaml
model:
  class_path: opendiagnostics.modules.ImageClassifier
  init_args:
    network:
      class_path: timm.create_model
      init_args: {model_name: convnext_small.fb_in22k_ft_in1k, pretrained: true}
      dict_kwargs: {num_classes: 9, in_chans: 3}
```

`dict_kwargs` passes arguments that a factory takes through `**kwargs`. A typo there fails
when the network is built, before training starts.

* **A dataset** is a `MedicalDataModule` subclass that sets `info` (a `DatasetInfo`) in
  `__init__` and returns a `Source` from `load()`: indexable `(image, target)` pairs plus
  one sample ID and one patient ID per item, and the official split if there is one.
* **A task** is a `DiagnosticModule` subclass: loss, per-sample predictions, calibration on
  validation, and scoring with bootstrap intervals.
* **A network or transform** needs no code: name its class in the config.

The CLI links `data.info` into the model, so the module knows the label mode and class
names. The network's output size is set in the config; a mismatch is an error at the first
batch that says which argument to change.

## The data contract

Data modules serve batches of dicts:

| key | type | notes |
|---|---|---|
| `image` | `float32 (B, C, H, W)` or `(B, C, D, H, W)`; a list for detection | values in `[0, 1]` before the configured transforms |
| `target` | `int64 (B,)`, `float32 (B, K)` for multilabel, or a list of `{boxes, labels}` | boxes are pixel `(x1, y1, x2, y2)`, labels start at 0 |
| `id`, `patient_id` | `list[str]` | `patient_id` drives splits and the bootstrap |

Datasets hold only picklable data. MedMNIST arrays are unpacked to `.npy` files that each
DataLoader worker memory-maps, because Windows workers start by pickling the dataset.

## Data handling

### Loading

* **MedMNIST** comes from the official Zenodo archives, MD5-checked, using the
  `medmnist` package's URLs and metadata. Each array is streamed out of the archive to its
  own `.npy` file, so no split is held in memory.
* **Image files** listed in a CSV manifest are read with MONAI's `LoadImage` (DICOM,
  NIfTI, PNG, and more) and min-max scaled per image. DICOM (LPS) and NIfTI (RAS) use
  different patient coordinate systems; do not mix them in one dataset without resampling.

### Splits and leakage

Images of one patient are correlated. Splitting them across train and test inflates test
metrics. So:

* `split: patient` splits **patients**, not images, stratified by each patient's majority
  label.
* `split: predefined` uses the dataset's official split.
* Either way, `check_disjoint` runs in `setup()`, before training, and raises
  `LeakageError` if a patient appears in two splits.
* `splits.json` in each run records the sample IDs in every split.

Some datasets publish no patient IDs (MedMNIST, VinDr-CXR). Leakage then cannot be checked.
The run logs a warning and `run.json` records `patient_ids_available: false`.

## Training

* Lightning's `Trainer`, configured under `trainer:`. A non-finite loss stops the run.
* **Model selection** uses only the validation split: `ModelCheckpoint` keeps the epoch with
  the best `checkpoint.monitor` (`val/auroc` by default; `val/map_50` for detection). With
  `EMAWeightAveraging`, validation and the checkpoint use the averaged weights.
* The test split is scored once, after selection, with the selected checkpoint.

## Evaluation and metrics

* **Classification.**
  * Binary: AUROC, AUPRC (average precision), Brier score, expected calibration error,
    and sensitivity, specificity, PPV, NPV, accuracy and F1 at the decision threshold,
    plus `accuracy_at_0.5` for benchmarks that fix the threshold at 0.5.
  * Multiclass: macro one-vs-rest AUROC and AUPRC, accuracy, balanced accuracy, and
    per-class values.
  * Multilabel: per-label and macro AUROC and AUPRC, and accuracy at 0.5.
  * Metrics come from `torchmetrics`, which computes in float32; tests pin AUROC and AUPRC
    to scikit-learn within 1e-6, ties included.
* **Threshold.** The binary threshold is fitted on validation predictions (Youden's J by
  default, or a fixed value) and applied unchanged to test. A sample counts as positive at
  `score >= threshold`, the convention ROC thresholds use.
* **Detection.** COCO mAP, mAP@0.5, mAP@0.75 and mAR@100 from `torchmetrics`'
  `MeanAveragePrecision` (pycocotools backend).
* **Confidence intervals.**
  * Every metric gets a percentile bootstrap interval, 1,000 replicates by default.
  * The default unit is the **patient** (a cluster bootstrap): resampling images when
    patients contribute several makes intervals falsely narrow.
  * Replicates where a metric is undefined (a resample with no positives) are dropped,
    and `valid_replicates` records how many remained.
* Per-sample predictions are written to `predictions_<split>.csv`, so any further analysis
  (subgroups, other metrics, DeLong tests) can run without the model.

## Reproducibility

* `seed_everything` seeds Python, NumPy and torch, and DataLoader workers.
* `trainer.deterministic` sets the policy for nondeterministic operations: `true` (the
  default) errors on them, `warn` only warns. Several CUDA backward passes have no
  deterministic kernel, so GPU recipes use `warn`.
* The tests check that two CPU runs of one config give bit-identical metrics.
* `run.json` records:
  * the git commit, branch, and whether the tree had uncommitted changes (and warns if so);
  * Python and package versions, platform, and GPU;
  * the dataset's license, citation and fingerprint (for MedMNIST, the archive MD5);
  * the command line and start and end times.
* `config.yaml` in the run is the fully resolved config. Parsing is strict: unknown keys
  and wrong types are errors, so a typo cannot silently fall back to a default.

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
  models, `opendiagnostics compare` runs a paired patient-level bootstrap on the same test
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
`opendiagnostics notebook` generates `docs/notebook/` from `models/` before each build:

* a home page listing every project and its latest result;
* a page per project: the leaderboard against published baselines, the runs, and the
  model card;
* a page per run: metrics with intervals, training curves, ROC and precision-recall
  curves for binary tasks, example test images, the data splits, provenance, the resolved
  config, and downloads.

**Example images.** Each run renders up to eight test images into `examples/`, each beside
what the model made of it. For a classifier that is a Captum Integrated Gradients map for
the predicted class; for a detector, ground-truth boxes in green and predicted boxes with
scores in red. Classifier examples are the most confident correct prediction and the most
confident mistake per class, so the page shows failures as well as successes.
`opendiagnostics examples RUN_DIR` renders them for an existing run. Images are data, so
`publish` copies them only when the dataset's license allows redistribution (CC0, CC BY,
CC BY-NC, or generated data) and predictions are published too. Attribution maps show what
the network responded to; they are not proof that it uses the clinically relevant region.

The generated pages are not committed. They are rebuilt from the committed run files, so
they cannot go stale.

### Where artifacts live

| Artifact | Size | Where | Why |
|---|---|---|---|
| Config, run record, metrics, history, split summary, predictions | KB to a few MB | git, under `models/<project>/runs/<run-id>/`; served by the docs site | Reviewed with the code, versioned with it, and rendered and downloadable from the run page. `opendiagnostics publish` refuses files over 20 MB. |
| Checkpoints | about 200 MB each | git through Git LFS, next to the run's other files, when the dataset's license allows sharing weights | One place for a run and its weights, versioned with the code; CI pulls them to build the demo image. The run page links the file in the repository and shows its SHA-256 from `artifacts.yaml`. |
| Citable snapshots | the repository at a release | Zenodo, through its GitHub release integration | A DOI per version plus a concept DOI for all versions, for papers that cite our results. |

We weighed four hosts for checkpoints, against limits taken from each host's documentation
on 2026-11-26:

* **Git LFS in this repository (chosen).**
  * The free quota is 10 GiB of storage and 10 GiB of bandwidth a month. Twelve MedMNIST
    checkpoints are about 2.4 GB.
  * Every LFS object stays in storage after it is replaced, so publish runs worth keeping,
    not every experiment.
  * GitHub Pages cannot serve LFS files, so the notebook does not copy checkpoints into
    the site; the run page links them in the repository instead.
  * The demo image's CI caches LFS objects by their pointers, so a rebuild downloads only
    checkpoints that changed.
* **Hugging Face Hub.** Built for model weights, with no per-repo size limit, but it needs
  an organization and a second place to keep in step with the runs. Revisit if LFS storage
  or bandwidth runs short.
* **GitHub Releases.** Assets under 2 GiB, no total or bandwidth limit, but tied to code
  tags rather than runs, with no folders.
* **Zenodo.** 50 GB and 100 files per record, with DOIs. Right for archival snapshots that
  papers cite, too heavy for every run.

`opendiagnostics publish` copies the checkpoint into the published run folder, where
`.gitattributes` routes it through Git LFS, only when the dataset's license is in
`OPEN_LICENSES` (`publish.py`). For credentialed data, publish metrics only
(`opendiagnostics publish --no-predictions --no-checkpoint`) unless the agreement allows more.
Run `git lfs install` once on any machine that publishes.

## Roadmap

* A segmentation task (Dice and HD95 with bootstrap intervals), since many leading
  datasets (BraTS, KiTS, MSD) are segmentation-first.
* 3D detection: boxes in `(x1, y1, z1, x2, y2, z2)`, through MONAI's 3D RetinaNet or our
  own head.
* Resampling to a common voxel spacing and orientation as a transform.
* Training features for state-of-the-art results: weight EMA, test-time augmentation,
  higher-resolution pretrained backbones, and GPU recipes per benchmark.
* Multi-seed runs aggregated on the project page, to match papers that report means.
* Subgroup reports (sex, age, site or scanner) when the manifest provides those columns.
* Dataset adapters for the starter datasets in [datasets.md](datasets.md).
