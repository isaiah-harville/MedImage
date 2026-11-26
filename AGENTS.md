# AGENTS.md

OpenDiagnostics is a monorepo of PyTorch models for medical image classification and
detection: open research models, not a medical device. The goal is models that beat
published results, with the statistics to back the claim.
Everything runs through one pipeline in `src/opendiagnostics`. The docs site is the lab notebook.
Read [docs/design.md](docs/design.md) before changing behavior; it holds the data shapes,
the evaluation rules, the hosting decision, and the roadmap. The dataset survey, with
licenses and access terms, is in [docs/datasets.md](docs/datasets.md).

## Commands

```bash
uv sync --locked --extra cpu --extra monai && uv run ruff format --check && uv run ruff check && uv run ty check && uv run pytest
```

That is the check command. CI runs it on Ubuntu with Python 3.12, then builds the docs and
smoke-trains every config in `configs/smoke/`. On an NVIDIA machine, use
`--extra cu132` instead of `--extra cpu`.

| Task | Command |
|---|---|
| One test file | `uv run pytest tests/test_metrics_detection.py` |
| Fill or update expectations | `uv run pytest --expect-create`, then `uv run xpt failures` and `uv run xpt accept <id>` |
| Train | `uv run opendiagnostics train models/<name>/config.yaml [key.path=value ...]` |
| Compare two runs, paired | `uv run opendiagnostics compare <a>/predictions_test.csv <b>/predictions_test.csv` |
| Publish a run | `uv run opendiagnostics publish runs/<name>/<run-id> models/<name>` |
| Docs | `uv run opendiagnostics notebook && uv run zensical serve` |
| Registered components | `uv run opendiagnostics list` |

## Layout

| Path | Owns |
|---|---|
| `src/opendiagnostics/` | the pipeline: registries, config, I/O, transforms, datasets, models, tasks, metrics, engine, publish, notebook |
| `models/<name>/` | one model on one benchmark: `config.yaml`, `MODEL_CARD.md`, `baselines.yaml`, `runs/<run-id>/` |
| `configs/smoke/` | seconds-long configs on generated data |
| `docs/` | design, dataset survey, site home page; `docs/notebook/` is generated and ignored |
| `runs/` | local training output, ignored |

## Invariants

1. **Splits are patient-level.** No patient appears in two splits. `check_disjoint` runs
   before training. Datasets without patient IDs use their official split, and the run
   records that leakage could not be checked.
2. **The test split is for final numbers.** Model selection, thresholds, and calibration
   use validation data only.
3. **Every headline metric has an interval.** A percentile bootstrap that resamples
   patients. A claim that one model beats another needs `opendiagnostics compare` (a paired
   test) or a published value outside our interval.
4. **Comparisons with published results use the benchmark's protocol.** Same split, same
   metric definition. Every number in `baselines.yaml` has a source.
5. **Published runs come from clean commits.** `opendiagnostics publish` refuses a dirty-tree run.
   Runs are deterministic on CPU with `deterministic: strict`, and a test checks that.
6. **Respect dataset licenses.** Never commit or upload data, weights, or per-image
   predictions that a dataset's terms forbid sharing. The survey lists each dataset's terms.
7. **No patient data in git.** No images, DICOM headers, reports, or anything with PHI.
   Run files hold only sample IDs and model outputs.
8. **Config parsing is strict.** Unknown keys and wrong types are errors.

## Stop and ask first

On top of the hl stop points, ask before:

- applying for, downloading, or accepting the terms of a registration or credentialed
  dataset (PhysioNet, Stanford AIMI, ADNI, and others);
- creating an external account, organization, or repository (Hugging Face, Zenodo), or
  uploading a checkpoint or results to one;
- publishing a release or turning on GitHub Pages.

Open datasets that need no account (MedMNIST, for example) can be downloaded without asking.

## How to make common changes

- **Add a dataset.** Write a `DatasetSource` subclass in `src/opendiagnostics/datasets/`. Register
  it with `@DATASETS.register("<name>")` and import it in `datasets/__init__.py`. Expose
  `patient_ids`, `strata`, and `predefined_splits()` without reading pixels. Test it on a
  tiny fixture written to `tmp_path`, never on real data.
- **Add a model.** Register a builder in `src/opendiagnostics/models/` that takes `in_channels`,
  `num_classes`, and `spatial_dims`. Add a forward-shape test.
- **Add a task.** Subclass `Task` in `src/opendiagnostics/tasks/` and register it. The engine
  must not change.
- **Add a metric.** Put it in `src/opendiagnostics/metrics/` as a pure NumPy function. Test it
  against a reference implementation when one exists.
- **Start a model project.** Copy `models/_template/` and follow
  [models/README.md](models/README.md).

## Tests

- **Whole outputs use xpt.** For a test whose expected value is a whole output (rendered
  Markdown, CLI text, error messages, summaries), write `expect(actual)` from `xpt` and run
  `uv run pytest --expect-create` to fill in the value. Read the filled value before you
  commit: xpt records what the code did, not what it should do.
- **Intended changes go through xpt.** When an output changes on purpose, run
  `uv run xpt failures`, read each diff, then `uv run xpt accept <id>` for the ones you mean.
  Don't run `xpt accept --all` on diffs you haven't read.
- **Rules use plain asserts.** Invariants and properties (a metric matches scikit-learn,
  patients never span splits, a run reproduces) use `assert`.
- **Tests stay hermetic.** No network except tests marked `network`, no real datasets, and
  no files outside `tmp_path`.

## Deliberate hand-rolls

- **Metrics.** AUROC, AUPRC, COCO mAP, and FROC are written in NumPy so the bootstrap can
  re-aggregate cached per-image results. The tests pin them to scikit-learn and
  pycocotools, which are dev dependencies only.
- **Medical image loaders.** These are thin wrappers over pydicom and nibabel, so that
  DICOM rescale, `MONOCHROME1`, series ordering, and NIfTI orientation are explicit and
  tested.
- **The YAML float resolver** in `config.py`. PyYAML reads `1e-4` as a string.
