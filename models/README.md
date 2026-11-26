# Model projects

Each directory here is one model on one benchmark. The shared code lives in
`src/opendiagnostics`; a project holds only what is specific to it.

```
models/<dataset>-<model>/
  config.yaml       the experiment; `uv run opendiagnostics train models/<name>/config.yaml`
  MODEL_CARD.md     intended use, data, training, results with CIs, limitations
  baselines.yaml    published results on the same benchmark, with sources
  runs/<run-id>/    published runs: small files in git, checkpoint listed in artifacts.yaml
  plugin.py         optional: dataset-specific code, listed under `plugins:` in config.yaml
```

The lab notebook (`uv run opendiagnostics notebook`, then `uv run zensical serve`) renders every
project and run from these files.

## Adding a project

1. Copy `_template/` to `models/<dataset>-<model>/`.
2. Point `data.dataset` at a built-in dataset (`csv_classification`, `coco_detection`,
   `medmnist`) or register your own in `plugin.py`.
3. Use `split.method: patient` unless the benchmark has an official patient-disjoint split.
   Use the official split whenever you compare with published results.
4. Write `baselines.yaml` with the published numbers you mean to beat: same split, same
   metric, with a source for each. See [docs/design.md](../docs/design.md#beating-published-results).
5. Check the dataset's license allows what you publish. Credentialed data (PhysioNet, for
   example) may forbid sharing derived weights or per-image predictions. See
   [docs/datasets.md](../docs/datasets.md).

## Publishing a run

1. Commit your changes, then train. A run on a dirty tree cannot be reproduced from its
   commit, and `opendiagnostics publish` refuses it.
2. `uv run opendiagnostics publish runs/<name>/<run-id> models/<name>`. Add `--no-predictions`
   when the license forbids sharing per-image outputs.
3. To compare with an earlier run on the same split:
   `uv run opendiagnostics compare <old>/predictions_test.csv <new>/predictions_test.csv`.
4. Update the model card from the run's `metrics.json`, commit, and rebuild the notebook.
5. To share the checkpoint, upload it to the project's Hugging Face repo and put its URL in
   the run's `artifacts.yaml`.

| Project | Data | Task | Best test result |
|---|---|---|---|
| [pneumoniamnist-cnn](pneumoniamnist-cnn/) | PneumoniaMNIST (chest X-ray, CC BY 4.0) | binary classification | AUROC 0.9616 [0.9442, 0.9770] |
