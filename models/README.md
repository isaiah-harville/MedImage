# Model projects

Each directory here is one trained model on one dataset. The shared code lives in
`src/medimage`; a project holds only what is specific to it.

```
models/<dataset>-<model>/
  config.yaml      the experiment; `uv run medimage train models/<name>/config.yaml`
  MODEL_CARD.md    intended use, data, training, results with CIs, limitations
  results/         metrics.json and run.json from the reported run (no checkpoints)
  plugin.py        optional: dataset-specific code, listed under `plugins:` in config.yaml
```

## Adding a project

1. Copy `_template/` to `models/<dataset>-<model>/`.
2. Point `data.dataset` at a built-in dataset (`csv_classification`, `coco_detection`,
   `medmnist`) or register your own in `plugin.py`.
3. Use `split.method: patient` unless the dataset has an official patient-disjoint split.
4. Train on a clean commit, then copy `metrics.json` and `run.json` from the run directory
   into `results/` and fill in the model card from them.
5. Check the dataset's license allows what you publish. Credentialed data (PhysioNet,
   for example) may forbid sharing derived weights or per-image predictions. See
   [docs/datasets.md](../docs/datasets.md).

| Project | Data | Task | Test result |
|---|---|---|---|
| [pneumoniamnist-cnn](pneumoniamnist-cnn/) | PneumoniaMNIST (chest X-ray, CC BY 4.0) | binary classification | see its model card |
