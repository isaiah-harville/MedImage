# Model card: CHANGE-ME

## Summary
One paragraph: what the model predicts, from which images, and how well (test metric with CI).

## Intended use
Research only. Not a medical device. State what it must not be used for.

## Data
- Dataset, version, citation, and link.
- License and access terms; what they allow us to publish.
- Split method (patient-level or official) and per-split patient and image counts (from `splits.json`).
- Label source (radiologist, NLP-extracted, pathology) and known label noise.

## Training
Config: `config.yaml`. Commit, hardware, and training time: from the run's `run.json`.

## Results
Test split, best-validation checkpoint, 95% patient-level bootstrap CIs from the run's `metrics.json`.

| Metric | Value | 95% CI |
|---|---|---|

Decision threshold and how it was chosen (validation only).

## Limitations and biases
Site and scanner shift, population, label noise, missing subgroups.

## Reproduce
```bash
uv run opendiagnostics fit -c models/CHANGE-ME/config.yaml
```
