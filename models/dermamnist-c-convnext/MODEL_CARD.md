# Model card: DermaMNIST-C, ConvNeXt-Small

## Summary
The dermamnist-convnext recipe trained and tested on DermaMNIST-C (Abhishek et al. 2024), the
corrected DermaMNIST split that keeps each HAM10000 lesion in one split, so no test lesion
was seen in training. Results come from each run's `metrics.json`.

## Intended use
Research only. Not a medical device.

## Data
DermaMNIST (MedMNIST v2) from HAM10000, CC BY-NC 4.0. DermaMNIST-C is Abhishek, Jain and
Hamarneh's corrected release (Zenodo 10.5281/zenodo.11101337, CC BY-NC 4.0). MedMNIST has no
patient IDs, so the bootstrap resamples images.

## Training
Config: `config.yaml`. Commit, hardware and time: each run's `run.json`.

## Limitations and biases
One run, with the MedMNIST recipe's settings, not tuned for this split.

## Reproduce
```bash
uv run opendiagnostics fit -c models/dermamnist-c-convnext/config.yaml --seed_everything=<seed>
```
