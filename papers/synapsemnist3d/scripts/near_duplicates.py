"""How close each SynapseMNIST3D test volume is to a training or validation volume, and the test
AUROC with the closest ones left out.

MedMNIST v2 split SynapseMNIST3D by volume, not by synapse or animal, so a test volume could be
a near copy of a training one. Each volume is z-scored and compared with every train and val
volume, in all 48 axis flips and permutations of the training volume, by Pearson correlation.

Run from the repository root after ``uv run opendiagnostics prepare -c
models/synapsemnist3d-resnet18/config.yaml``:

    uv run python -I papers/synapsemnist3d/scripts/near_duplicates.py [--out FILE]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path.cwd()
DATA = ROOT / "data/medmnist/synapsemnist3d_64"
RUNS = ROOT / "models/synapsemnist3d-resnet18/runs"
THRESHOLDS = (0.99, 0.95, 0.90, 0.80)


def _zscored(images: np.ndarray) -> torch.Tensor:
    flat = torch.from_numpy(images.reshape(len(images), -1).astype(np.float32))
    flat = flat - flat.mean(1, keepdim=True)
    return flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _symmetries(volumes: np.ndarray):
    for axes in itertools.permutations((1, 2, 3)):
        permuted = volumes.transpose(0, *axes)
        for flips in itertools.product((False, True), repeat=3):
            dims = tuple(d + 1 for d, flip in enumerate(flips) if flip)
            yield np.flip(permuted, dims) if dims else permuted


def best_match(test: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """For each test volume, the highest correlation with any reference volume in any of the
    48 orientations of the reference."""
    queries = _zscored(test)
    best = torch.full((len(test),), -1.0)
    for oriented in _symmetries(reference):
        keys = _zscored(np.ascontiguousarray(oriented))
        best = torch.maximum(best, (queries @ keys.T).max(1).values)
    return best.numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write the summary as JSON here")
    args = parser.parse_args()
    test = np.load(DATA / "test_images.npy")
    reference = np.concatenate([np.load(DATA / f"{s}_images.npy") for s in ("train", "val")])
    similarity = best_match(test, reference)
    [run] = sorted(p for p in RUNS.iterdir() if (p / "metrics.json").exists())[:1]
    predictions = pd.read_csv(run / "predictions_test.csv")
    index = predictions["id"].str.removeprefix("test-").astype(int).to_numpy()
    target = predictions["target"].to_numpy()
    score = predictions["prob/excitatory synapse"].to_numpy()
    close = similarity[index]
    summary = {
        "run": run.name,
        "test_volumes": len(test),
        "reference_volumes": len(reference),
        "similarity_quantiles": {
            q: round(float(np.quantile(similarity, q)), 4) for q in (0.5, 0.9, 0.99, 1.0)
        },
        "auroc_all": round(float(roc_auc_score(target, score)), 4),
        "thresholds": {},
    }
    for threshold in THRESHOLDS:
        keep = close < threshold
        summary["thresholds"][str(threshold)] = {
            "near_duplicates": int((~keep).sum()),
            "auroc_without": round(float(roc_auc_score(target[keep], score[keep])), 4),
        }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
