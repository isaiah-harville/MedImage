"""Recover NIH patient IDs for ChestMNIST, then rescore the runs with a patient-level bootstrap and
count patients shared between ChestMNIST's train and test splits.

ChestMNIST ships images and labels without NIH file names. Each ChestMNIST image is matched to
the NIH ChestX-ray14 image with the same 14 labels whose 32 x 32 thumbnail correlates best with
it; the margin over the second-best candidate shows how certain each match is.

Heavy: it reads all 112,120 NIH images (about 42 GB). Run it on the training machine, from the
repository root, after ``scripts/datasets/nih_chestxray14.py data/nih_chestxray14`` and
``uv run opendiagnostics prepare -c models/chestmnist-convnext/config.yaml``:

    uv run python -I papers/chestmnist/scripts/patient_ids.py --nih data/nih_chestxray14
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from opendiagnostics.metrics.bootstrap import bootstrap

ROOT = Path.cwd()
RUNS = ROOT / "models/chestmnist-convnext/runs"
OUT = ROOT / "papers/chestmnist/generated"
LABELS = (
    "atelectasis",
    "cardiomegaly",
    "effusion",
    "infiltration",
    "mass",
    "nodule",
    "pneumonia",
    "pneumothorax",
    "consolidation",
    "edema",
    "emphysema",
    "fibrosis",
    "pleural",
    "hernia",
)
NIH_NAMES = {name.split("_")[0].lower(): i for i, name in enumerate(LABELS)}
THUMB = 32
SPLITS = ("train", "val", "test")


def thumbnail(image: Image.Image) -> np.ndarray:
    small = image.convert("L").resize((THUMB, THUMB), Image.Resampling.BOX)
    flat = np.asarray(small, dtype=np.float32).ravel()
    flat = flat - flat.mean()
    return flat / max(float(np.linalg.norm(flat)), 1e-6)


def _nih_thumbnail(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return thumbnail(image)


def nih_key(label: str) -> tuple[int, ...]:
    findings = [f for f in label.split("|") if f] if isinstance(label, str) else []
    return tuple(sorted(NIH_NAMES[f.split("_")[0].lower()] for f in findings))


def chest_key(row: np.ndarray) -> tuple[int, ...]:
    return tuple(int(i) for i in np.flatnonzero(row))


def match(
    queries: np.ndarray, keys: list[tuple[int, ...]], nih: np.ndarray, nih_keys: list
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best NIH index, its correlation, and the second-best correlation, per query."""
    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, key in enumerate(nih_keys):
        buckets[key].append(index)
    best = np.full(len(queries), -1, np.int64)
    first = np.full(len(queries), np.nan, np.float32)
    second = np.full(len(queries), np.nan, np.float32)
    by_key: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        by_key[key].append(index)
    for key, members in by_key.items():
        candidates = np.asarray(buckets.get(key, []), np.int64)
        if len(candidates) == 0:
            continue
        bank = torch.from_numpy(nih[candidates])
        for start in range(0, len(members), 2048):
            chunk = np.asarray(members[start : start + 2048])
            scores = torch.from_numpy(queries[chunk]) @ bank.T
            top = scores.topk(min(2, len(candidates)), dim=1)
            best[chunk] = candidates[top.indices[:, 0].numpy()]
            first[chunk] = top.values[:, 0].numpy()
            if top.values.shape[1] > 1:
                second[chunk] = top.values[:, 1].numpy()
    return best, first, second


def macro_auroc(target: np.ndarray, score: np.ndarray):
    def statistic(index: np.ndarray) -> dict[str, float]:
        values = []
        for label in range(target.shape[1]):
            y = target[index, label]
            values.append(roc_auc_score(y, score[index, label]) if 0 < y.sum() < len(y) else np.nan)
        return {"auroc": float(np.nanmean(values))}

    return statistic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nih", type=Path, required=True, help="NIH ChestX-ray14 folder")
    parser.add_argument(
        "--chest",
        type=Path,
        default=Path.home() / ".medmnist/chestmnist_224",
        help="unpacked ChestMNIST arrays, where the chestmnist-convnext config puts them",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = pd.read_csv(args.nih / "manifest.csv")
    with ProcessPoolExecutor(args.workers) as pool:
        nih = np.stack(
            list(pool.map(_nih_thumbnail, (args.nih / p for p in manifest["path"]), chunksize=256))
        )
    nih_keys = [nih_key(label) for label in manifest["label"]]

    rows = []
    for split in SPLITS:
        images = np.load(args.chest / f"{split}_images.npy", mmap_mode="r")
        labels = np.load(args.chest / f"{split}_labels.npy")
        queries = np.stack([thumbnail(Image.fromarray(np.asarray(im))) for im in images])
        best, first, second = match(queries, [chest_key(r) for r in labels], nih, nih_keys)
        found = best >= 0
        rows.append(
            pd.DataFrame(
                {
                    "id": [f"{split}-{i}" for i in range(len(images))],
                    "split": split,
                    "nih_image": np.where(found, manifest["path"].to_numpy()[best], ""),
                    "nih_split": np.where(found, manifest["split"].to_numpy()[best], ""),
                    "patient_id": np.where(found, manifest["patient_id"].to_numpy()[best], ""),
                    "correlation": first,
                    "runner_up": second,
                }
            )
        )
    table = pd.concat(rows, ignore_index=True)
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "patient_ids.csv.gz", index=False)

    confident = (table["correlation"] >= 0.95) & (
        table["runner_up"].isna() | (table["correlation"] - table["runner_up"] >= 0.02)
    )
    test = table[table["split"] == "test"]
    patients = {s: set(table.loc[(table["split"] == s) & confident, "patient_id"]) for s in SPLITS}
    summary = {
        "matched": {s: int((table["split"] == s).sum()) for s in SPLITS},
        "confident_matches": {s: int(((table["split"] == s) & confident).sum()) for s in SPLITS},
        "correlation_quantiles": {
            q: round(float(table["correlation"].quantile(q)), 4) for q in (0.001, 0.01, 0.5)
        },
        "test_rows_from_nih_test": int((test["nih_split"] == "test").sum()),
        "patients": {s: len(p) for s, p in patients.items()},
        "patients_in_train_and_test": len(patients["train"] & patients["test"]),
        "patients_in_val_and_test": len(patients["val"] & patients["test"]),
        "runs": {},
    }
    ids = test.set_index("id")["patient_id"]
    for run in sorted(p for p in RUNS.iterdir() if (p / "predictions_test.csv").exists()):
        predictions = pd.read_csv(run / "predictions_test.csv")
        target = predictions[[f"target/{n}" for n in LABELS]].to_numpy()
        score = predictions[[f"prob/{n}" for n in LABELS]].to_numpy()
        groups = [ids.get(i) or i for i in predictions["id"]]
        statistic = macro_auroc(target, score)
        image = bootstrap(statistic, len(predictions))["auroc"]
        patient = bootstrap(statistic, len(predictions), groups)["auroc"]
        recorded = json.loads((run / "metrics.json").read_text())["test"]["auroc"]
        summary["runs"][run.name] = {
            "recorded": [recorded["value"], recorded["lower"], recorded["upper"]],
            "image_level": [image.value, image.lower, image.upper],
            "patient_level": [patient.value, patient.lower, patient.upper],
        }
    text = json.dumps(summary, indent=2)
    (OUT / "patient_ids.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
