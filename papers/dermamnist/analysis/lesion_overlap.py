"""Score the DermaMNIST run on test images whose lesion is, or is not, also in training.

    uv run python papers/dermamnist/analysis/lesion_overlap.py SPLIT_INFO METADATA

SPLIT_INFO is dermamnist_split_info.csv and METADATA is CSV_files/combined_metadata_corrected.csv,
both from github.com/kakumarabhishek/Corrected-Skin-Image-Datasets (DermaMNIST/DermaMNIST_Analysis).
They map each DermaMNIST index to its HAM10000 image and lesion. The script first checks that
the mapped diagnosis equals the label in predictions_test.csv for every test image, then
bootstraps AUROC, accuracy, and balanced accuracy on each subset with the pipeline's code.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from opendiagnostics.metrics import classification as cm
from opendiagnostics.metrics.bootstrap import bootstrap
from opendiagnostics.metrics.compare import read_predictions

RUN = (
    Path(__file__).resolve().parents[3] / "models/dermamnist-convnext/runs/20261126-141911-ab39eb97"
)
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def score(target: np.ndarray, prob: np.ndarray) -> dict:
    def statistic(index: np.ndarray) -> dict[str, float]:
        t, p = target[index], prob[index]
        predicted = p.argmax(1)
        return {
            "auroc": cm.nanmean(cm.per_class(t, p, cm.auroc)),
            "accuracy": float(np.mean(predicted == t)),
            "balanced_accuracy": cm.balanced_accuracy(t, predicted, len(CLASSES)),
        }

    return bootstrap(statistic, len(target), replicates=1000, seed=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("split_info", type=Path)
    parser.add_argument("metadata", type=Path)
    args = parser.parse_args()

    index_to_image = {(r["split"], int(r["index"])): r["image_id"] for r in rows(args.split_info)}
    meta = {r["image_id"]: r for r in rows(args.metadata)}
    split_of = {image: split for (split, _), image in index_to_image.items()}
    lesions: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for image, split in split_of.items():
        lesions[split].add(meta[image]["lesion_id"])

    pred = read_predictions(RUN / "predictions_test.csv")
    images = [index_to_image[("test", int(i.removeprefix("test-")))] for i in pred.ids]
    mapped = np.array([CLASSES.index(meta[image]["dx"]) for image in images])
    mismatched = int(np.sum(mapped != pred.target))
    if mismatched:
        raise SystemExit(
            f"{mismatched} test labels differ from the mapped diagnosis; the mapping is wrong"
        )

    lesion = [meta[image]["lesion_id"] for image in images]
    in_train = np.array([le in lesions["train"] for le in lesion])
    in_train_or_val = np.array([le in lesions["train"] | lesions["val"] for le in lesion])
    corrected_test = np.array([meta[image]["split"] == "test" for image in images])
    print(f"test images: {len(images)}, labels match the mapping: all {len(images)}")
    print(
        f"test lesions: {len(set(lesion))}, test images per lesion up to {max(lesion.count(x) for x in set(lesion))}"
    )
    subsets = {
        "all test images": np.ones(len(images), bool),
        "lesion also in train": in_train,
        "lesion not in train": ~in_train,
        "lesion in neither train nor val": ~in_train_or_val,
        "DermaMNIST-C test images": corrected_test,
    }
    for name, mask in subsets.items():
        result = score(pred.target[mask], pred.prob[mask])
        counts = np.bincount(pred.target[mask], minlength=len(CLASSES))
        print(
            f"\n{name}: n={int(mask.sum())}, per class {dict(zip(CLASSES, counts.tolist(), strict=True))}"
        )
        for key, estimate in result.items():
            print(f"  {key}: {estimate}")


if __name__ == "__main__":
    main()
