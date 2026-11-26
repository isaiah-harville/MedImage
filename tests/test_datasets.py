import csv
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from opendiagnostics.datasets import medmnist_info
from opendiagnostics.registry import DATASETS


def test_medmnist_reads_verified_npz(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    sizes = {"train": 6, "val": 2, "test": 3}
    images = {s: rng.integers(0, 256, size=(n, 28, 28), dtype=np.uint8) for s, n in sizes.items()}
    labels = {s: rng.integers(0, 2, size=(n, 1)) for s, n in sizes.items()}
    path = tmp_path / "pneumoniamnist.npz"
    np.savez(
        path,
        train_images=images["train"],
        train_labels=labels["train"],
        val_images=images["val"],
        val_labels=labels["val"],
        test_images=images["test"],
        test_labels=labels["test"],
    )
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    subset = {**medmnist_info.SUBSETS["pneumoniamnist"], "md5": {"28": md5}}
    monkeypatch.setitem(medmnist_info.SUBSETS, "pneumoniamnist", subset)

    source = DATASETS.build("medmnist", subset="pneumoniamnist", root=str(tmp_path))
    assert len(source) == 11
    assert {k: len(v) for k, v in source.predefined_splits().items()} == {
        "train": 6,
        "val": 2,
        "test": 3,
    }
    sample = source[7]
    assert sample["image"].shape == (1, 28, 28)
    assert 0 <= sample["image"].min() <= sample["image"].max() <= 1
    assert sample["label"].item() == labels["val"][1, 0]
    assert source.info.label_mode == "binary"
    assert source.info.license == "CC BY 4.0"
    assert not source.info.patient_ids_available


def test_medmnist_rejects_tampered_file(tmp_path):
    (tmp_path / "pneumoniamnist.npz").write_bytes(b"not the real file")
    with pytest.raises(OSError, match="MD5"):
        DATASETS.build("medmnist", subset="pneumoniamnist", root=str(tmp_path))


def test_medmnist_metadata_covers_2d_and_3d():
    assert len(medmnist_info.SUBSETS) == 18
    assert set(medmnist_info.SUBSETS["nodulemnist3d"]["md5"]) == {"28", "64"}


@pytest.mark.network
def test_medmnist_download(tmp_path):
    source = DATASETS.build("medmnist", subset="breastmnist", root=str(tmp_path), download=True)
    assert len(source) == 780


def _write_png(path, value):
    Image.fromarray(np.full((8, 8), value, np.uint8)).save(path)


def test_csv_classification_multilabel(tmp_path):
    rows = [
        {"path": "a.png", "patient_id": "p1", "effusion": "1", "edema": "", "split": "train"},
        {"path": "b.png", "patient_id": "p1", "effusion": "0", "edema": "-1", "split": "train"},
        {"path": "c.png", "patient_id": "p2", "effusion": "1", "edema": "1", "split": "test"},
    ]
    for row in rows:
        _write_png(tmp_path / row["path"], 100)
    with (tmp_path / "labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="uncertain"):
        DATASETS.build(
            "csv_classification",
            manifest=str(tmp_path / "labels.csv"),
            label_mode="multilabel",
            label_columns=["effusion", "edema"],
        )
    source = DATASETS.build(
        "csv_classification",
        manifest=str(tmp_path / "labels.csv"),
        label_mode="multilabel",
        label_columns=["effusion", "edema"],
        uncertain_label=0.0,
        split_column="split",
    )
    assert source.labels == [[1, 0], [0, 0], [1, 1]]
    assert source.patient_ids == ["p1", "p1", "p2"]
    assert source.predefined_splits() == {"train": [0, 1], "val": [], "test": [2]}
    assert source[2]["label"].tolist() == [1.0, 1.0]
    assert source[0]["image"].shape == (1, 8, 8)


def test_coco_detection_converts_boxes(tmp_path):
    _write_png(tmp_path / "x.png", 0)
    coco = {
        "images": [{"id": 1, "file_name": "x.png", "patient_id": "p9"}],
        "categories": [{"id": 7, "name": "nodule"}, {"id": 3, "name": "mass"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": [1, 2, 3, 4]},
            {"id": 2, "image_id": 1, "category_id": 3, "bbox": [0, 0, 2, 2], "iscrowd": 1},
        ],
    }
    (tmp_path / "ann.json").write_text(json.dumps(coco))
    source = DATASETS.build("coco_detection", annotations=str(tmp_path / "ann.json"))
    sample = source[0]
    assert source.info.class_names == ("mass", "nodule")
    assert sample["boxes"].tolist() == [[1, 2, 4, 6]]
    assert sample["labels"].tolist() == [1]
    assert sample["patient_id"] == "p9"


def test_coco_requires_patient_ids(tmp_path):
    coco = {"images": [{"id": 1, "file_name": "x.png"}], "categories": [], "annotations": []}
    (tmp_path / "ann.json").write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="allow_missing_patient"):
        DATASETS.build("coco_detection", annotations=str(tmp_path / "ann.json"))


def test_synthetic_data_is_deterministic():
    a = DATASETS.build("synthetic_detection", num_samples=4)
    b = DATASETS.build("synthetic_detection", num_samples=4)
    for i in range(4):
        assert (a[i]["image"] == b[i]["image"]).all()
        assert (a[i]["boxes"] == b[i]["boxes"]).all()
