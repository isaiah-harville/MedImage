import csv
import pickle

import numpy as np
import pytest
from PIL import Image
from xpt import expect

from opendiagnostics.data import LeakageError
from opendiagnostics.data.manifest import CSVImageDataModule

FINDINGS = ["Effusion", "Mass", "Nodule"]


def test_images_come_out_channels_first_with_height_before_width(tmp_path):
    rows = []
    for i in range(6):
        pixels = np.zeros((6, 4, 3), np.uint8)
        pixels[0, :, 0] = 255
        Image.fromarray(pixels).save(tmp_path / f"{i}.png")
        split = ["train", "val", "test"][i % 3]
        rows.append(
            {"path": f"{i}.png", "patient_id": f"p{i}", "label": "ab"[i % 2], "split": split}
        )
    with (tmp_path / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    data = CSVImageDataModule(
        manifest=str(tmp_path / "manifest.csv"), split_column="split", in_channels=3, batch_size=2
    )
    data.setup()
    image = next(iter(data.train_dataloader()))["image"][0]
    assert tuple(image.shape) == (3, 6, 4)
    assert image[0, 0].min() > 0.99
    assert image[0, 1:].max() == 0


def _manifest(root, rows, image_mode="L"):
    (root / "images").mkdir(exist_ok=True)
    for path, *_ in rows:
        Image.new(image_mode, (12, 10), color=(90,) * len(image_mode)).save(root / path)
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "patient_id", "label", "split"])
        writer.writerows(rows)
    return manifest


def _chest_rows(patients=12, images=3):
    findings = ["", "Effusion", "Mass|Nodule", "Effusion|Mass|Nodule"]
    rows = []
    for p in range(patients):
        split = "test" if p % 4 == 0 else "train"
        for n in range(images):
            label = findings[(p + n) % len(findings)]
            rows.append([f"images/{p:03d}_{n}.png", str(p), label, split])
    return rows


def test_multilabel_manifest_gives_multi_hot_targets_and_channel_first_images(tmp_path):
    manifest = _manifest(tmp_path, _chest_rows(patients=4, images=1))
    data = CSVImageDataModule(
        str(manifest), split_column="split", class_names=FINDINGS, label_separator="|"
    )
    data.setup()
    assert data.info.label_mode == "multilabel"
    assert data.info.class_names == tuple(FINDINGS)
    batch = next(iter(data.loader("train")))
    assert tuple(batch["image"].shape) == (3, 1, 10, 12)
    assert batch["target"].dtype.is_floating_point
    assert batch["target"].tolist() == [[1, 0, 0], [0, 1, 1], [1, 1, 1]]
    test = data.loader("test").dataset[0]
    assert tuple(test["image"].shape) == (1, 10, 12)
    assert test["target"].tolist() == [0, 0, 0]


def test_kfold_validation_on_a_multilabel_manifest_never_splits_a_patient(tmp_path):
    rows = _chest_rows()
    manifest = _manifest(tmp_path, rows)
    data = CSVImageDataModule(
        str(manifest),
        split_column="split",
        class_names=FINDINGS,
        label_separator="|",
        split="kfold",
        folds=3,
        stratify=False,
    )
    data.setup()
    owner = {}
    for name, indices in data.splits.items():
        for i in indices:
            assert owner.setdefault(rows[i][1], name) == name
    assert {rows[i][1] for i in data.splits["test"]} == {"0", "4", "8"}
    assert data.splits["val"]
    assert sorted(i for s in data.splits.values() for i in s) == list(range(len(rows)))


def test_a_patient_in_two_official_splits_is_refused(tmp_path):
    rows = [["images/a.png", "p1", "Mass", "train"], ["images/b.png", "p1", "", "test"]]
    manifest = _manifest(tmp_path, rows)
    data = CSVImageDataModule(
        str(manifest), split_column="split", class_names=FINDINGS, label_separator="|"
    )
    with pytest.raises(LeakageError, match="patient 'p1'"):
        data.setup()


def test_labels_outside_class_names_are_an_error(tmp_path):
    rows = [["images/a.png", "p1", "Mass|Hernia|No Finding", "train"]]
    manifest = _manifest(tmp_path, rows)
    with pytest.raises(ValueError, match="not in class_names") as error:
        CSVImageDataModule(str(manifest), class_names=FINDINGS, label_separator="|")
    expect(
        str(error.value).replace(str(manifest), "manifest.csv"),
        "manifest.csv has labels that are not in class_names: ['Hernia', 'No Finding']. Add them to class_names, or leave a row's label empty when it has none.",
    )


def test_label_separator_without_class_names_is_an_error(tmp_path):
    manifest = _manifest(tmp_path, [["images/a.png", "p1", "Mass", "train"]])
    with pytest.raises(ValueError, match="needs class_names"):
        CSVImageDataModule(str(manifest), label_separator="|")


def test_single_label_manifests_stratify_on_class_indices(tmp_path):
    rows = [
        ["images/a.png", "p1", "PNEUMONIA", "train"],
        ["images/b.png", "p2", "NORMAL", "train"],
        ["images/c.png", "p3", "NORMAL", "test"],
    ]
    data = CSVImageDataModule(str(_manifest(tmp_path, rows)), split_column="split")
    data.setup()
    assert data.info.label_mode == "binary"
    assert data.info.class_names == ("NORMAL", "PNEUMONIA")
    assert data.source.strata == [1, 0, 0]
    assert next(iter(data.loader("train")))["target"].tolist() == [1, 0]


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_image_mode_reads_colour_files_as_one_channel(tmp_path, mode):
    rows = [["images/a.png", "p1", "NORMAL", "train"], ["images/b.png", "p2", "OTHER", "test"]]
    manifest = _manifest(tmp_path, rows, image_mode=mode)
    data = CSVImageDataModule(str(manifest), split_column="split", image_mode="L")
    data.setup()
    batch = next(iter(data.loader("train")))
    assert tuple(batch["image"].shape) == (1, 1, 10, 12)
    loaded = pickle.loads(pickle.dumps(data.source.dataset))
    assert tuple(loaded[1][0].shape) == (1, 10, 12)
