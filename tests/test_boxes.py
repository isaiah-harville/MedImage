import csv
import json
import pickle

import numpy as np
import pytest
import yaml
from fake_dicom import write_dicom

from opendiagnostics.cli import fit
from opendiagnostics.data import BoxManifestDataModule, LeakageError

FIELDS = ["path", "patient_id", "split", "label", "x1", "y1", "x2", "y2"]


def _manifest(tmp_path, rows):
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(zip(FIELDS, row, strict=True)))
    return path


def _image(tmp_path, name, bright=None, shape=(12, 20)):
    pixels = np.zeros(shape, np.uint16)
    if bright:
        row, col = bright
        pixels[row, col] = 1000
    write_dicom(tmp_path / name, pixels)
    return name


def test_boxes_and_negatives_load_from_dicom_in_pixel_coordinates(tmp_path):
    lesion = _image(tmp_path, "a.dcm", bright=(3, 15))
    negative = _image(tmp_path, "b.dcm")
    manifest = _manifest(
        tmp_path,
        [
            (lesion, "p1", "train", "opacity", 14, 2, 17, 5),
            (lesion, "p1", "train", "opacity", 0, 0, 4, 4),
            (negative, "p2", "train", "", "", "", "", ""),
        ],
    )
    data = BoxManifestDataModule(manifest=str(manifest), split="patient", fractions=(1, 0, 0))
    data.setup()
    image, target = data.source.dataset[0]
    assert image.shape == (1, 12, 20)
    assert np.unravel_index(image[0].argmax(), image[0].shape) == (3, 15)
    assert target["boxes"].tolist() == [[14, 2, 17, 5], [0, 0, 4, 4]]
    assert target["labels"].tolist() == [0, 0]
    batch = next(iter(data.loader("train")))
    served = batch["image"][batch["id"].index(lesion)]
    assert served.shape == (1, 12, 20)
    x1, y1, x2, y2 = batch["target"][batch["id"].index(lesion)]["boxes"][0].int().tolist()
    assert served[0, y1:y2, x1:x2].max().item() == 1.0
    assert served[0].sum().item() == 1.0
    _, empty = data.source.dataset[1]
    assert empty["boxes"].shape == (0, 4)
    assert data.info.label_mode == "detection"
    assert data.info.class_names == ("opacity",)
    assert data.source.strata == ["opacity", ""]
    assert len(pickle.dumps(data.source.dataset)) < 10_000


def _split_rows(tmp_path):
    rows = []
    for i, (patient, split) in enumerate(
        [("p1", "train"), ("p1", "train"), ("p2", "train"), ("p3", "train"), ("p4", "test")]
    ):
        name = _image(tmp_path, f"{i}.dcm")
        rows.append((name, patient, split, "mass", 1, 1, 5, 5))
    return rows


def test_kfold_holds_out_whole_training_patients_and_keeps_the_official_test(tmp_path):
    manifest = _manifest(tmp_path, _split_rows(tmp_path))
    data = BoxManifestDataModule(
        manifest=str(manifest), split_column="split", split="kfold", folds=3, fold=0
    )
    data.setup()
    patients = data.source.patient_ids
    assert data.splits["test"] == [4]
    assert data.splits["val"]
    assert sorted(data.splits["train"] + data.splits["val"]) == [0, 1, 2, 3]
    assert not {patients[i] for i in data.splits["train"]} & {
        patients[i] for i in data.splits["val"]
    }


def test_a_patient_in_two_official_splits_is_refused(tmp_path):
    rows = _split_rows(tmp_path)
    rows[-1] = (rows[-1][0], "p1", *rows[-1][2:])
    data = BoxManifestDataModule(
        manifest=str(_manifest(tmp_path, rows)), split_column="split", split="kfold", folds=2
    )
    with pytest.raises(LeakageError, match="'p1'"):
        data.setup()


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (("a.dcm", "p1", "train", "mass", 5, 1, 2, 4), "x1 < x2"),
        (("a.dcm", "p1", "train", "mass", "left", 1, 2, 4), "must be numbers"),
        (("a.dcm", "p1", "train", "", 1, 1, 2, 4), "needs a label"),
        (("a.dcm", "p1", "holdout", "mass", 1, 1, 2, 4), "split must be one of"),
        (("b.dcm", "p9", "train", "mass", 1, 1, 2, 4), "more than one patient ID"),
    ],
)
def test_malformed_rows_are_refused_with_their_line(tmp_path, row, message):
    good = [("b.dcm", "p1", "train", "mass", 1, 1, 2, 2)]
    manifest = _manifest(tmp_path, [*good, row])
    with pytest.raises(ValueError, match=message):
        BoxManifestDataModule(manifest=str(manifest), split_column="split")


def test_a_label_outside_class_names_is_refused(tmp_path):
    manifest = _manifest(tmp_path, [("a.dcm", "p1", "train", "mass", 1, 1, 2, 2)])
    with pytest.raises(ValueError, match=r"labels \['mass'\] not in class_names"):
        BoxManifestDataModule(manifest=str(manifest), class_names=["calcification"])


@pytest.mark.slow
@pytest.mark.parametrize("project", ["rsna-pneumonia-fasterrcnn", "cbis-ddsm-fasterrcnn"])
def test_detection_project_trains_and_reports_patient_level_intervals(root, tmp_path, project):
    config = yaml.safe_load((root / "models" / project / "config.yaml").read_text())
    names = config["data"]["init_args"]["class_names"]
    rows = []
    for i in range(20):
        name = _image(tmp_path, f"{i}.dcm", bright=(10, 10 + i), shape=(48, 64))
        split = "test" if i >= 15 else "train"
        if project.startswith("rsna") and i % 2:
            rows.append((name, f"p{i}", split, "", "", "", "", ""))
        else:
            rows.append((name, f"p{i}", split, names[i % len(names)], 5, 5, 20 + i, 30))
    config["trainer"] |= {"accelerator": "cpu", "max_epochs": 1, "deterministic": True}
    config["model"]["init_args"]["network"]["init_args"]["weights_backbone"] = None
    config["model"]["init_args"]["network"]["dict_kwargs"] |= {"min_size": 64, "max_size": 64}
    config["data"]["init_args"] |= {
        "manifest": str(_manifest(tmp_path, rows)),
        "num_workers": 0,
        "batch_size": 4,
    }
    config["eval"] = {"bootstrap_samples": 20}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    run_dir = fit(["-c", str(path), f"--output_dir={tmp_path / 'runs'}"])
    metrics = json.loads((run_dir / "metrics.json").read_text())
    headline = metrics["test"]["map_50"]
    assert headline["unit"] == "patient"
    assert headline["replicates"] == 20
    assert headline["lower"] <= headline["value"] <= headline["upper"]
    assert metrics["test"]["n_patients"] == (3 if project.startswith("rsna") else 5)
