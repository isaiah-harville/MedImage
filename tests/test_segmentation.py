import csv
import json
import pickle
import shutil

import nibabel as nib
import numpy as np
import pytest
import torch
from conftest import fit_smoke
from monai.data import MetaTensor
from monai.networks.nets import UNet
from monai.transforms import Spacingd
from sklearn.metrics import f1_score
from torch import nn

from opendiagnostics.cli import evaluate_run
from opendiagnostics.data import DatasetInfo, NiftiSegmentationDataModule
from opendiagnostics.metrics.segmentation import case_metrics, dice, hd95
from opendiagnostics.modules import SegmentationModule
from opendiagnostics.segmentation_examples import choose_segmentation

NAMES = ["background", "kidney", "tumor"]


def _cube(shape, start, size):
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[tuple(slice(s, s + size) for s in start)] = True
    return mask


def test_dice_matches_scikit_learn_f1_on_flattened_masks():
    rng = np.random.default_rng(0)
    for _ in range(5):
        a, b = rng.random((2, 12, 10, 8)) > 0.6
        expected = f1_score(b.ravel(), a.ravel())
        assert dice(torch.from_numpy(a), torch.from_numpy(b)) == pytest.approx(expected, abs=1e-6)


def test_dice_scores_a_region_missing_from_both_as_one_and_from_one_as_zero():
    empty = torch.zeros(6, 6, 6, dtype=torch.bool)
    full = _cube((6, 6, 6), (1, 1, 1), 3)
    assert dice(empty, empty) == 1.0
    assert dice(full, empty) == 0.0
    assert np.isnan(dice(full, empty, ignore_empty=True))


def test_hd95_of_a_shifted_cube_is_the_shift_in_millimetres():
    reference = _cube((24, 24, 24), (6, 6, 6), 8)
    shifted = _cube((24, 24, 24), (6, 6, 9), 8)
    assert hd95(shifted, reference, (1.0, 1.0, 2.0)) == pytest.approx(6.0)
    assert np.isnan(hd95(torch.zeros_like(reference), reference, (1.0, 1.0, 1.0)))


def test_hd95_is_the_same_cropped_to_the_masks_as_on_the_whole_volume():
    from monai.metrics import compute_hausdorff_distance

    rng = np.random.default_rng(1)
    a = torch.from_numpy(rng.random((20, 18, 16)) > 0.97)
    b = torch.from_numpy(rng.random((20, 18, 16)) > 0.97)
    whole = compute_hausdorff_distance(
        a[None, None].float(), b[None, None].float(), percentile=95, spacing=[1.0, 2.0, 3.0]
    )
    assert hd95(a, b, (1.0, 2.0, 3.0)) == pytest.approx(float(whole))


def test_regions_union_label_values_like_kits23_evaluation_classes():
    target = torch.zeros(10, 10, 10, dtype=torch.long)
    target[2:8, 2:8, 2:8] = 1
    target[3:5, 3:5, 3:5] = 2
    pred = target.clone()
    pred[3:5, 3:5, 3:5] = 1
    regions = {"kidney_and_masses": [1, 2], "tumor": [2]}
    metrics = case_metrics(pred, target, regions, (1.0, 1.0, 1.0))
    assert metrics["dice"].tolist() == [1.0, 0.0]
    assert np.isnan(metrics["hd95"][1])


def _module(**kwargs):
    info = DatasetInfo(
        label_mode="segmentation", class_names=tuple(NAMES), in_channels=1, spatial_dims=3
    )
    network = UNet(3, 1, len(NAMES), channels=(4, 8), strides=(2,))
    return SegmentationModule(network, patch_size=[8, 8, 8], info=info, **kwargs)


def _batch(target, image_affine=None):
    n = target.shape[0]
    image = target.float() / 2
    return {
        "image": image,
        "target": target.to(torch.int16),
        "image_affine": image_affine if image_affine is not None else torch.eye(4).repeat(n, 1, 1),
        "target_affine": torch.eye(4, dtype=torch.float64).repeat(n, 1, 1),
        "spacing": [(1.0, 1.0, 1.0)] * n,
        "id": [f"c{i}" for i in range(n)],
        "patient_id": [f"p{i // 2}" for i in range(n)],
    }


def _target(n=4, size=12):
    target = torch.zeros(n, 1, size, size, size, dtype=torch.long)
    target[:, :, 2:8, 2:8, 2:8] = 1
    target[:, :, 4:6, 4:6, 4:6] = 2
    return target


def test_scores_have_per_region_dice_and_a_patient_bootstrap():
    model = _module().eval()
    predictions = model.predict_batch(_batch(_target()))
    bootstrap = {"replicates": 50, "confidence": 0.95, "unit": "patient", "seed": 0}
    scores = model.score(predictions, bootstrap=bootstrap)
    assert {"dice", "dice/kidney", "dice/tumor", "hd95", "hd95/kidney"} <= set(scores)
    assert scores["dice"]["unit"] == "patient"
    assert scores["n_cases"] == 4
    assert scores["n_patients"] == 2
    row = model.prediction_rows(predictions)[0]
    assert set(row) == {
        "id",
        "patient_id",
        "dice/kidney",
        "dice/tumor",
        "hd95/kidney",
        "hd95/tumor",
    }


def test_validation_skips_distances():
    model = _module().eval()
    model.validation_step(_batch(_target()), 0)
    scores = model.score(model._val_predictions)
    assert "dice" in scores
    assert "hd95" not in scores


def test_prediction_on_a_coarser_grid_is_scored_on_the_label_grid():
    model = _module().eval()
    target = _target(n=1, size=16)
    batch = _batch(target)
    coarse = Spacingd(["image"], pixdim=(2.0, 2.0, 2.0))(
        {"image": MetaTensor(batch["image"][0], affine=torch.eye(4))}
    )["image"]
    assert isinstance(coarse, MetaTensor)
    batch["image"] = torch.as_tensor(coarse)[None]
    batch["image_affine"] = torch.as_tensor(coarse.affine)[None]
    labels = model.segment(
        batch["image"][0], batch["image_affine"][0], batch["target_affine"][0], (16, 16, 16)
    )
    assert labels.shape == (16, 16, 16)
    assert set(labels.unique().tolist()) <= {0, 1, 2}


def test_network_with_wrong_output_channels_is_explained():
    model = _module()
    model.network = UNet(3, 1, 2, channels=(4, 8), strides=(2,))
    with pytest.raises(ValueError, match="out_channels to 3"):
        model.training_step(_batch(_target(n=1)), 0)


def test_regions_outside_the_label_values_are_refused():
    with pytest.raises(ValueError, match=r"label values \[5\]"):
        _module(regions={"mass": [2, 5]})


def test_segmentation_module_refuses_a_classification_dataset():
    info = DatasetInfo(label_mode="multiclass", class_names=("a", "b"), in_channels=1)
    with pytest.raises(ValueError, match="segmentation data module"):
        SegmentationModule(nn.Identity(), patch_size=[8, 8], info=info)


def _nifti(path, array, spacing=(0.8, 0.8, 2.5)):
    nib.save(nib.Nifti1Image(array, np.diag([*spacing, 1.0])), path)


def _manifest(root, cases=6, channels=2):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(cases):
        case = root / f"case_{i}"
        case.mkdir()
        label = np.zeros((20, 22, 10), np.uint8)
        label[5:15, 5:15, 2:8] = 1 + i % 2
        _nifti(case / "label.nii.gz", label)
        images = {}
        for c in range(channels):
            _nifti(case / f"image_{c}.nii.gz", rng.random((20, 22, 10)).astype(np.float32))
            images[f"image_{c}"] = f"case_{i}/image_{c}.nii.gz"
        split = ("train", "train", "val", "val", "test", "test")[i]
        rows.append(
            {"id": f"case_{i}", **images, "label": f"case_{i}/label.nii.gz"}
            | {"patient_id": f"p{i}", "split": split}
        )
    with (root / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root / "manifest.csv"


def _nifti_module(manifest, **kwargs):
    return NiftiSegmentationDataModule(
        manifest=str(manifest),
        class_names=NAMES,
        image_columns=["image_0", "image_1"],
        **kwargs,
    )


def test_nifti_manifest_stacks_image_columns_and_keeps_the_official_split(tmp_path):
    data = _nifti_module(_manifest(tmp_path))
    data.setup()
    assert {k: len(v) for k, v in data.splits.items()} == {"train": 2, "val": 2, "test": 2}
    assert data.info.in_channels == 2
    batch = next(iter(data.loader("test")))
    assert batch["image"].shape == (1, 2, 20, 22, 10)
    assert batch["target"].shape == (1, 1, 20, 22, 10)
    assert batch["spacing"] == [pytest.approx((0.8, 0.8, 2.5))]
    assert batch["id"] == ["case_4"]
    assert len(pickle.dumps(data.dataset("train"))) < 20_000


def test_multi_patch_crops_all_go_into_the_training_batch(tmp_path):
    from monai.transforms import RandCropByPosNegLabeld

    crop = RandCropByPosNegLabeld(["image", "label"], "label", (8, 8, 4), num_samples=3)
    data = _nifti_module(_manifest(tmp_path), train_transforms=crop, batch_size=2)
    data.setup()
    batch = next(iter(data.loader("train", train=True)))
    assert batch["image"].shape == (6, 2, 8, 8, 4)
    assert len(batch["id"]) == 6


def test_cache_dir_serves_the_same_cases_as_loading_from_disk(tmp_path):
    (tmp_path / "data").mkdir()
    manifest = _manifest(tmp_path / "data")
    spacing = Spacingd(["image"], pixdim=(1.6, 1.6, 5.0))
    plain = _nifti_module(manifest, eval_transforms=spacing)
    cached = _nifti_module(manifest, eval_transforms=spacing, cache_dir=str(tmp_path / "cache"))
    plain.setup()
    cached.setup()
    for _ in range(2):
        a, b = next(iter(plain.loader("val"))), next(iter(cached.loader("val")))
        assert torch.equal(a["image"], b["image"])
        assert torch.equal(a["image_affine"], b["image_affine"])
    assert any((tmp_path / "cache").iterdir())


def test_missing_manifest_points_at_the_download_scripts(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts/datasets"):
        _nifti_module(tmp_path / "missing.csv")


def test_choose_segmentation_alternates_best_and_worst_cases():
    rows = [
        {"id": "a", "dice/x": "0.9", "dice/y": "0.7"},
        {"id": "b", "dice/x": "0.1", "dice/y": "nan"},
        {"id": "c", "dice/x": "0.5", "dice/y": "0.5"},
        {"id": "d", "dice/x": "nan", "dice/y": "nan"},
    ]
    assert [p["id"] for p in choose_segmentation(rows, 3)] == ["a", "b", "c"]


def test_smoke_segmentation_trains_evaluates_and_reproduces(tmp_path):
    run_dir = fit_smoke("segmentation", tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    for key in ("dice", "dice/class_1", "hd95"):
        test = metrics["test"][key]
        assert test["lower"] <= test["value"] <= test["upper"]
        assert test["unit"] == "patient"
    manifest = json.loads((run_dir / "examples" / "examples.json").read_text())
    assert manifest
    assert all((run_dir / "examples" / e["overlay"]).is_file() for e in manifest)
    copy = tmp_path / "copy"
    shutil.copytree(run_dir, copy)
    assert evaluate_run(copy)["test"] == metrics["test"]


def test_smoke_segmentation_runs_are_bitwise_reproducible_on_cpu(tmp_path):
    first = fit_smoke("segmentation", tmp_path / "a")
    second = fit_smoke("segmentation", tmp_path / "b")
    assert (first / "metrics.json").read_text() == (second / "metrics.json").read_text()
