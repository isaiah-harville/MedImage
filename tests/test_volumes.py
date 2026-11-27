import json
from typing import Any

import nibabel as nib
import numpy as np
import pytest
import torch
from monai.networks.nets import ResNet
from pydantic import ValidationError

from opendiagnostics.data import LeakageError, VolumeManifestDataModule
from opendiagnostics.data.volumes import voxel_boxes_to_world
from opendiagnostics.modules import VolumeDetector

SHAPE = (20, 24, 16)
CUBE = (slice(5, 9), slice(6, 12), slice(3, 7))
# Stored the way SimpleITK writes an LPS-aligned scan: x and y run right-to-left and
# anterior-to-posterior, so the RAS affine negates them.
AFFINE = np.array([[-1.5, 0, 0, 30], [0, -1.5, 0, 40], [0, 0, 2, 10], [0, 0, 0, 1]])
# World LPS of the cube's outer voxel faces, worked by hand: x_lps = -x_ras = 1.5 i - 30,
# y_lps = 1.5 j - 40 and z = 2 k + 10 at voxel centers, so the faces of voxels 5..8, 6..11
# and 3..6 sit at i = 4.5 and 8.5, j = 5.5 and 11.5, k = 2.5 and 6.5.
WORLD_BOX = [-23.25, -31.75, 15.0, -17.25, -22.75, 23.0]


def _volume(path, value=1.0):
    data = np.zeros(SHAPE, np.float32)
    data[CUBE] = value
    nib.save(nib.Nifti1Image(data, AFFINE), path)


def _manifest(tmp_path, volumes, **extra):
    spec = {
        "dataset": "fixture",
        "license": "generated",
        "class_names": ["lesion"],
        "coordinates": "world-lps-mm",
        "volumes": volumes,
    } | extra
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(spec))
    return path


def _entry(name, patient, fold=None, channels=1, boxes=(WORLD_BOX,)):
    return {
        "id": name,
        "patient_id": patient,
        "images": [f"{name}-{c}.nii.gz" for c in range(channels)],
        "fold": fold,
        "boxes": list(boxes),
        "labels": [0] * len(boxes),
    }


@pytest.fixture
def volumes(tmp_path):
    entries = [_entry(f"v{n}", f"p{n}", fold=n % 3) for n in range(6)]
    for entry in entries:
        _volume(tmp_path / entry["images"][0])
    return _manifest(tmp_path, entries)


def _module(manifest, **kwargs):
    options: dict[str, Any] = {"split": "predefined", "test_fold": 0, "patch_size": SHAPE}
    data = VolumeManifestDataModule(str(manifest), intensity_range=(0.0, 1.0), **(options | kwargs))
    data.setup()
    return data


def _first(loader):
    return next(iter(loader))


def _box_mask(box):
    mask = torch.zeros(SHAPE, dtype=torch.bool)
    lo = [round(float(v)) for v in box[:3]]
    hi = [round(float(v)) for v in box[3:]]
    mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True
    return mask


def test_world_boxes_land_on_the_lesion_voxels(volumes):
    batch = _first(_module(volumes, crops_per_volume=1).loader("train", train=True))
    image, box = batch["image"][0, 0], batch["target"][0]["boxes"][0]
    inside = _box_mask(box)
    assert inside.sum() == 4 * 6 * 4
    assert image[inside].min() == 1.0
    assert image[~inside].max() == 0.0


def test_predictions_map_back_to_the_manifest_world_boxes(volumes):
    data = _module(volumes, crops_per_volume=1)
    voxel = _first(data.loader("train", train=True))["target"][0]["boxes"]
    affine = _first(data.loader("val"))["affine"][0]
    world = voxel_boxes_to_world(voxel, affine)
    np.testing.assert_allclose(world.numpy(), [WORLD_BOX], atol=1e-6)


def test_resampling_keeps_the_box_on_the_lesion(volumes):
    data = _module(volumes, crops_per_volume=1, spacing=(1.0, 1.0, 1.0), patch_size=(24, 24, 24))
    batch = _first(data.loader("train", train=True))
    image, (box,) = batch["image"][0, 0], batch["target"][0]["boxes"]
    lo = [round(float(v)) for v in box[:3]]
    hi = [round(float(v)) for v in box[3:]]
    assert [h - lo_ for lo_, h in zip(lo, hi, strict=True)] == [6, 9, 8]
    assert image[lo[0] + 1 : hi[0] - 1, lo[1] + 1 : hi[1] - 1, lo[2] + 1 : hi[2] - 1].min() > 0.9


def test_evaluation_serves_whole_volumes_with_world_truth(volumes):
    batch = _first(_module(volumes).loader("test"))
    assert batch["image"].shape == (1, 1, *SHAPE)
    np.testing.assert_allclose(batch["target"][0]["world_boxes"].numpy(), [WORLD_BOX])
    assert batch["target"][0]["world_ignore"].shape == (0, 6)


def test_training_serves_several_crops_per_volume(volumes):
    batch = _first(_module(volumes, crops_per_volume=3, batch_size=2).loader("train", train=True))
    assert batch["image"].shape[0] == 6


def test_folds_give_test_validation_and_training_splits(volumes):
    data = _module(volumes, test_fold=1)
    ids = {k: [data.source.sample_ids[i] for i in v] for k, v in data.splits.items()}
    assert ids == {"train": ["v0", "v3"], "val": ["v2", "v5"], "test": ["v1", "v4"]}
    data = _module(volumes, test_fold=1, val_fold=0)
    assert [data.source.sample_ids[i] for i in data.splits["val"]] == ["v0", "v3"]


def test_a_patient_in_two_folds_is_refused(tmp_path):
    entries = [_entry("a", "same", fold=0), _entry("b", "same", fold=1), _entry("c", "c", fold=2)]
    with pytest.raises(LeakageError, match="same"):
        _module(_manifest(tmp_path, entries))


def test_a_fold_manifest_needs_a_test_fold(volumes):
    with pytest.raises(ValueError, match="set test_fold"):
        _module(volumes, test_fold=None)


def test_several_images_per_volume_are_channels(tmp_path):
    entries = [_entry(f"v{n}", f"p{n}", channels=2) for n in range(4)]
    for entry in entries:
        for n, image in enumerate(entry["images"]):
            _volume(tmp_path / image, value=n + 1.0)
    data = _module(_manifest(tmp_path, entries), split="patient", test_fold=None)
    assert data.info.in_channels == 2
    batch = _first(data.loader("test"))
    assert batch["image"].shape[1] == 2


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"labels": []}, "1 boxes but 0 labels"),
        ({"boxes": [[0, 0, 0, 0, 1, 1]]}, "needs x1 < x2"),
        ({"labels": [3]}, "outside 0..0"),
        ({"spacing": 1}, "Extra inputs"),
    ],
)
def test_manifest_errors_name_the_problem(tmp_path, change, message):
    path = _manifest(tmp_path, [_entry("v", "p") | change])
    with pytest.raises(ValidationError, match=message):
        VolumeManifestDataModule(str(path))


def test_a_missing_manifest_points_to_the_download_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts/datasets"):
        VolumeManifestDataModule(str(tmp_path / "manifest.json"))


def test_eval_transforms_are_refused(volumes):
    with pytest.raises(ValueError, match="eval_transforms"):
        VolumeManifestDataModule(str(volumes), eval_transforms=lambda d: d)


def test_detector_refuses_a_backbone_with_the_wrong_input_channels(volumes):
    data = VolumeManifestDataModule(str(volumes))
    backbone = ResNet("basic", [1, 1, 1, 1], [8, 16, 32, 64], spatial_dims=3, n_input_channels=2)
    with pytest.raises(ValueError, match="n_input_channels to 1"):
        VolumeDetector(backbone, data.info)
