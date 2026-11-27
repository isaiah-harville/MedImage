import pickle
from typing import Any

import numpy as np
import pytest

from opendiagnostics.data import MedMNISTDataModule


def _fake_archive(root, flag="dermamnist", size=64, counts=(6, 2, 3)):
    arrays: dict[str, Any] = {}
    for split, n in zip(("train", "val", "test"), counts, strict=True):
        arrays[f"{split}_images"] = np.full((n, size, size, 3), 128, np.uint8)
        arrays[f"{split}_labels"] = np.arange(n).reshape(-1, 1) % 7
    np.savez(root / f"{flag}_{size}.npz", **arrays)


def test_medmnist_uses_the_official_split_and_package_metadata(tmp_path):
    _fake_archive(tmp_path)
    data = MedMNISTDataModule(flag="dermamnist", size=64, root=str(tmp_path), download=False)
    data.setup()
    assert {k: len(v) for k, v in data.splits.items()} == {"train": 6, "val": 2, "test": 3}
    assert data.info.label_mode == "multiclass"
    assert data.info.num_classes == 7
    assert data.info.license == "CC BY-NC 4.0"
    assert not data.info.patient_ids_available
    batch = next(iter(data.loader("test")))
    assert batch["image"].shape == (3, 3, 64, 64)
    assert batch["image"].max().item() == pytest.approx(128 / 255)
    assert batch["target"].tolist() == [0, 1, 2]
    assert batch["id"] == ["test-0", "test-1", "test-2"]


def test_medmnist_3d_volumes_batch_channel_first_in_unit_range(tmp_path):
    arrays: dict[str, Any] = {}
    for split, n in zip(("train", "val", "test"), (4, 2, 3), strict=True):
        volumes = np.zeros((n, 8, 8, 8), np.uint8)
        volumes[:, 0] = 255
        arrays[f"{split}_images"] = volumes
        arrays[f"{split}_labels"] = np.arange(n).reshape(-1, 1) % 2
    np.savez(tmp_path / "nodulemnist3d_64.npz", **arrays)
    data = MedMNISTDataModule(flag="nodulemnist3d", size=64, root=str(tmp_path), download=False)
    data.setup()
    image = next(iter(data.loader("test")))["image"]
    assert image.shape == (3, 1, 8, 8, 8)
    assert image[:, :, 0].min().item() == 1.0
    assert image[:, :, 1:].max().item() == 0.0


def test_medmnist_dataset_pickles_without_its_images(tmp_path):
    _fake_archive(tmp_path)
    data = MedMNISTDataModule(flag="dermamnist", size=64, root=str(tmp_path), download=False)
    data.setup()
    data.source.dataset[0]
    assert len(pickle.dumps(data.source.dataset)) < 10_000


def test_medmnist_without_archive_or_download_is_an_error(tmp_path):
    data = MedMNISTDataModule(flag="dermamnist", size=64, root=str(tmp_path), download=False)
    with pytest.raises(FileNotFoundError, match="set download: true"):
        data.setup()
