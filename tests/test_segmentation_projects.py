import csv
import importlib.util
import sys

import nibabel as nib
import numpy as np
import pytest
import yaml

from opendiagnostics.cli import OpenDiagnosticsCLI


def _recipes(root, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "segmentation_recipes", root / "scripts/segmentation_recipes.py"
    )
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_segmentation_project_configs_match_their_generator(root, monkeypatch):
    recipes = _recipes(root, monkeypatch)
    for name in recipes.RECIPES:
        written = (root / f"models/{name}/config.yaml").read_text()
        assert written == recipes.render(name), (
            f"models/{name}/config.yaml is stale; run `uv run python scripts/segmentation_recipes.py`"
        )


def _fixture(tmp_path, config):
    data = config["data"]["init_args"]
    classes = data["class_names"]
    columns = data["image_columns"]
    channels = data["in_channels"]
    affine = np.diag([0.9, 0.9, 2.0, 1.0])
    rng = np.random.default_rng(0)
    rows = []
    for i, split in enumerate(["train", "train", "val", "test"]):
        label = np.zeros((40, 36, 12), np.uint8)
        for k in range(1, len(classes)):
            label[2 + k % 30 : 8 + k % 30, 4:12, 3:7] = k
        nib.save(nib.Nifti1Image(label, affine), tmp_path / f"l{i}.nii.gz")
        row = {"id": f"c{i}", "label": f"l{i}.nii.gz", "patient_id": f"p{i}", "split": split}
        for c, column in enumerate(columns):
            shape = (40, 36, 12) if channels == len(columns) else (40, 36, 12, channels)
            image = rng.normal(size=shape).astype(np.float32)
            nib.save(nib.Nifti1Image(image, affine), tmp_path / f"i{i}_{c}.nii.gz")
            row[column] = f"i{i}_{c}.nii.gz"
        rows.append(row)
    with (tmp_path / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path / "manifest.csv"


@pytest.mark.parametrize(
    "name",
    [
        "kits23-dynunet",
        "msd-braintumour-dynunet",
        "msd-hippocampus-dynunet",
        "msd-prostate-dynunet",
        "totalsegmentator-dynunet",
        "acdc-dynunet",
        "picai-dynunet",
    ],
)
def test_segmentation_project_builds_and_serves_patches_and_volumes(name, root, tmp_path):
    config = yaml.safe_load((root / f"models/{name}/config.yaml").read_text())
    data = config["data"]["init_args"]
    data["manifest"] = str(_fixture(tmp_path, config))
    data["cache_dir"] = str(tmp_path / "cache")
    data["num_workers"] = 0
    config["trainer"]["accelerator"] = "cpu"
    config["output_dir"] = str(tmp_path / "runs")
    cli = OpenDiagnosticsCLI(args=config, run=False, save_config_callback=None)
    module, datamodule = cli.model, cli.datamodule
    datamodule.setup()
    patch = config["model"]["init_args"]["patch_size"]
    batch = next(iter(datamodule.loader("train", train=True)))
    channels = data["in_channels"]
    assert batch["image"].shape[1:] == (channels, *patch)
    assert batch["target"].shape[1:] == (1, *patch)
    assert int(batch["target"].max()) < len(data["class_names"])
    case = next(iter(datamodule.loader("test")))
    assert case["target"].shape[2:] == (40, 36, 12)
    assert case["spacing"][0] == pytest.approx((0.9, 0.9, 2.0))
    assert module.info.num_classes == len(data["class_names"])
