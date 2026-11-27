"""Write the config of every segmentation model project from one recipe.

    uv run python scripts/segmentation_recipes.py

The recipe follows nnU-Net's 3D full-resolution defaults where they fit a 12 GB GPU:
MONAI's DynUNet (a residual U-Net with instance norm), Dice plus cross-entropy, SGD with
Nesterov momentum 0.99 and a polynomial learning-rate decay, patches cropped half the time
around foreground, flips and intensity jitter, bf16 mixed precision, and Gaussian-weighted
sliding-window inference with half-patch overlap. Datasets differ in voxel spacing, patch,
intensity scaling, flips and batch. Volumes are resampled to the dataset's spacing for the
network, and scored on each label's original grid.

Every project trains for about ``STEPS`` optimizer steps (at most ``MAX_EPOCHS`` epochs),
and validates 20 times. The data scripts in ``scripts/datasets/`` write the manifests.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
STEPS = 50_000
MAX_EPOCHS = 1000
ISOTROPIC = ([[1, 1, 1]] + [[2, 2, 2]] * 5, [[3, 3, 3]] * 6)
ANISOTROPIC = (
    [[1, 1, 1], [2, 2, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 1]],
    [[3, 3, 1], [3, 3, 1]] + [[3, 3, 3]] * 4,
)
SMALL = ([[1, 1, 1]] + [[2, 2, 2]] * 3, [[3, 3, 3]] * 4)
RAS = [["L", "R"], ["P", "A"], ["I", "S"]]


@dataclass(frozen=True)
class Recipe:
    """One dataset's settings. ``window`` (HU) means CT: clip and scale to [0, 1]; without
    it the image is z-scored per channel over nonzero voxels (MRI). ``train_cases`` is the
    approximate size of the train split, used only to size the schedule. Spacing and patch
    are in RAS order (left-right, posterior-anterior, inferior-superior)."""

    manifest: str
    classes: list[str]
    spacing: list[float]
    patch: list[int]
    train_cases: int
    license: str
    citation: str
    notes: str
    window: tuple[int, int] | None = None
    image_columns: list[str] = field(default_factory=lambda: ["image"])
    in_channels: int | None = None
    network: tuple[list[list[int]], list[list[int]]] = ISOTROPIC
    batch_size: int = 1
    crops: int = 2
    flips: tuple[int, ...] = (0, 1, 2)
    regions: dict[str, list[int]] | None = None
    ignore_empty: bool = False
    patient_ids_available: bool = True
    sw_batch_size: int = 2


MSD_CITATION = (
    "Antonelli, M. et al. The Medical Segmentation Decathlon. Nature Communications 13, 4128 "
    "(2022). https://doi.org/10.1038/s41467-022-30695-9"
)
MSD_NOTES = (
    "Medical Segmentation Decathlon {task}. Test labels are hidden, so the labelled cases "
    "are split 70/10/20; MSD has no patient IDs."
)


def _msd(task: str, classes: list[str], train: int, **kwargs: Any) -> Recipe:
    kwargs.setdefault("spacing", [1.0, 1.0, 1.0])
    kwargs.setdefault("patch", [128, 128, 128])
    return Recipe(
        manifest=f"data/msd/{task}/manifest.csv",
        classes=["background", *classes],
        train_cases=train,
        license="CC BY-SA 4.0",
        citation=MSD_CITATION,
        notes=MSD_NOTES.format(task=task),
        patient_ids_available=False,
        **kwargs,
    )


def _totalsegmentator_classes() -> list[str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "totalsegmentator", ROOT / "scripts" / "datasets" / "totalsegmentator.py"
    )
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts" / "datasets"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ROOT / "scripts" / "datasets"))
    return list(module.CLASSES)


RECIPES: dict[str, Recipe] = {
    "kits23-dynunet": Recipe(
        manifest="data/kits23/manifest.csv",
        classes=["background", "kidney", "tumor", "cyst"],
        window=(-150, 350),
        spacing=[1.0, 1.0, 1.0],
        patch=[128, 128, 128],
        train_cases=342,
        regions={"kidney_and_masses": [1, 2, 3], "masses": [2, 3], "tumor": [2]},
        license="CC BY-NC-SA 4.0",
        citation=(
            "Heller, N. et al. The KiTS21 Challenge: Automatic segmentation of kidneys, renal "
            "tumors, and renal cysts in corticomedullary-phase CT. arXiv:2307.01984 (2023)."
        ),
        notes=(
            "KiTS23 training cases. Test labels are hidden, so the 489 cases are split 70/10/20 "
            "by patient; scored on the KiTS23 hierarchical evaluation classes."
        ),
    ),
    "msd-braintumour-dynunet": _msd(
        "Task01_BrainTumour",
        ["edema", "non_enhancing_tumor", "enhancing_tumor"],
        339,
        in_channels=4,
    ),
    "msd-heart-dynunet": _msd(
        "Task02_Heart", ["left_atrium"], 14, spacing=[1.25, 1.25, 1.25], patch=[160, 128, 96]
    ),
    "msd-liver-dynunet": _msd("Task03_Liver", ["liver", "cancer"], 92, window=(-150, 250)),
    "msd-hippocampus-dynunet": _msd(
        "Task04_Hippocampus",
        ["anterior", "posterior"],
        184,
        patch=[48, 64, 48],
        network=SMALL,
        batch_size=8,
        crops=1,
        flips=(0,),
    ),
    "msd-prostate-dynunet": _msd(
        "Task05_Prostate",
        ["peripheral_zone", "transition_zone"],
        22,
        in_channels=2,
        spacing=[0.625, 0.625, 3.6],
        patch=[192, 192, 24],
        network=ANISOTROPIC,
        flips=(0,),
    ),
    "msd-lung-dynunet": _msd("Task06_Lung", ["cancer"], 45, window=(-1000, 400)),
    "msd-pancreas-dynunet": _msd(
        "Task07_Pancreas", ["pancreas", "cancer"], 197, window=(-100, 240)
    ),
    "msd-hepaticvessel-dynunet": _msd(
        "Task08_HepaticVessel", ["vessel", "tumour"], 212, window=(-100, 300)
    ),
    "msd-spleen-dynunet": _msd("Task09_Spleen", ["spleen"], 29, window=(-150, 250)),
    "msd-colon-dynunet": _msd("Task10_Colon", ["colon_cancer"], 88, window=(-150, 250)),
    "totalsegmentator-dynunet": Recipe(
        manifest="data/totalsegmentator/manifest.csv",
        classes=["background", *_totalsegmentator_classes()],
        window=(-1000, 1000),
        spacing=[3.0, 3.0, 3.0],
        patch=[128, 128, 128],
        train_cases=1082,
        flips=(),
        ignore_empty=True,
        patient_ids_available=False,
        sw_batch_size=1,
        license="CC BY 4.0",
        citation=(
            "Wasserthal, J. et al. TotalSegmentator: Robust Segmentation of 104 Anatomic "
            "Structures in CT Images. Radiology: Artificial Intelligence 5, e230024 (2023). "
            "Data v2.0.1: https://doi.org/10.5281/zenodo.10047292"
        ),
        notes=(
            "TotalSegmentator v2.0.1, all 117 structures at 3 mm, the official meta.csv split. "
            "The dataset has no patient IDs."
        ),
    ),
    "acdc-dynunet": Recipe(
        manifest="data/acdc/manifest.csv",
        classes=["background", "right_ventricle", "myocardium", "left_ventricle"],
        spacing=[1.5, 1.5, 5.0],
        patch=[192, 192, 16],
        network=ANISOTROPIC,
        batch_size=2,
        train_cases=160,
        license="CC BY-NC-SA 4.0",
        citation=(
            "Bernard, O. et al. Deep Learning Techniques for Automatic MRI Cardiac "
            "Multi-Structures Segmentation and Diagnosis: Is the Problem Solved? IEEE TMI "
            "37(11), 2514-2525 (2018). https://doi.org/10.1109/TMI.2018.2837502"
        ),
        notes=(
            "ACDC end-diastole and end-systole frames. Official test split (50 patients); "
            "the 100 training patients are split 80/20 into train and validation."
        ),
    ),
    "picai-dynunet": Recipe(
        manifest="data/picai/manifest.csv",
        classes=["background", "cspca"],
        image_columns=["image_t2w", "image_adc", "image_hbv"],
        spacing=[0.5, 0.5, 3.0],
        patch=[256, 256, 20],
        network=ANISOTROPIC,
        flips=(0,),
        train_cases=1050,
        ignore_empty=True,
        license="CC BY-NC 4.0",
        citation=(
            "Saha, A. et al. Artificial intelligence and radiologists in prostate cancer "
            "detection on MRI (PI-CAI). Lancet Oncology 25, 879-887 (2024). "
            "https://doi.org/10.1016/S1470-2045(24)00220-1"
        ),
        notes=(
            "PI-CAI public studies with expert csPCa masks. Hidden test labels, so studies "
            "are split 70/10/20 by patient. Dice is over studies with a lesion."
        ),
    ),
}


def _t(name: str, **init_args: Any) -> dict[str, Any]:
    return {"class_path": f"monai.transforms.{name}", "init_args": init_args}


def _preprocess(r: Recipe, train: bool) -> list[dict[str, Any]]:
    keys = ["image", "label"]
    steps = [
        _t("Orientationd", keys=keys, axcodes="RAS", labels=RAS),
        _t("Spacingd", keys=keys, pixdim=r.spacing, mode=["bilinear", "nearest"])
        if train
        else _t("Spacingd", keys=["image"], pixdim=r.spacing, mode="bilinear"),
    ]
    if r.window:
        low, high = r.window
        steps.append(
            _t(
                "ScaleIntensityRanged",
                keys=["image"],
                a_min=low,
                a_max=high,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )
    else:
        steps.append(_t("NormalizeIntensityd", keys=["image"], nonzero=True, channel_wise=True))
    steps.append(_t("CastToTyped", keys=["image"], dtype="float16"))
    if train:
        steps.append(_t("SpatialPadd", keys=keys, spatial_size=r.patch))
    return steps


def _augment(r: Recipe) -> list[dict[str, Any]]:
    keys = ["image", "label"]
    steps = [
        _t(
            "RandCropByPosNegLabeld",
            keys=keys,
            label_key="label",
            spatial_size=r.patch,
            pos=1.0,
            neg=1.0,
            num_samples=r.crops,
        )
    ]
    steps += [_t("RandFlipd", keys=keys, prob=0.5, spatial_axis=axis) for axis in r.flips]
    steps += [
        _t("RandScaleIntensityd", keys=["image"], factors=0.1, prob=0.5),
        _t("RandShiftIntensityd", keys=["image"], offsets=0.1, prob=0.5),
    ]
    return steps


def recipe(name: str) -> dict[str, Any]:
    r = RECIPES[name]
    steps_per_epoch = math.ceil(r.train_cases / r.batch_size)
    epochs = min(MAX_EPOCHS, math.ceil(STEPS / steps_per_epoch))
    strides, kernels = r.network
    channels = r.in_channels or len(r.image_columns)
    model_args: dict[str, Any] = {
        "patch_size": r.patch,
        "network": {
            "class_path": "monai.networks.nets.DynUNet",
            "init_args": {
                "spatial_dims": 3,
                "in_channels": channels,
                "out_channels": len(r.classes),
                "kernel_size": kernels,
                "strides": strides,
                "upsample_kernel_size": strides[1:],
                "norm_name": "instance",
                "res_block": True,
            },
        },
        "sw_batch_size": r.sw_batch_size,
        "overlap": 0.5,
    }
    if r.regions:
        model_args["regions"] = r.regions
    if r.ignore_empty:
        model_args["ignore_empty"] = True
    project = name.removesuffix("-dynunet")
    return {
        "name": name,
        "notes": r.notes,
        "seed_everything": 0,
        "trainer": {
            "accelerator": "auto",
            "precision": "bf16-mixed",
            "deterministic": "warn",
            "max_epochs": epochs,
            "check_val_every_n_epoch": max(1, epochs // 20),
            "gradient_clip_val": 12.0,
            "log_every_n_steps": 50,
            "callbacks": [
                {
                    "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                    "init_args": {"logging_interval": "epoch"},
                }
            ],
        },
        "model": {
            "class_path": "opendiagnostics.modules.SegmentationModule",
            "init_args": model_args,
        },
        "data": {
            "class_path": "opendiagnostics.data.NiftiSegmentationDataModule",
            "init_args": {
                "manifest": r.manifest,
                "class_names": r.classes,
                "image_columns": r.image_columns,
                "in_channels": channels,
                "patient_ids_available": r.patient_ids_available,
                "license": r.license,
                "citation": r.citation,
                "split": "predefined",
                "batch_size": r.batch_size,
                "num_workers": 6,
                "cache_dir": f"data/cache/{project}",
                "train_transforms": _t(
                    "Compose", transforms=[*_preprocess(r, train=True), *_augment(r)]
                ),
                "eval_transforms": _t("Compose", transforms=_preprocess(r, train=False)),
            },
        },
        "optimizer": {
            "class_path": "torch.optim.SGD",
            "init_args": {"lr": 0.01, "momentum": 0.99, "nesterov": True, "weight_decay": 3.0e-5},
        },
        "lr_scheduler": {
            "class_path": "torch.optim.lr_scheduler.PolynomialLR",
            "init_args": {"total_iters": epochs, "power": 0.9},
        },
        "checkpoint": {"monitor": "val/dice", "mode": "max"},
        "eval": {"bootstrap_samples": 1000},
    }


class _NoAliases(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def render(name: str) -> str:
    header = (
        "# Generated by scripts/segmentation_recipes.py. Edit the script, then rerun it.\n"
        f"#   uv run opendiagnostics fit -c models/{name}/config.yaml\n"
    )
    return header + yaml.dump(recipe(name), Dumper=_NoAliases, sort_keys=False, width=100)


def main(out: Path = ROOT / "models") -> None:
    for name in RECIPES:
        project = out / name
        project.mkdir(parents=True, exist_ok=True)
        (project / "config.yaml").write_text(render(name))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models")
