"""Write one model project config per MedMNIST subset from two recipes, one 2D and one 3D.

    uv run python scripts/medmnist_recipes.py

The 2D recipe: ConvNeXt-S pretrained on ImageNet-22k (timm), fine-tuned end to end at 224 px
with AdamW, a cosine schedule, bf16 mixed precision, and EMA weights. It fits a 12 GB GPU.
Subsets differ only in epochs and in whether flips are safe: chest X-rays and abdominal CT
have a left and a right, so those subsets get small affine jitter instead.

The 3D recipe: a ResNet-18 with 3D convolutions (MONAI), trained from scratch on the 64-voxel
cubes with AdamW, a cosine schedule, bf16 mixed precision, and EMA weights. There is no 3D
ImageNet, and MONAI's MedicalNet weights were pretrained on LiTS among others, the source of
OrganMNIST3D's test volumes. The first convolution keeps stride 1, so a 64-voxel nodule crop
is not halved before the residual stages; that costs about 260 MB of activations per volume
in bf16, so a batch of 16 fits in 12 GB with room to spare. Every subset gets a small random
affine (10 degrees, 10% scale, 4 voxels shift). The crops and shapes have no fixed side, so
flips on all three axes are label-preserving, and test-time augmentation adds the flipped
view, except OrganMNIST3D: its volumes keep the body's left and right. Shapes (Adrenal,
Vessel) are binary masks, so the affine pads with zeros; CT and microscopy crops pad by
repeating their border.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from medmnist import INFO

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = "convnext_small.fb_in22k_ft_in1k"
BATCH = 64
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
GRAY = ([0.449], [0.226])

# subset: (epochs, flips are label-preserving)
RECIPES: dict[str, tuple[int, bool]] = {
    "pathmnist": (10, True),
    "chestmnist": (10, False),
    "dermamnist": (40, True),
    "octmnist": (10, True),
    "pneumoniamnist": (40, False),
    "retinamnist": (60, True),
    "breastmnist": (80, True),
    "bloodmnist": (30, True),
    "tissuemnist": (8, True),
    "organamnist": (20, False),
    "organcmnist": (30, False),
    "organsmnist": (30, False),
}

NETWORK_3D = "resnet18"
BATCH_3D = 16
# subset: (epochs, flips are label-preserving, affine padding)
RECIPES_3D: dict[str, tuple[int, bool, str]] = {
    "organmnist3d": (100, False, "border"),
    "nodulemnist3d": (100, True, "border"),
    "fracturemnist3d": (100, True, "border"),
    "adrenalmnist3d": (100, True, "zeros"),
    "vesselmnist3d": (100, True, "zeros"),
    "synapsemnist3d": (100, True, "border"),
}


def _transform(name: str, **init_args: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {"class_path": f"torchvision.transforms.v2.{name}"}
    if init_args:
        spec["init_args"] = init_args
    return spec


def _monai(name: str, **init_args: Any) -> dict[str, Any]:
    return {"class_path": f"monai.transforms.{name}", "init_args": init_args}


def project(flag: str) -> str:
    return f"{flag}-{NETWORK_3D}" if flag in RECIPES_3D else f"{flag}-convnext"


def _task(flag: str, batch: int, epochs: int) -> tuple[int, bool, float]:
    """Network outputs, whether the task is multiclass, and the EMA decay."""
    meta: dict[str, Any] = INFO[flag]
    classes = len(meta["label"])
    multilabel = meta["task"].startswith("multi-label")
    outputs = 1 if classes == 2 and not multilabel else classes
    steps = epochs * -(-meta["n_samples"]["train"] // batch)
    # An EMA horizon of about a tenth of training, capped at the usual 0.9999.
    decay = round(min(0.9999, 1 - 10 / steps), 5)
    return outputs, not multilabel and classes > 2, decay


def _experiment(
    flag: str,
    notes: str,
    epochs: int,
    decay: float,
    model_args: dict[str, Any],
    data_args: dict[str, Any],
    lr: float,
) -> dict[str, Any]:
    return {
        "name": project(flag),
        "notes": notes,
        "seed_everything": 0,
        "trainer": {
            "accelerator": "auto",
            "precision": "bf16-mixed",
            "deterministic": "warn",
            "max_epochs": epochs,
            "log_every_n_steps": 50,
            "callbacks": [
                {
                    "class_path": "lightning.pytorch.callbacks.EMAWeightAveraging",
                    "init_args": {"decay": decay},
                },
                {
                    "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                    "init_args": {"logging_interval": "epoch"},
                },
            ],
        },
        "model": {"class_path": "opendiagnostics.modules.ImageClassifier", "init_args": model_args},
        "data": {"class_path": "opendiagnostics.data.MedMNISTDataModule", "init_args": data_args},
        "optimizer": {
            "class_path": "torch.optim.AdamW",
            "init_args": {"lr": lr, "weight_decay": 0.05},
        },
        "lr_scheduler": {
            "class_path": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "init_args": {"T_max": epochs, "eta_min": 1e-6},
        },
        "checkpoint": {"monitor": "val/auroc", "mode": "max"},
        "eval": {"bootstrap_samples": 1000},
    }


def recipe_2d(flag: str) -> dict[str, Any]:
    epochs, flips = RECIPES[flag]
    meta: dict[str, Any] = INFO[flag]
    channels = meta["n_channels"]
    outputs, multiclass, decay = _task(flag, BATCH, epochs)
    mean, std = IMAGENET if channels == 3 else GRAY
    augment = [_transform("RandomResizedCrop", size=224, scale=[0.7, 1.0], antialias=True)]
    if flips:
        augment += [_transform("RandomHorizontalFlip"), _transform("RandomVerticalFlip")]
    else:
        augment.append(_transform("RandomAffine", degrees=10, translate=[0.05, 0.05]))
    model_args: dict[str, Any] = {
        "network": {
            "class_path": "timm.create_model",
            "init_args": {"model_name": BACKBONE, "pretrained": True},
            "dict_kwargs": {"num_classes": outputs, "in_chans": channels, "drop_path_rate": 0.2},
        },
        "tta": flips,
    }
    if multiclass:
        model_args["label_smoothing"] = 0.1
    data_args = {
        "flag": flag,
        "size": 224,
        "split": "predefined",
        "batch_size": BATCH,
        "eval_batch_size": 256,
        "num_workers": 8,
        "train_transforms": _transform(
            "Compose", transforms=[*augment, _transform("Normalize", mean=mean, std=std)]
        ),
        "eval_transforms": _transform(
            "Compose",
            transforms=[
                _transform("Resize", size=224, antialias=True),
                _transform("Normalize", mean=mean, std=std),
            ],
        ),
    }
    notes = f"{meta['python_class']} at 224 px. Official split; MedMNIST has no patient IDs."
    return _experiment(flag, notes, epochs, decay, model_args, data_args, lr=1e-4)


def recipe_3d(flag: str) -> dict[str, Any]:
    epochs, flips, padding = RECIPES_3D[flag]
    meta: dict[str, Any] = INFO[flag]
    outputs, multiclass, decay = _task(flag, BATCH_3D, epochs)
    augment = [
        _monai(
            "RandAffine",
            prob=0.8,
            rotate_range=[0.17, 0.17, 0.17],
            scale_range=[0.1, 0.1, 0.1],
            translate_range=[4, 4, 4],
            padding_mode=padding,
        )
    ]
    if flips:
        augment += [_monai("RandFlip", prob=0.5, spatial_axis=axis) for axis in range(3)]
    model_args: dict[str, Any] = {
        "network": {
            "class_path": "monai.networks.nets.ResNet",
            "init_args": {
                "block": "basic",
                "layers": [2, 2, 2, 2],
                "block_inplanes": [64, 128, 256, 512],
                "spatial_dims": 3,
                "n_input_channels": meta["n_channels"],
                "num_classes": outputs,
            },
        },
        "tta": flips,
    }
    if multiclass:
        model_args["label_smoothing"] = 0.1
    data_args = {
        "flag": flag,
        "size": 64,
        "root": "data/medmnist",
        "split": "predefined",
        "batch_size": BATCH_3D,
        "eval_batch_size": 64,
        "num_workers": 8,
        "train_transforms": _monai("Compose", transforms=augment),
    }
    notes = f"{meta['python_class']} at 64 voxels. Official split; MedMNIST has no patient IDs."
    return _experiment(flag, notes, epochs, decay, model_args, data_args, lr=1e-3)


def recipe(flag: str) -> dict[str, Any]:
    return recipe_3d(flag) if flag in RECIPES_3D else recipe_2d(flag)


class _NoAliases(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def render(flag: str) -> str:
    header = (
        "# Generated by scripts/medmnist_recipes.py. Edit the script, then rerun it.\n"
        f"#   uv run opendiagnostics fit -c models/{project(flag)}/config.yaml\n"
    )
    return header + yaml.dump(recipe(flag), Dumper=_NoAliases, sort_keys=False, width=100)


def main(out: Path = ROOT / "models") -> None:
    for flag in [*RECIPES, *RECIPES_3D]:
        path = out / project(flag)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.yaml").write_text(render(flag))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models")
