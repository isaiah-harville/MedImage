"""Write one model project config per MedMNIST 2D subset from a single recipe.

    uv run python scripts/medmnist_recipes.py

The recipe: ConvNeXt-S pretrained on ImageNet-22k (timm), fine-tuned end to end at 224 px
with AdamW, a cosine schedule, bf16 mixed precision, and EMA weights. It fits a 12 GB GPU.
Subsets differ only in epochs and in whether flips are safe: chest X-rays and abdominal CT
have a left and a right, so those subsets get small affine jitter instead.
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


def _transform(name: str, **init_args: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {"class_path": f"torchvision.transforms.v2.{name}"}
    if init_args:
        spec["init_args"] = init_args
    return spec


def recipe(flag: str) -> dict[str, Any]:
    epochs, flips = RECIPES[flag]
    meta: dict[str, Any] = INFO[flag]
    channels = meta["n_channels"]
    classes = len(meta["label"])
    multilabel = meta["task"].startswith("multi-label")
    outputs = 1 if classes == 2 and not multilabel else classes
    mean, std = IMAGENET if channels == 3 else GRAY
    steps = epochs * -(-meta["n_samples"]["train"] // BATCH)
    # An EMA horizon of about a tenth of training, capped at the usual 0.9999.
    decay = round(min(0.9999, 1 - 10 / steps), 5)
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
    if not multilabel and classes > 2:
        model_args["label_smoothing"] = 0.1
    return {
        "name": f"{flag}-convnext",
        "notes": f"{meta['python_class']} at 224 px. Official split; MedMNIST has no patient IDs.",
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
        "data": {
            "class_path": "opendiagnostics.data.MedMNISTDataModule",
            "init_args": {
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
            },
        },
        "optimizer": {
            "class_path": "torch.optim.AdamW",
            "init_args": {"lr": 1e-4, "weight_decay": 0.05},
        },
        "lr_scheduler": {
            "class_path": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "init_args": {"T_max": epochs, "eta_min": 1e-6},
        },
        "checkpoint": {"monitor": "val/auroc", "mode": "max"},
        "eval": {"bootstrap_samples": 1000},
    }


class _NoAliases(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def render(flag: str) -> str:
    name = f"{flag}-convnext"
    header = (
        "# Generated by scripts/medmnist_recipes.py. Edit the script, then rerun it.\n"
        f"#   uv run opendiagnostics fit -c models/{name}/config.yaml\n"
    )
    return header + yaml.dump(recipe(flag), Dumper=_NoAliases, sort_keys=False, width=100)


def main(out: Path = ROOT / "models") -> None:
    for flag in RECIPES:
        project = out / f"{flag}-convnext"
        project.mkdir(parents=True, exist_ok=True)
        (project / "config.yaml").write_text(render(flag))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models")
