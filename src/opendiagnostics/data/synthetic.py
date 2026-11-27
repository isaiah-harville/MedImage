"""Generated images from MONAI's test-image generators, for seconds-long smoke runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import tifffile
import torch
from monai.data.synthetic import create_test_image_2d, create_test_image_3d
from torchvision.ops import masks_to_boxes

from opendiagnostics.data.base import DatasetInfo, MedicalDataModule, Source
from opendiagnostics.data.slides import Slide, SlideFeatureDataModule


class SyntheticClassificationDataModule(MedicalDataModule):
    """Class ``k`` draws ``k + 1`` blobs, so a small network can learn it in a few epochs."""

    def __init__(
        self,
        num_samples: int = 160,
        image_size: int = 32,
        spatial_dims: Literal[2, 3] = 2,
        num_classes: int = 2,
        images_per_patient: int = 1,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("split", "patient")
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.image_size = image_size
        self.spatial_dims = spatial_dims
        self.images_per_patient = images_per_patient
        self.seed = seed
        self.info = DatasetInfo(
            label_mode="binary" if num_classes == 2 else "multiclass",
            class_names=tuple(f"class_{k}" for k in range(num_classes)),
            in_channels=1,
            spatial_dims=spatial_dims,
            license="generated",
        )

    def load(self) -> Source:
        rng = np.random.RandomState(self.seed)
        size = self.image_size
        labels = rng.randint(0, self.info.num_classes, self.num_samples)
        images = []
        radius = max(size // 6, 3)
        for label in labels:
            objects = int(label) + 1
            if self.spatial_dims == 2:
                generated = create_test_image_2d(
                    size, size, objects, radius, 1, 0.2, 1, random_state=rng
                )
            else:
                generated = create_test_image_3d(
                    size, size, size, objects, radius, 1, 0.2, 1, random_state=rng
                )
            images.append(torch.from_numpy(generated[0][None].astype(np.float32)))
        ids = [f"s{i}" for i in range(self.num_samples)]
        return Source(
            dataset=list(zip(images, labels.tolist(), strict=True)),
            sample_ids=ids,
            patient_ids=[f"p{i // self.images_per_patient}" for i in range(self.num_samples)],
            strata=labels.tolist(),
            fingerprint={"seed": self.seed, "num_samples": self.num_samples},
        )


class SyntheticDetectionDataModule(MedicalDataModule):
    """Blobs on noise, one ``blob`` box per blob."""

    def __init__(
        self, num_samples: int = 48, image_size: int = 64, seed: int = 0, **kwargs: Any
    ) -> None:
        kwargs.setdefault("split", "patient")
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed
        self.info = DatasetInfo(
            label_mode="detection", class_names=("blob",), in_channels=1, license="generated"
        )

    def load(self) -> Source:
        rng = np.random.RandomState(self.seed)
        items = []
        for _ in range(self.num_samples):
            generated = create_test_image_2d(
                self.image_size,
                self.image_size,
                num_objs=int(rng.randint(1, 4)),
                rad_max=self.image_size // 6,
                rad_min=3,
                noise_max=0.2,
                num_seg_classes=1,
                random_state=rng,
                return_instance_id=True,
            )
            image, instances = generated[0], generated[-1]
            ids = [i for i in np.unique(instances) if i]
            masks = torch.from_numpy(np.stack([instances == i for i in ids]))
            boxes = masks_to_boxes(masks) if ids else torch.zeros(0, 4)
            image = torch.from_numpy(image[None].astype(np.float32))
            items.append((image, {"boxes": boxes, "labels": np.zeros(len(ids), np.int64)}))
        sample_ids = [f"s{i}" for i in range(self.num_samples)]
        return Source(
            dataset=items,
            sample_ids=sample_ids,
            patient_ids=sample_ids,
            fingerprint={"seed": self.seed, "num_samples": self.num_samples},
        )


def write_pyramid(path: Path, image: np.ndarray, levels: int, tile: int = 64) -> None:
    """Write ``uint8 (H, W, 3)`` as a tiled pyramidal TIFF, halving per level, that
    OpenSlide reads as a generic tiled TIFF."""
    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        for level in range(levels):
            tif.write(
                image[:: 2**level, :: 2**level],
                tile=(tile, tile),
                photometric="rgb",
                compression="zlib",
                subfiletype=1 if level else 0,
            )


class SyntheticSlideDataModule(SlideFeatureDataModule):
    """Tiny generated slides: a pink tissue rectangle on white, with dark purple nests in
    ``tumor`` slides. ``prepare_data()`` writes them to ``root/images`` before extracting."""

    def __init__(
        self,
        num_slides: int = 24,
        slide_size: int = 512,
        slides_per_patient: int = 2,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("split", "patient")
        super().__init__(**kwargs)
        self.num_slides = num_slides
        self.slide_size = slide_size
        self.slides_per_patient = slides_per_patient
        self.seed = seed
        self.info = DatasetInfo(
            label_mode="binary", class_names=("normal", "tumor"), in_channels=3, license="generated"
        )

    def slides(self) -> list[Slide]:
        labels = np.random.RandomState(self.seed).randint(0, 2, self.num_slides)
        return [
            Slide(
                slide_id=f"slide_{n:03d}",
                path=self.root / "images" / f"slide_{n:03d}.tif",
                patient_id=f"p{n // self.slides_per_patient}",
                label=int(label),
            )
            for n, label in enumerate(labels)
        ]

    def prepare_data(self) -> None:
        rng = np.random.RandomState(self.seed + 1)
        size = self.slide_size
        for slide in self.slides():
            if slide.path.exists():
                continue
            image = np.full((size, size, 3), 240, np.uint8)
            top, left = rng.randint(size // 8, size // 4, 2)
            image[top : size - top, left : size - left] = (220, 150, 190)
            if slide.label:
                for _ in range(4):
                    y, x = rng.randint(size // 4, 3 * size // 4 - size // 8, 2)
                    image[y : y + size // 8, x : x + size // 8] = (90, 40, 120)
            slide.path.parent.mkdir(parents=True, exist_ok=True)
            partial = slide.path.with_suffix(".partial")
            write_pyramid(partial, image, levels=4)
            partial.replace(slide.path)
        super().prepare_data()
