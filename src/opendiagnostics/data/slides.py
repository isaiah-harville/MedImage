"""Whole-slide images as bags of patch features, for multiple-instance learning.

Training on gigapixel slides is done in two stages. ``prepare_data()`` finds tissue on a
low-resolution level, tiles it, runs a frozen pretrained encoder over every tile, and writes
one ``<slide_id>.pt`` file of features per slide to ``features_dir``. It skips slides whose
file exists, so an interrupted extraction resumes where it stopped. Training then reads only
the cached features: each sample is a ``float32 (N, D)`` bag of ``N`` tile features, served
one slide per batch because bags differ in size.

Slides are read with MONAI's ``WSIReader``. The default OpenSlide backend reads one tile at
a time; MONAI's tifffile backend decodes a whole level per read, so it suits only tiny
slides.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from monai.data import WSIReader
from monai.transforms import ForegroundMask
from torch import nn
from torch.utils.data import DataLoader, Dataset

from opendiagnostics.data.base import MedicalDataModule, Source, Transform, patient_split

log = logging.getLogger("opendiagnostics")

EXTRACTION_RECORD = "extraction.json"


@dataclass(frozen=True)
class Slide:
    """One slide. ``split`` is ``train`` or ``test`` for an official split, else ``None``."""

    slide_id: str
    path: Path
    patient_id: str
    label: int
    split: str | None = None
    stratum: Hashable | None = None

    @property
    def stratify_on(self) -> Hashable:
        return self.label if self.stratum is None else self.stratum


def level_for_downsample(ratios: Sequence[float], target: float) -> int:
    """The pyramid level whose downsample ratio is closest to ``target`` on a log scale."""
    return int(np.argmin([abs(np.log(r / target)) for r in ratios]))


def tile_grid(mask: np.ndarray, size: tuple[int, int], tile: int, min_tissue: float) -> np.ndarray:
    """Top-left level-0 ``(y, x)`` of every whole ``tile``-pixel tile with enough tissue.

    ``mask`` is a boolean tissue mask covering the level-0 image of ``size`` ``(H, W)``.
    A tile is kept when the fraction of its mask pixels that are tissue is at least
    ``min_tissue``. Tiles smaller than a mask pixel use the pixel they fall in.
    """
    rows, cols = size[0] // tile, size[1] // tile
    if rows == 0 or cols == 0:
        return np.zeros((0, 2), np.int64)
    integral = np.pad(np.cumsum(np.cumsum(mask.astype(np.int64), 0), 1), ((1, 0), (1, 0)))

    def edges(count: int, extent: int, pixels: int) -> tuple[np.ndarray, np.ndarray]:
        start = np.floor(np.arange(count) * tile * pixels / extent).astype(np.int64)
        stop = np.floor(np.arange(1, count + 1) * tile * pixels / extent).astype(np.int64)
        start = np.minimum(start, pixels - 1)
        return start, np.clip(stop, start + 1, pixels)

    y0, y1 = edges(rows, size[0], mask.shape[0])
    x0, x1 = edges(cols, size[1], mask.shape[1])
    tissue = integral[y1][:, x1] - integral[y0][:, x1] - integral[y1][:, x0] + integral[y0][:, x0]
    fraction = tissue / ((y1 - y0)[:, None] * (x1 - x0)[None, :])
    keep_rows, keep_cols = np.nonzero(fraction >= min_tissue)
    return np.stack([keep_rows * tile, keep_cols * tile], axis=1).astype(np.int64)


def pool(output: Any) -> torch.Tensor:
    """One vector per tile from an encoder's output.

    A list (timm ``features_only``) contributes its last feature map; a feature map
    ``(B, D, h, w)`` is averaged over space; a ``(B, D)`` output is used as is.
    """
    if isinstance(output, list | tuple):
        output = output[-1]
    if output.ndim == 4:
        output = output.mean((2, 3))
    if output.ndim != 2:
        raise ValueError(
            f"the encoder returned shape {tuple(output.shape)}; it needs (batch, features) "
            "or a (batch, features, h, w) map. For timm, set num_classes: 0 or features_only."
        )
    return output


class SlideTiles(Dataset):
    """The tiles of one slide as ``uint8 (3, size, size)``. Each worker opens the slide once."""

    def __init__(self, path: Path, backend: str, coords: np.ndarray, size: int, level: int) -> None:
        self.path = path
        self.backend = backend
        self.coords = coords
        self.size = size
        self.level = level
        self._open: tuple[WSIReader, Any] | None = None

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: int) -> torch.Tensor:
        if self._open is None:
            reader = WSIReader(backend=self.backend)
            self._open = reader, reader.read(str(self.path))
        reader, wsi = self._open
        y, x = (int(v) for v in self.coords[index])
        tile, _ = reader.get_data(
            wsi, location=(y, x), size=(self.size, self.size), level=self.level
        )
        return torch.tensor(np.asarray(tile), dtype=torch.uint8)

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__ | {"_open": None}


@dataclass(frozen=True)
class Extraction:
    """How tiles are cut and encoded. Features from different settings are never mixed."""

    patch_size: int
    patch_level: int
    mask_downsample: float
    min_tissue: float
    backend: str
    batch_size: int
    workers: int
    device: str

    def record(self, encoder: nn.Module, transform: Transform | None) -> dict[str, Any]:
        digest = hashlib.sha256()
        for name, tensor in sorted(encoder.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return {
            "patch_size": self.patch_size,
            "patch_level": self.patch_level,
            "mask_downsample": self.mask_downsample,
            "min_tissue": self.min_tissue,
            "encoder": type(encoder).__name__,
            "encoder_sha256": digest.hexdigest(),
            "transform": repr(transform),
        }

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@torch.inference_mode()
def extract_slide(
    path: Path, settings: Extraction, encoder: nn.Module, transform: Transform | None
) -> dict[str, torch.Tensor]:
    """Features ``float16 (N, D)`` and level-0 tile corners ``int64 (N, 2)`` of one slide."""
    reader = WSIReader(backend=settings.backend)
    wsi = reader.read(str(path))
    ratios = [reader.get_downsample_ratio(wsi, n) for n in range(reader.get_level_count(wsi))]
    if settings.patch_level >= len(ratios):
        raise ValueError(
            f"{path} has {len(ratios)} levels; patch_level {settings.patch_level} does not exist"
        )
    thumbnail, _ = reader.get_data(
        wsi, level=level_for_downsample(ratios, settings.mask_downsample)
    )
    mask = np.squeeze(np.asarray(ForegroundMask(hsv_threshold={"S": "otsu"})(thumbnail))) > 0
    tile = round(settings.patch_size * ratios[settings.patch_level])
    size = reader.get_size(wsi, 0)
    coords = tile_grid(mask, (int(size[0]), int(size[1])), tile, settings.min_tissue)
    if not len(coords):
        raise ValueError(
            f"found no tissue tiles in {path}; lower min_tissue ({settings.min_tissue}) or "
            "check that the slide is not blank"
        )
    device = settings.resolved_device()
    tiles = DataLoader(
        SlideTiles(path, settings.backend, coords, settings.patch_size, settings.patch_level),
        batch_size=settings.batch_size,
        num_workers=settings.workers,
        pin_memory=device.type == "cuda",
    )
    features = []
    for batch in tiles:
        images = batch.to(device, non_blocking=True).float() / 255
        if transform is not None:
            images = transform(images)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            features.append(pool(encoder(images)).half().cpu())
    return {"features": torch.cat(features), "coords": torch.from_numpy(coords)}


class FeatureBags:
    """Cached feature files as ``(float32 (N, D), label)`` pairs, read when indexed."""

    def __init__(self, paths: list[Path], labels: list[int]) -> None:
        self.paths = paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        saved = torch.load(self.paths[index], map_location="cpu", weights_only=True)
        return saved["features"].float(), self.labels[index]


class SlideFeatureDataModule(MedicalDataModule):
    """Base for whole-slide datasets. Subclasses set ``info`` and implement :meth:`slides`.

    ``encoder`` is any network that maps ``float (B, 3, H, W)`` tiles in ``[0, 1]`` (after
    ``encoder_transform``) to features, e.g. ``timm.create_model``; it is frozen and used
    only by ``prepare_data()``. Tiles are ``patch_size`` pixels at pyramid ``patch_level``
    and need ``min_tissue`` of their area to be tissue in an Otsu mask made at the level
    closest to ``mask_downsample``. When the dataset's official split has a train part but
    no validation part, ``val_fraction`` of the training patients become validation.
    """

    def __init__(
        self,
        root: str,
        features_dir: str,
        encoder: nn.Module,
        encoder_transform: Transform | None = None,
        patch_size: int = 256,
        patch_level: int = 1,
        mask_downsample: float = 32.0,
        min_tissue: float = 0.25,
        reader: Literal["openslide", "cucim", "tifffile"] = "openslide",
        extract_batch_size: int = 128,
        extract_workers: int = 4,
        extract_device: str = "auto",
        val_fraction: float = 0.15,
        batch_size: int = 1,
        eval_batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        if batch_size != 1 or eval_batch_size not in (None, 1):
            raise ValueError("slide bags differ in size, so batch_size must be 1")
        super().__init__(batch_size=1, **kwargs)
        self.root = Path(root)
        self.features_dir = Path(features_dir)
        self.encoder = encoder
        self.encoder_transform = encoder_transform
        self.val_fraction = val_fraction
        self.extraction = Extraction(
            patch_size,
            patch_level,
            mask_downsample,
            min_tissue,
            reader,
            extract_batch_size,
            extract_workers,
            extract_device,
        )

    def slides(self) -> list[Slide]:
        raise NotImplementedError

    def feature_path(self, slide: Slide) -> Path:
        return self.features_dir / f"{slide.slide_id}.pt"

    def prepare_data(self) -> None:
        """Extract features for every slide that has no feature file yet."""
        self._check_record(self.extraction.record(self.encoder, self.encoder_transform))
        todo = [s for s in self.slides() if not self.feature_path(s).exists()]
        if not todo:
            return
        missing = [s.path for s in todo if not s.path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} slides are missing, the first {missing[0]}. Download them "
                f"into {self.root} first; see the scripts in scripts/datasets/."
            )
        device = self.extraction.resolved_device()
        encoder = self.encoder.to(device).eval()
        for n, slide in enumerate(todo, start=1):
            saved = extract_slide(slide.path, self.extraction, encoder, self.encoder_transform)
            target = self.feature_path(slide)
            partial = target.with_suffix(".partial")
            torch.save(saved, partial)
            partial.replace(target)
            log.info(
                "features %d/%d: %s, %d tiles", n, len(todo), slide.slide_id, len(saved["coords"])
            )
        encoder.cpu()

    def _check_record(self, record: dict[str, Any]) -> None:
        self.features_dir.mkdir(parents=True, exist_ok=True)
        path = self.features_dir / EXTRACTION_RECORD
        if path.exists():
            existing = json.loads(path.read_text())
            if existing != record:
                changed = sorted(k for k in record if existing.get(k) != record[k])
                raise ValueError(
                    f"{self.features_dir} holds features made with other settings "
                    f"({', '.join(changed)}). Point features_dir at a new directory."
                )
            return
        path.write_text(json.dumps(record, indent=2) + "\n")

    def load(self) -> Source:
        self.prepare_data()
        slides = self.slides()
        splits = None
        if any(s.split for s in slides):
            train = [i for i, s in enumerate(slides) if s.split == "train"]
            carved = patient_split(
                [slides[i].patient_id for i in train],
                (1 - self.val_fraction, self.val_fraction, 0.0),
                [slides[i].stratify_on for i in train],
                self.split_seed,
            )
            splits = {
                "train": [train[i] for i in carved["train"]],
                "val": [train[i] for i in carved["val"]],
                "test": [i for i, s in enumerate(slides) if s.split == "test"],
            }
        record = json.loads((self.features_dir / EXTRACTION_RECORD).read_text())
        return Source(
            dataset=FeatureBags([self.feature_path(s) for s in slides], [s.label for s in slides]),
            sample_ids=[s.slide_id for s in slides],
            patient_ids=[s.patient_id for s in slides],
            strata=[s.stratify_on for s in slides],
            splits=splits,
            fingerprint={"slides": len(slides), "features": record},
        )
