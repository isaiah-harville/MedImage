"""Reading medical images into channel-first float arrays.

Every loader returns an :class:`ImageData` whose ``array`` is ``float32`` shaped
``(C, *spatial)`` and whose ``spacing`` gives the physical size of a voxel along each spatial
axis, in millimetres where the format records it.

Axis conventions
    DICOM: ``(rows, columns)`` for a single image and ``(slices, rows, columns)`` for a
    series or multi-frame file, slices ordered along the slice normal.
    NIfTI: reoriented to the closest canonical RAS+ orientation, then transposed to
    ``(z, y, x)`` so the slice axis comes first, as with DICOM. The two formats still use
    different patient coordinate systems (DICOM LPS, NIfTI RAS), so do not mix them in one
    dataset without resampling to a common orientation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DICOM_SUFFIXES = {".dcm", ".dicom"}
NIFTI_SUFFIXES = {".nii", ".nii.gz"}
RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class ImageData:
    array: np.ndarray
    spacing: tuple[float, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def spatial_dims(self) -> int:
        return self.array.ndim - 1


def _suffix(path: Path) -> str:
    name = path.name.lower()
    return ".nii.gz" if name.endswith(".nii.gz") else path.suffix.lower()


def load_image(path: str | Path, **kwargs: Any) -> ImageData:
    """Load any supported file, or a directory holding one DICOM series."""
    path = Path(path)
    if path.is_dir():
        return load_dicom_series(path, **kwargs)
    suffix = _suffix(path)
    if suffix in DICOM_SUFFIXES:
        return load_dicom(path, **kwargs)
    if suffix in NIFTI_SUFFIXES:
        return load_nifti(path, **kwargs)
    if suffix in RASTER_SUFFIXES:
        return load_raster(path, **kwargs)
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False).astype(np.float32)
        return ImageData(array[None] if kwargs.get("add_channel", True) else array, ())
    raise ValueError(f"unsupported image format: {path}")


def _dicom_pixels(dataset: Any, apply_voi: bool) -> np.ndarray:
    from pydicom.pixels import apply_modality_lut, apply_voi_lut

    pixels = dataset.pixel_array
    pixels = apply_modality_lut(pixels, dataset)
    if apply_voi:
        pixels = apply_voi_lut(pixels, dataset)
    pixels = pixels.astype(np.float32)
    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        # MONOCHROME1 shows low values as white; invert so higher means brighter everywhere.
        pixels = np.max(pixels) + np.min(pixels) - pixels
    return pixels


def load_dicom(path: str | Path, apply_voi: bool = False) -> ImageData:
    """Load one DICOM file, single- or multi-frame.

    The modality LUT (rescale slope and intercept) is always applied, so CT values are in
    Hounsfield units. The VOI LUT (display windowing) is applied only if ``apply_voi`` is
    true, because windowing discards information a model could use.
    """
    import pydicom

    dataset = pydicom.dcmread(path)
    pixels = _dicom_pixels(dataset, apply_voi)
    samples_per_pixel = int(getattr(dataset, "SamplesPerPixel", 1))
    if samples_per_pixel > 1:
        pixels = np.moveaxis(pixels, -1, 0)
    else:
        pixels = pixels[None]
    row, col = (float(v) for v in getattr(dataset, "PixelSpacing", (1.0, 1.0)))
    spacing: tuple[float, ...] = (row, col)
    if pixels.ndim == 4:
        spacing = (float(getattr(dataset, "SliceThickness", 1.0) or 1.0), row, col)
    meta = {
        "format": "dicom",
        "modality": str(getattr(dataset, "Modality", "")),
        "patient_id": str(getattr(dataset, "PatientID", "")),
        "study_uid": str(getattr(dataset, "StudyInstanceUID", "")),
        "series_uid": str(getattr(dataset, "SeriesInstanceUID", "")),
        "photometric": str(getattr(dataset, "PhotometricInterpretation", "")),
    }
    return ImageData(pixels, spacing, meta)


def load_dicom_series(directory: str | Path, apply_voi: bool = False) -> ImageData:
    """Load the DICOM slices in ``directory`` as one volume ordered along the slice normal."""
    import pydicom
    from pydicom.errors import InvalidDicomError

    files = sorted(
        p for p in Path(directory).iterdir() if p.is_file() and not p.name.startswith(".")
    )
    slices = []
    for file in files:
        try:
            slices.append(pydicom.dcmread(file))
        except InvalidDicomError:
            continue
    if not slices:
        raise ValueError(f"no DICOM files in {directory}")
    series = {str(getattr(s, "SeriesInstanceUID", "")) for s in slices}
    if len(series) > 1:
        raise ValueError(f"{directory} holds {len(series)} series; expected one")

    orientation = np.asarray(slices[0].ImageOrientationPatient, dtype=float)
    normal = np.cross(orientation[:3], orientation[3:])
    positions = [float(np.dot(normal, np.asarray(s.ImagePositionPatient, float))) for s in slices]
    order = np.argsort(positions)
    slices = [slices[i] for i in order]
    positions = [positions[i] for i in order]

    volume = np.stack([_dicom_pixels(s, apply_voi) for s in slices])[None]
    row, col = (float(v) for v in slices[0].PixelSpacing)
    gaps = np.diff(positions)
    if len(gaps) and not np.allclose(gaps, gaps[0], rtol=1e-3, atol=1e-3):
        raise ValueError(f"{directory}: slices are unevenly spaced; resample before training")
    depth = float(gaps[0]) if len(gaps) else float(getattr(slices[0], "SliceThickness", 1.0))
    meta = {
        "format": "dicom-series",
        "modality": str(getattr(slices[0], "Modality", "")),
        "patient_id": str(getattr(slices[0], "PatientID", "")),
        "series_uid": series.pop(),
        "num_slices": len(slices),
    }
    return ImageData(volume.astype(np.float32), (abs(depth), row, col), meta)


def load_nifti(path: str | Path, canonical: bool = True) -> ImageData:
    """Load a NIfTI volume, reoriented to RAS+ and transposed to ``(z, y, x)``.

    4D images (for example multi-sequence MRI stored as one file) put the fourth axis first
    as channels.
    """
    import nibabel as nib
    from nibabel.spatialimages import SpatialImage

    image = nib.load(str(path))
    if not isinstance(image, SpatialImage):
        raise ValueError(f"{path} is not a NIfTI image with a spatial affine")
    if canonical:
        image = nib.as_closest_canonical(image)
    data = np.asarray(image.dataobj, dtype=np.float32)
    zooms = tuple(float(z) for z in image.header.get_zooms()[:3])
    if data.ndim == 2:
        data = data[..., None]
    if data.ndim == 3:
        array = data.transpose(2, 1, 0)[None]
    elif data.ndim == 4:
        array = data.transpose(3, 2, 1, 0)
    else:
        raise ValueError(f"{path}: expected a 3D or 4D NIfTI image, got {data.ndim}D")
    meta = {
        "format": "nifti",
        "orientation": "".join(nib.aff2axcodes(image.affine)),
        "affine": image.affine.tolist(),
    }
    return ImageData(np.ascontiguousarray(array), zooms[::-1], meta)


def load_raster(path: str | Path, mode: str | None = None) -> ImageData:
    """Load PNG, JPEG, TIFF, or BMP. 16-bit grayscale PNGs keep their full range."""
    from PIL import Image

    with Image.open(path) as image:
        if mode is not None:
            image = image.convert(mode)
        array = np.asarray(image)
    array = array.astype(np.float32)
    array = array[None] if array.ndim == 2 else np.moveaxis(array, -1, 0)
    return ImageData(np.ascontiguousarray(array), (1.0, 1.0), {"format": "raster"})


__all__ = [
    "ImageData",
    "load_dicom",
    "load_dicom_series",
    "load_image",
    "load_nifti",
    "load_raster",
]
