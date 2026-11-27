"""Shared helpers for the 3D detection download scripts (luna16, lidc_idri, duke_breast_mri).

They download resumably, read TCIA's public REST API anonymously, convert scans to NIfTI
with SimpleITK, and write the manifest that ``VolumeManifestDataModule`` reads. Boxes are
``(x1, y1, z1, x2, y2, z2)`` in world LPS millimetres, the frame of DICOM and ITK.

SimpleITK and pydicom are imported where they are used, so the tests can import the pure
functions without them.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from _download import retrying

TCIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
USER_AGENT = "opendiagnostics-dataset-script"
CHUNK = 1 << 20


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _request(url: str, start: int = 0) -> urllib.request.Request:
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.Request(url, headers=headers)


def get_json(url: str) -> Any:
    with urllib.request.urlopen(_request(url), timeout=120) as response:
        return json.load(response)


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


@retrying
def download(url: str, dest: Path, md5: str | None = None, size: int | None = None) -> Path:
    """Download ``url`` to ``dest``, resuming a ``.part`` file left by an interrupted run.

    A finished file is kept when its size matches. With ``md5`` the download is checked and
    a mismatch deletes it and raises, so the next run starts over.
    """
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    with urllib.request.urlopen(_request(url, start), timeout=300) as response:
        resumed = start and response.status == 206
        if start and not resumed:
            log(f"{url} does not support resuming; starting over")
        with part.open("ab" if resumed else "wb") as handle:
            shutil.copyfileobj(response, handle, CHUNK)
    if size is not None and part.stat().st_size != size:
        raise OSError(
            f"{dest.name}: got {part.stat().st_size} bytes, expected {size}; rerun to resume"
        )
    if md5 is not None and (actual := md5sum(part)) != md5:
        part.unlink()
        raise OSError(
            f"{dest.name}: MD5 {actual} does not match {md5}; deleted, rerun to fetch it again"
        )
    os.replace(part, dest)
    return dest


def tcia_series(collection: str, modality: str | None = None) -> list[dict[str, Any]]:
    query = f"Collection={collection}" + (f"&Modality={modality}" if modality else "")
    return get_json(f"{TCIA}/getSeries?{query}")


def tcia_dicom(series_uid: str, image_count: int, folder: Path, delete_archive: bool) -> Path:
    """One series' DICOM files in ``folder``, fetched as TCIA's zip unless already there."""
    if folder.is_dir() and len(list(folder.glob("*.dcm"))) == image_count:
        return folder
    archive = download(
        f"{TCIA}/getImage?SeriesInstanceUID={series_uid}", folder.with_suffix(".zip")
    )
    shutil.rmtree(folder, ignore_errors=True)
    extract(archive, folder)
    found = len(list(folder.glob("*.dcm")))
    if found != image_count:
        archive.unlink()
        raise OSError(f"series {series_uid}: {found} DICOM files, TCIA lists {image_count}; rerun")
    if delete_archive:
        archive.unlink()
    return folder


def tcia_volume(
    series: dict[str, Any], root: Path, integer: bool, delete_archive: bool
) -> dict[str, dict[str, list[float]]]:
    """Convert one TCIA series to ``images/<uid>.nii.gz`` and keep its slice geometry in
    ``geometry/<uid>.json``, so boxes can be rebuilt later without the DICOM files. Skips
    series already converted; the extracted DICOM files are removed afterwards."""
    uid = series["SeriesInstanceUID"]
    image, cached = root / "images" / f"{uid}.nii.gz", root / "geometry" / f"{uid}.json"
    if image.exists() and cached.exists():
        return json.loads(cached.read_text())
    folder = tcia_dicom(uid, int(series["ImageCount"]), root / "dicom" / uid, delete_archive)
    files = sorted_series(folder)
    geometry = slice_geometry(files)
    write_nifti(files, image, integer=integer)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(geometry) + "\n")
    shutil.rmtree(folder)
    return geometry


def extract(archive: Path, folder: Path, members: Iterable[str] | None = None) -> None:
    """Extract ``archive`` into ``folder``, refusing members that would land outside it."""
    root = folder.resolve()
    with zipfile.ZipFile(archive) as handle:
        for name in members if members is not None else handle.namelist():
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise OSError(f"{archive.name} has a member outside the folder: {name}")
            handle.extract(name, root)


def write_nifti(source: Path | Sequence[Path], dest: Path, integer: bool) -> None:
    """Convert a MetaImage file or a DICOM series (a list of slice files) to NIfTI, keeping
    the geometry. ``integer`` stores 16-bit integers (CT in HU)."""
    import SimpleITK as sitk  # ty: ignore[import-not-found]

    if isinstance(source, Path):
        image = sitk.ReadImage(str(source))
    else:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(p) for p in source])
        image = reader.Execute()
    if integer:
        image = sitk.Cast(image, sitk.sitkInt16)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part.nii.gz")
    sitk.WriteImage(image, str(part), useCompression=True)
    os.replace(part, dest)


def sorted_series(folder: Path) -> list[Path]:
    """The slice files of one DICOM series in GDCM's order along the slice normal."""
    import SimpleITK as sitk  # ty: ignore[import-not-found]

    return [Path(p) for p in sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder))]


def slice_geometry(files: Sequence[Path]) -> dict[str, dict[str, list[float]]]:
    """Each slice's position, orientation and pixel spacing, keyed by SOP Instance UID."""
    import pydicom

    geometry = {}
    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        geometry[str(ds.SOPInstanceUID)] = {
            "position": [float(v) for v in ds.ImagePositionPatient],
            "orientation": [float(v) for v in ds.ImageOrientationPatient],
            "spacing": [float(v) for v in ds.PixelSpacing],
        }
    return geometry


def _normal(orientation: Sequence[float]) -> list[float]:
    r, c = orientation[:3], orientation[3:]
    return [r[1] * c[2] - r[2] * c[1], r[2] * c[0] - r[0] * c[2], r[0] * c[1] - r[1] * c[0]]


def _height(geometry: dict[str, list[float]]) -> float:
    normal = _normal(geometry["orientation"])
    return sum(p * n for p, n in zip(geometry["position"], normal, strict=True))


def by_height(slices: Iterable[dict[str, list[float]]]) -> list[dict[str, list[float]]]:
    """Slices in ascending position along the slice normal."""
    return sorted(slices, key=_height)


def slice_gap(slices: Iterable[dict[str, list[float]]]) -> float:
    """The median distance between neighbouring slices along the slice normal."""
    heights = [_height(s) for s in by_height(slices)]
    gaps = sorted(b - a for a, b in itertools.pairwise(heights) if b - a > 1e-3)
    if not gaps:
        raise ValueError("a series needs two slices at different positions")
    return gaps[len(gaps) // 2]


def pixel_box(
    points: Sequence[tuple[dict[str, list[float]], float, float]], gap: float
) -> list[float]:
    """The world LPS box covering whole pixels: each point is ``(slice, column, row)`` with
    0-based pixel indices, and the box reaches half a pixel past the outermost pixel
    centers in plane and half a slice gap along the normal."""
    corners = []
    for geometry, column, row in points:
        position, o = geometry["position"], geometry["orientation"]
        row_spacing, column_spacing = geometry["spacing"]
        for dc in (-0.5, 0.5):
            for dr in (-0.5, 0.5):
                corners.append(
                    [
                        position[k]
                        + (column + dc) * column_spacing * o[k]
                        + (row + dr) * row_spacing * o[3 + k]
                        for k in range(3)
                    ]
                )
    normal = _normal(points[0][0]["orientation"])
    low = [min(c[k] for c in corners) - abs(normal[k]) * gap / 2 for k in range(3)]
    high = [max(c[k] for c in corners) + abs(normal[k]) * gap / 2 for k in range(3)]
    return [*low, *high]


def cube(center: Sequence[float], diameter: float) -> list[float]:
    half = diameter / 2
    return [*(c - half for c in center), *(c + half for c in center)]


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(manifest, indent=1) + "\n")
    os.replace(part, path)
    volumes = manifest["volumes"]
    lesions = sum(len(v["boxes"]) for v in volumes)
    log(f"wrote {path}: {len(volumes)} volumes, {lesions} boxes")
