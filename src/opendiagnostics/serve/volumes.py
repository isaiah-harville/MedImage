"""Read an uploaded 3D volume from bytes, in memory: a NumPy ``.npy`` array or a NIfTI-1 or
NIfTI-2 file, gzipped or not.

Nothing is written to disk. Gzip is inflated with a cap, so a small upload cannot expand
into more memory than ``MAX_VOLUME_BYTES``.
"""

from __future__ import annotations

import io
import zlib

import nibabel as nib
import numpy as np
from nibabel.spatialimages import HeaderDataError
from nibabel.wrapstruct import WrapStructError

MAX_VOLUME_BYTES = 256 * 1024 * 1024
MAX_VOLUME_SIDE = 1024
NPY_MAGIC = b"\x93NUMPY"
GZIP_MAGIC = b"\x1f\x8b"


class VolumeError(ValueError):
    """The upload is not a volume this service can read; the message says why."""


def _inflate(data: bytes) -> bytes:
    inflater = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)
    try:
        out = inflater.decompress(data, MAX_VOLUME_BYTES + 1)
    except zlib.error as error:
        raise VolumeError("the gzip stream is corrupt") from error
    if len(out) > MAX_VOLUME_BYTES or inflater.unconsumed_tail:
        raise VolumeError(f"the volume unpacks to more than {MAX_VOLUME_BYTES // 2**20} MB")
    return out


def _nifti(data: bytes) -> np.ndarray:
    sizes = {int.from_bytes(data[:4], order) for order in ("little", "big")}
    image_class = nib.Nifti1Image if 348 in sizes else nib.Nifti2Image if 540 in sizes else None
    if image_class is None:
        raise VolumeError("upload a .npy array or a NIfTI (.nii or .nii.gz) volume")
    try:
        return np.asanyarray(image_class.from_bytes(data).dataobj)
    except (HeaderDataError, WrapStructError, ValueError, EOFError) as error:
        raise VolumeError(f"the NIfTI file could not be read: {error}") from error


def read_volume(data: bytes) -> np.ndarray:
    """The volume as a ``(D, H, W)`` array. NIfTI's ``(x, y, z)`` axes are reversed, so the
    first axis runs along z, the way the training volumes are stored."""
    if data.startswith(GZIP_MAGIC):
        data = _inflate(data)
    if data.startswith(NPY_MAGIC):
        try:
            volume = np.load(io.BytesIO(data), allow_pickle=False)
        except ValueError as error:
            raise VolumeError("the .npy file could not be read") from error
    else:
        volume = _nifti(data).transpose()
    volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise VolumeError(f"expected a 3D volume, got an array shaped {volume.shape}")
    if max(volume.shape) > MAX_VOLUME_SIDE:
        raise VolumeError(f"volumes may be at most {MAX_VOLUME_SIDE} voxels on a side")
    if not (np.issubdtype(volume.dtype, np.integer) or np.issubdtype(volume.dtype, np.floating)):
        raise VolumeError(f"expected numbers, got an array of {volume.dtype}")
    if not np.isfinite(volume).all():
        raise VolumeError("the volume contains NaN or infinite values")
    return volume
