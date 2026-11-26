import numpy as np
import pydicom
import pytest
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
from xpt import expect

from opendiagnostics.io import load_image


def _write_dicom(
    path, pixels, z=0.0, series_uid="1.2.3", photometric="MONOCHROME2", intercept=-1024.0
):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.PatientID = "P001"
    ds.SeriesInstanceUID = series_uid
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = intercept
    ds.PixelSpacing = [0.5, 0.75]
    ds.SliceThickness = 2.5
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, z]
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    pydicom.dcmwrite(path, ds, enforce_file_format=True)


def test_dicom_applies_rescale_to_hounsfield_units(tmp_path):
    pixels = np.array([[0, 1024], [2048, 1000]], dtype=np.uint16)
    _write_dicom(tmp_path / "a.dcm", pixels)
    image = load_image(tmp_path / "a.dcm")
    assert image.array.shape == (1, 2, 2)
    assert image.array[0].tolist() == [[-1024, 0], [1024, -24]]
    assert image.spacing == (0.5, 0.75)
    expect(
        {k: v for k, v in image.meta.items() if k != "series_uid"},
        {
            "format": "dicom",
            "modality": "CT",
            "patient_id": "P001",
            "study_uid": "",
            "photometric": "MONOCHROME2",
        },
    )


def test_dicom_monochrome1_is_inverted(tmp_path):
    _write_dicom(
        tmp_path / "a.dcm",
        np.array([[0, 10]], dtype=np.uint16),
        photometric="MONOCHROME1",
        intercept=0,
    )
    assert load_image(tmp_path / "a.dcm").array[0].tolist() == [[10, 0]]


def test_dicom_series_is_sorted_by_position(tmp_path):
    for i, z in enumerate([5.0, 0.0, 2.5]):  # written out of order
        _write_dicom(
            tmp_path / f"s{i}.dcm", np.full((2, 2), int(z * 10), np.uint16), z=z, intercept=0
        )
    image = load_image(tmp_path)
    assert image.array.shape == (1, 3, 2, 2)
    assert image.array[0, :, 0, 0].tolist() == [0, 25, 50]
    assert image.spacing == (2.5, 0.5, 0.75)


def test_dicom_series_rejects_uneven_spacing(tmp_path):
    for i, z in enumerate([0.0, 1.0, 5.0]):
        _write_dicom(tmp_path / f"s{i}.dcm", np.zeros((2, 2), np.uint16), z=z)
    with pytest.raises(ValueError, match="unevenly spaced"):
        load_image(tmp_path)


def test_nifti_is_canonical_and_slice_first(tmp_path):
    nib = pytest.importorskip("nibabel")
    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)  # (x, y, z)
    # A flipped x axis (LAS); loading should reorient to RAS.
    affine = np.diag([-2.0, 3.0, 4.0, 1.0])
    nib.save(nib.Nifti1Image(data, affine), tmp_path / "v.nii.gz")
    image = load_image(tmp_path / "v.nii.gz")
    assert image.array.shape == (1, 4, 3, 2)
    assert image.spacing == (4.0, 3.0, 2.0)
    assert image.meta["orientation"] == "RAS"
    np.testing.assert_array_equal(image.array[0], data[::-1].transpose(2, 1, 0))


def test_16_bit_png_keeps_range(tmp_path):
    pixels = np.array([[0, 40000]], dtype=np.uint16)
    Image.fromarray(pixels).save(tmp_path / "a.png")
    assert load_image(tmp_path / "a.png").array.max() == 40000


def test_unknown_format(tmp_path):
    (tmp_path / "a.xyz").write_text("")
    with pytest.raises(ValueError, match="unsupported"):
        load_image(tmp_path / "a.xyz")
