from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import UID, ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid


def write_dicom(path: Path, pixels: np.ndarray, **tags: str) -> Path:
    """A minimal monochrome DICOM file; ``tags`` set UIDs or ``PatientID``."""
    sop = UID(tags.pop("SOPInstanceUID", generate_uid()))
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = sop
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = sop
    ds.StudyInstanceUID = tags.pop("StudyInstanceUID", generate_uid())
    ds.SeriesInstanceUID = tags.pop("SeriesInstanceUID", generate_uid())
    ds.PatientID = tags.pop("PatientID", "anonymous")
    for key, value in tags.items():
        setattr(ds, key, value)
    ds.Modality = "OT"
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    return path
