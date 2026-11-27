import csv
import hashlib
import importlib
import io
import json
import tarfile

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def load(root, monkeypatch):
    """Import a module from scripts/datasets, the way running the script there would."""
    monkeypatch.syspath_prepend(str(root / "scripts" / "datasets"))
    return importlib.import_module


def _served(body: bytes):
    return lambda request: io.BytesIO(body)


def test_download_checks_a_published_sha256(load, tmp_path):
    fetch = load("_download")
    body = b"kidney" * 100
    dest = tmp_path / "case.nii.gz"
    digest = hashlib.sha256(body).hexdigest()
    fetch.download(
        "https://example.test/f", dest, size=len(body), opener=_served(body), sha256=digest
    )
    assert dest.read_bytes() == body
    with pytest.raises(RuntimeError, match="do not match"):
        fetch.download(
            "https://example.test/f", tmp_path / "bad", opener=_served(b"corrupt"), sha256=digest
        )
    assert not (tmp_path / "bad.part").exists()


def _tar(path, members):
    with tarfile.open(path, "w:gz") as bundle:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))


def test_extract_unpacks_chosen_tar_members_and_deletes_the_archive(load, tmp_path):
    fetch = load("_download")
    archive = tmp_path / "a.tar.gz"
    _tar(archive, {"x/keep.txt": b"1", "x/._keep.txt": b"2"})
    out = fetch.extract(archive, tmp_path / "out", True, keep=lambda n: "/._" not in n)
    assert (out / "x" / "keep.txt").read_bytes() == b"1"
    assert not (out / "x" / "._keep.txt").exists()
    assert not archive.exists()
    assert fetch.extract(archive, tmp_path / "out") == out


def test_extract_refuses_tar_members_outside_the_destination(load, tmp_path):
    archive = tmp_path / "evil.tar.gz"
    _tar(archive, {"../escape.txt": b"x"})
    with pytest.raises(tarfile.OutsideDestinationError):
        load("_download").extract(archive, tmp_path / "out")


def test_totalsegmentator_merges_masks_in_class_order(load, tmp_path):
    totalsegmentator = load("totalsegmentator")
    case = tmp_path / "s0001"
    (case / "segmentations").mkdir(parents=True)
    affine = np.diag([1.5, 1.5, 2.0, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 3), np.int16), affine), case / "ct.nii.gz")
    for name in totalsegmentator.CLASSES:
        mask = np.zeros((4, 4, 3), np.uint8)
        if name == "spleen":
            mask[0, 0, 0] = mask[1, 1, 1] = 1
        if name == "liver":
            mask[1, 1, 1] = 1
        nib.save(nib.Nifti1Image(mask, affine), case / "segmentations" / f"{name}.nii.gz")
    assert totalsegmentator.merge(case) == 1
    label = nib.Nifti1Image.load(case / "label.nii.gz")
    values = np.asarray(label.dataobj)
    assert values[0, 0, 0] == 1
    assert values[1, 1, 1] == totalsegmentator.CLASSES.index("liver") + 1
    assert np.allclose(label.affine, affine)


def test_msd_manifest_lists_the_labelled_training_cases(load, tmp_path, monkeypatch):
    msd = load("msd")
    listing = {
        "training": [
            {"image": f"./imagesTr/c{i}.nii.gz", "label": f"./labelsTr/c{i}.nii.gz"}
            for i in range(10)
        ]
    }
    monkeypatch.setattr(
        msd,
        "download",
        lambda url, dest, **kwargs: _tar(
            dest, {"Task09_Spleen/dataset.json": json.dumps(listing).encode()}
        ),
    )
    msd.prepare("Task09_Spleen", tmp_path, delete=True)
    with (tmp_path / "Task09_Spleen" / "manifest.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0] | {"split": ""} == {
        "id": "c0",
        "image": "imagesTr/c0.nii.gz",
        "label": "labelsTr/c0.nii.gz",
        "patient_id": "c0",
        "split": "",
    }
    assert sorted({r["split"] for r in rows}) == ["test", "train", "val"]
    assert not (tmp_path / "Task09_Spleen.tar").exists()


def test_acdc_reads_the_phase_frames_from_info_cfg(load):
    assert load("acdc").info("ED: 1\nES: 12\nGroup: DCM\nHeight: 184.0\n") == {
        "ED": "1",
        "ES": "12",
        "Group": "DCM",
        "Height": "184.0",
    }
