import csv
import hashlib
import importlib
import io
import json
import tarfile
import zipfile

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
    out = fetch.extract(
        archive, tmp_path / "out", True, members=lambda n: None if "/._" in n else n
    )
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


def _nifti_bytes(tmp_path, array):
    path = tmp_path / "scratch.nii.gz"
    nib.save(nib.Nifti1Image(array, np.eye(4)), path)
    return path.read_bytes()


def _rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_kits23_downloads_checked_images_and_repository_masks(load, tmp_path, monkeypatch):
    from huggingface_hub.hf_api import RepoFile

    kits23 = load("kits23")
    cases = [f"case_{n:05d}" for n in range(10)]
    image = b"ct volume"
    digest = hashlib.sha256(image).hexdigest()
    lfs = {"size": len(image), "oid": digest, "pointerSize": 1}
    listing = [
        RepoFile(path=f"images/{c}.nii.gz", size=len(image), oid="x", lfs=lfs) for c in cases
    ]

    class Api:
        def list_repo_tree(self, *args, **kwargs):
            return listing

    calls = []

    def fake_download(url, dest, size=None, md5=None, opener=None, sha256=None):
        calls.append((url, sha256))
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url == kits23.LABELS_URL:
            _tar(dest, {f"kits23-x/dataset/{c}/segmentation.nii.gz": c.encode() for c in cases})
        else:
            dest.write_bytes(image)
        return dest

    monkeypatch.setattr(kits23, "HfApi", Api)
    monkeypatch.setattr(kits23, "TRAINING_CASES", list(range(10)))
    monkeypatch.setattr(kits23, "download", fake_download)
    monkeypatch.setattr("sys.argv", ["kits23.py", str(tmp_path), "--delete-archives"])
    kits23.main()
    rows = _rows(tmp_path / "manifest.csv")
    assert [r["id"] for r in rows] == cases
    assert (tmp_path / rows[3]["label"]).read_bytes() == b"case_00003"
    assert all(sha == digest for url, sha in calls if url != kits23.LABELS_URL)
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    assert not list(tmp_path.glob("*.tar.gz"))


def test_acdc_keeps_the_official_test_patients_and_both_phases(load, tmp_path, monkeypatch):
    acdc = load("acdc")
    patients = {"training": [f"patient{n:03d}" for n in range(1, 11)], "testing": ["patient101"]}

    def fake_get(path, **query):
        parent = query.get("parentId") or query.get("folderId") or ""
        if path == "folder" and query["parentType"] == "collection":
            return [{"_id": "db", "name": "database"}]
        if path == "folder" and parent == "db":
            return [{"_id": "training", "name": "training"}, {"_id": "testing", "name": "testing"}]
        if path == "folder":
            return [{"_id": p, "name": p} for p in patients[parent]]
        names = ["Info.cfg", f"{parent}_4d.nii.gz", f"{parent}_frame01.nii.gz"]
        names += [f"{parent}_frame01_gt.nii.gz", f"{parent}_frame12.nii.gz"]
        names += [f"{parent}_frame12_gt.nii.gz"]
        return [{"_id": f"{parent}/{n}", "name": n, "size": 1} for n in names]

    def fake_download(url, dest, size=None, **kwargs):
        assert not dest.name.endswith("_4d.nii.gz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        group = "NOR" if int(dest.parent.name[-3:]) % 2 else "DCM"
        dest.write_text(f"ED: 1\nES: 12\nGroup: {group}\n" if dest.name == "Info.cfg" else "x")
        return dest

    monkeypatch.setattr(acdc, "get", fake_get)
    monkeypatch.setattr(acdc, "download", fake_download)
    monkeypatch.setattr("sys.argv", ["acdc.py", str(tmp_path)])
    acdc.main()
    rows = _rows(tmp_path / "manifest.csv")
    assert len(rows) == 22
    assert {r["split"] for r in rows if r["patient_id"] == "patient101"} == {"test"}
    assert {r["split"] for r in rows if r["patient_id"] != "patient101"} == {"train", "val"}
    first = [r for r in rows if r["patient_id"] == "patient001"]
    assert [(r["phase"], r["image"]) for r in first] == [
        ("ED", "training/patient001/patient001_frame01.nii.gz"),
        ("ES", "training/patient001/patient001_frame12.nii.gz"),
    ]


def test_totalsegmentator_keeps_the_meta_csv_split(load, tmp_path, monkeypatch):
    totalsegmentator = load("totalsegmentator")
    mask = _nifti_bytes(tmp_path, np.zeros((2, 2, 2), np.uint8))
    ct = _nifti_bytes(tmp_path, np.zeros((2, 2, 2), np.int16))
    splits = {"s0001": "train", "s0002": "val", "s0003": "test"}
    meta = "image_id;age;split\n" + "".join(f"{c};50;{s}\n" for c, s in splits.items())

    def fake_download(url, dest, size=None, md5=None, **kwargs):
        assert (size, md5) == (totalsegmentator.SIZE, totalsegmentator.MD5)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w") as bundle:
            bundle.writestr("meta.csv", meta)
            for case in splits:
                bundle.writestr(f"{case}/ct.nii.gz", ct)
                for name in totalsegmentator.CLASSES:
                    bundle.writestr(f"{case}/segmentations/{name}.nii.gz", mask)
        return dest

    monkeypatch.setattr(totalsegmentator, "download", fake_download)
    monkeypatch.setattr("sys.argv", ["totalsegmentator.py", str(tmp_path)])
    totalsegmentator.main()
    rows = _rows(tmp_path / "manifest.csv")
    assert {r["id"]: r["split"] for r in rows} == splits
    assert all((tmp_path / r["label"]).is_file() for r in rows)


def test_picai_puts_every_sequence_and_a_binary_mask_on_the_t2w_grid(load, tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    picai = load("picai")
    images = {}
    for sequence, spacing in (("t2w", 0.5), ("adc", 2.0), ("hbv", 2.0)):
        size = 32 if sequence == "t2w" else 8
        image = sitk.GetImageFromArray(np.random.default_rng(0).random((4, size, size)))
        image.SetSpacing((spacing, spacing, 3.0))
        images[sequence] = tmp_path / f"10000_1000000_{sequence}.mha"
        sitk.WriteImage(image, str(images[sequence]))
    grades = np.zeros((4, 8, 8), np.uint8)
    grades[1:3, 2:5, 2:5] = 3
    mask = sitk.GetImageFromArray(grades)
    mask.SetSpacing((2.0, 2.0, 3.0))
    sitk.WriteImage(mask, str(tmp_path / "10000_1000000.nii.gz"))
    out = tmp_path / "nifti"
    picai.convert(sitk, "10000_1000000", images, tmp_path / "10000_1000000.nii.gz", out)
    shapes = {
        name: nib.Nifti1Image.load(out / f"10000_1000000_{name}.nii.gz").shape
        for name in ("t2w", "adc", "hbv", "label")
    }
    assert set(shapes.values()) == {(32, 32, 4)}
    label = np.asarray(nib.Nifti1Image.load(out / "10000_1000000_label.nii.gz").dataobj)
    assert set(np.unique(label)) == {0, 1}
