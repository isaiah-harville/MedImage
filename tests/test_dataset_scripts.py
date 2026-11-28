import csv
import hashlib
import importlib.util
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from opendiagnostics.data import CSVImageDataModule, check_disjoint

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/datasets"


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))

    def load(name: str):
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        assert spec
        assert spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


class Response(io.BytesIO):
    def __init__(self, body: bytes, status: int) -> None:
        super().__init__(body)
        self.status = status


class FakeServer:
    """Serves bytes like an HTTP server that honours Range requests."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.ranges: list[str | None] = []

    def __call__(self, request):
        header = request.get_header("Range")
        self.ranges.append(header)
        start = int(header.removeprefix("bytes=").removesuffix("-")) if header else 0
        return Response(self.body[start:], 206 if header else 200)


def test_download_resumes_a_partial_file_and_checks_size_and_md5(script, tmp_path):
    fetch = script("_download")
    body = b"0123456789" * 1000
    dest = tmp_path / "archive.zip"
    (tmp_path / "archive.zip.part").write_bytes(body[:4000])
    server = FakeServer(body)
    fetch.download(
        "https://example/archive.zip", dest, len(body), hashlib.md5(body).hexdigest(), server
    )
    assert dest.read_bytes() == body
    assert server.ranges == ["bytes=4000-"]
    assert not (tmp_path / "archive.zip.part").exists()


def test_download_skips_a_file_that_already_matches(script, tmp_path):
    fetch = script("_download")
    dest = tmp_path / "labels.csv"
    dest.write_bytes(b"abc")
    server = FakeServer(b"abc")
    fetch.download("https://example/labels.csv", dest, 3, hashlib.md5(b"abc").hexdigest(), server)
    assert server.ranges == []


def test_download_of_the_wrong_size_is_an_error_and_is_discarded(script, tmp_path):
    fetch = script("_download")
    dest = tmp_path / "archive.zip"
    with pytest.raises(RuntimeError, match="do not match the published size 10"):
        fetch.download("https://example/archive.zip", dest, 10, None, FakeServer(b"short"))
    assert not dest.exists()
    assert not (tmp_path / "archive.zip.part").exists()


def test_extract_runs_once_and_can_delete_the_archive(script, tmp_path):
    fetch = script("_download")
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("train/x.jpg", b"x")
    into = tmp_path / "images"
    fetch.extract(archive, into)
    (into / "train/x.jpg").unlink()
    fetch.extract(archive, into, delete_archive=True)
    assert not (into / "train/x.jpg").exists()
    assert not archive.exists()


def _jpegs(root: Path, names: list[str]) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        Image.fromarray(np.full((8, 10, 3), 200, np.uint8)).save(root / f"{name}.jpg")
    return {name: root / f"{name}.jpg" for name in names}


def _groups_per_split(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[row["split"]].add(row["patient_id"])
    return groups


def _assert_groups_disjoint(rows: list[dict[str, str]]) -> None:
    splits: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        splits[row["split"]].append(index)
    check_disjoint(splits, [r["patient_id"] for r in rows])


def test_isic2020_manifest_drops_duplicates_and_splits_by_patient(script, tmp_path):
    isic = script("isic2020")
    truth = [
        {
            "image_name": f"ISIC_{i:03d}",
            "patient_id": f"IP_{i // 3:03d}",
            "target": str(int(i % 9 == 0)),
        }
        for i in range(60)
    ]
    dup = tmp_path / "dups.csv"
    dup.write_text("image_name_1,image_name_2\nISIC_001,ISIC_002\nISIC_009,ISIC_010\n")
    duplicates = isic.read_duplicates(dup, {r["image_name"] for r in truth})
    images = _jpegs(tmp_path / "images/train", [r["image_name"] for r in truth])
    rows = isic.manifest_rows(truth, duplicates, images, tmp_path)
    paths = {r["path"] for r in rows}
    assert "images/train/ISIC_001.jpg" in paths
    assert "images/train/ISIC_002.jpg" not in paths
    assert not {"images/train/ISIC_009.jpg", "images/train/ISIC_010.jpg"} & paths
    assert len(rows) == 57
    assert {r["label"] for r in rows} == {"benign", "melanoma"}
    assert set(_groups_per_split(rows)) == {"train", "val", "test"}
    _assert_groups_disjoint(rows)


def _jpeg_zip(names: list[str], folder: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        for name in names:
            image = io.BytesIO()
            Image.fromarray(np.full((8, 10, 3), 200, np.uint8)).save(image, format="JPEG")
            handle.writestr(f"{folder}/{name}.jpg", image.getvalue())
    return buffer.getvalue()


class Routes:
    """A fake web server: one body per URL, and a log of the URLs requested."""

    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies
        self.requested: list[str] = []

    def __call__(self, request):
        url = request if isinstance(request, str) else request.full_url
        self.requested.append(url)
        return Response(self.bodies[url], 200)

    def pin(self, url: str, name: str) -> tuple[str, int, str]:
        body = self.bodies[url]
        return name, len(body), hashlib.md5(body).hexdigest()


def test_isic2020_script_downloads_writes_the_manifest_and_reruns_offline(
    script, tmp_path, monkeypatch
):
    isic = script("isic2020")
    names = [f"ISIC_{i:03d}" for i in range(30)]
    truth = "image_name,patient_id,target\n" + "".join(
        f"{n},IP_{i // 2},{int(i % 5 == 0)}\n" for i, n in enumerate(names)
    )
    files = {
        "ISIC_2020_Training_JPEG.zip": _jpeg_zip(names, "train"),
        "ISIC_2020_Training_GroundTruth.csv": truth.encode(),
        "ISIC_2020_Training_Duplicates.csv": b"image_name_1,image_name_2\nISIC_001,ISIC_003\n",
    }
    server = Routes({f"{isic.BASE}/{name}": body for name, body in files.items()})
    monkeypatch.setattr("urllib.request.urlopen", server)
    for constant, name in (
        ("IMAGES", "ISIC_2020_Training_JPEG.zip"),
        ("GROUND_TRUTH", "ISIC_2020_Training_GroundTruth.csv"),
        ("DUPLICATES", "ISIC_2020_Training_Duplicates.csv"),
    ):
        monkeypatch.setattr(isic, constant, server.pin(f"{isic.BASE}/{name}", name))
    root = tmp_path / "isic2020"
    monkeypatch.setattr("sys.argv", ["isic2020.py", str(root), "--delete-archives"])
    isic.main()
    first = (root / "manifest.csv").read_text()
    assert len(server.requested) == 3
    assert not (root / "ISIC_2020_Training_JPEG.zip").exists()
    isic.main()
    assert len(server.requested) == 3
    assert (root / "manifest.csv").read_text() == first
    assert first.count("\n") == 30
    assert "images_768/ISIC_000.jpg" in first


def test_shrink_caps_the_longer_side_keeps_the_aspect_and_reuses_copies(script, tmp_path):
    download = script("_download")
    wide, small = tmp_path / "wide.jpg", tmp_path / "small.jpg"
    Image.new("RGB", (2000, 1000), "red").save(wide)
    Image.new("RGB", (300, 200), "blue").save(small)
    out = download.shrink({"wide": wide, "small": small}, tmp_path / "out", 768, workers=1)
    with Image.open(out["wide"]) as image:
        assert image.size == (768, 384)
    with Image.open(out["small"]) as image:
        assert image.size == (300, 200)
    stamp = out["wide"].stat().st_mtime_ns
    download.shrink({"wide": wide}, tmp_path / "out", 768, workers=1)
    assert out["wide"].stat().st_mtime_ns == stamp


def test_isic2019_manifest_groups_by_lesion_and_reads_one_hot_labels(script, tmp_path):
    isic = script("isic2019")
    classes = isic.CLASSES
    truth, metadata = [], []
    for i in range(48):
        label = classes[i % 4]
        truth.append(
            {"image": f"ISIC_{i:03d}", **{c: str(float(c == label)) for c in classes}, "UNK": "0.0"}
        )
        metadata.append({"image": f"ISIC_{i:03d}", "lesion_id": f"L{i // 2}" if i < 40 else ""})
    images = _jpegs(tmp_path / "images", [r["image"] for r in truth])
    rows = isic.manifest_rows(truth, metadata, images, tmp_path)
    assert [r["label"] for r in rows[:4]] == ["MEL", "NV", "BCC", "AK"]
    assert rows[0]["patient_id"] == rows[1]["patient_id"] == "L0"
    assert rows[45]["patient_id"] == "image:ISIC_045"
    _assert_groups_disjoint(rows)


def test_isic2019_row_without_exactly_one_diagnosis_is_an_error(script):
    isic = script("isic2019")
    with pytest.raises(RuntimeError, match="has 0 training diagnoses"):
        isic.diagnosis({"image": "ISIC_1", **dict.fromkeys(isic.CLASSES, "0.0")})


def test_ham10000_manifest_uses_the_isic2018_test_set_and_splits_the_rest_by_lesion(
    script, tmp_path
):
    ham = script("ham10000")
    classes = ham.CLASSES
    metadata = [
        {"lesion_id": f"HAM_{i // 2:04d}", "image_id": f"ISIC_{i:04d}", "dx": classes[(i // 2) % 7]}
        for i in range(84)
    ]
    test_truth = [
        {"lesion_id": f"HAMTEST_{i:07d}", "image_id": f"ISIC_9{i:03d}", "dx": classes[i % 7]}
        for i in range(7)
    ]
    images = _jpegs(tmp_path / "images", [r["image_id"] for r in [*metadata, *test_truth]])
    rows = ham.manifest_rows(metadata, test_truth, images, tmp_path)
    test = [r for r in rows if r["split"] == "test"]
    assert [r["label"] for r in test] == list(classes)
    assert {r["path"] for r in test} == {f"images/ISIC_9{i:03d}.jpg" for i in range(7)}
    assert set(_groups_per_split(rows)) == {"train", "val", "test"}
    _assert_groups_disjoint(rows)


def _dicom(path: Path, patient: str, side: str) -> None:
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientID, ds.ImageLaterality, ds.Modality = patient, side, "MG"
    pixels = np.zeros((12, 10), np.uint8)
    pixels[:6] = 255
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelData = pixels.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)


def test_cmmd_labels_each_dicom_by_patient_and_laterality(script, tmp_path):
    from openpyxl import Workbook

    cmmd = script("cmmd")
    book = Workbook()
    sheet = book.active
    assert sheet
    sheet.append(["ID1", "LeftRight", "Age", "number", "abnormality", "classification", "subtype"])
    for i in range(20):
        sheet.append([f"D1-{i:04d}", "L", 50, 2, "mass", "Malignant" if i % 2 else "Benign", None])
    book.save(tmp_path / "clinical.xlsx")
    labels = cmmd.read_clinical(tmp_path / "clinical.xlsx")
    assert labels[("D1-0001", "L")] == "malignant"

    images = []
    for i in range(20):
        for side in "LR":
            path = tmp_path / f"dicom/{i}{side}/1-1.dcm"
            _dicom(path, f"D1-{i:04d}", side)
            images.append((path, *cmmd.laterality(path)))
    rows = cmmd.manifest_rows(images, labels, tmp_path)
    assert len(rows) == 20
    assert all(r["path"].endswith("L/1-1.dcm") for r in rows)
    _assert_groups_disjoint(rows)


def test_cmmd_manifest_serves_dicoms_channel_first(tmp_path):
    _dicom(tmp_path / "dicom/a/1-1.dcm", "D1-0001", "L")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,patient_id,label,split\ndicom/a/1-1.dcm,D1-0001,malignant,train\n")
    data = CSVImageDataModule(
        manifest=str(manifest), split_column="split", class_names=["benign", "malignant"]
    )
    data.setup()
    batch = next(iter(data.loader("train")))
    assert batch["image"].shape == (1, 1, 12, 10)
    assert batch["image"][0, 0, 0, 0] == 1
    assert batch["image"][0, 0, -1, 0] == 0


def test_ham10000_script_fetches_dataverse_files_and_original_csvs(script, tmp_path, monkeypatch):
    ham = script("ham10000")
    train = [f"ISIC_{i:04d}" for i in range(28)]
    test = [f"ISIC_9{i:03d}" for i in range(7)]
    metadata = "lesion_id,image_id,dx\n" + "".join(
        f"HAM_{i // 2},{n},{ham.CLASSES[(i // 2) % 7]}\n" for i, n in enumerate(train)
    )
    truth = "lesion_id,image_id,dx,dx_type,age,sex,localization,dataset\n" + "".join(
        f"HAMTEST_{i:07d},{n},{ham.CLASSES[i]},histo,40.0,female,back,vidir_molemax\n"
        for i, n in enumerate(test)
    )
    archives = {
        "HAM10000_images_part_1.zip": _jpeg_zip(train[:14], "HAM10000_images_part_1"),
        "HAM10000_images_part_2.zip": _jpeg_zip(train[14:], "HAM10000_images_part_2"),
        "ISIC2018_Task3_Test_Images.zip": _jpeg_zip(test, "ISIC2018_Task3_Test_Images"),
    }
    server = Routes({})
    pinned = {}
    for number, (name, body) in enumerate(archives.items()):
        server.bodies[f"{ham.ACCESS}/{number}"] = body
        pinned[name] = (number, len(body), hashlib.md5(body).hexdigest())
    monkeypatch.setattr(ham, "ARCHIVES", pinned)
    for number, (constant, name, text) in enumerate(
        (
            ("METADATA", "HAM10000_metadata.csv", metadata),
            ("TEST_TRUTH", "ISIC2018_Task3_Test_GroundTruth.csv", truth),
        ),
        10,
    ):
        server.bodies[f"{ham.ACCESS}/{number}?format=original"] = text.encode()
        monkeypatch.setattr(ham, constant, (name, number, len(text.encode())))
    monkeypatch.setattr("urllib.request.urlopen", server)
    root = tmp_path / "ham10000"
    monkeypatch.setattr("sys.argv", ["ham10000.py", str(root)])
    ham.main()
    with (root / "manifest.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 35
    assert [r["label"] for r in rows if r["split"] == "test"] == list(ham.CLASSES)
    assert (root / "HAM10000_images_part_1.zip").exists()


def test_cmmd_script_downloads_each_series_and_labels_by_breast(script, tmp_path, monkeypatch):
    from openpyxl import Workbook

    cmmd = script("cmmd")
    series, bodies = [], {}
    for i in range(10):
        uid = f"1.2.{i}"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as handle:
            for view, side in enumerate("LR", 1):
                source = tmp_path / "src" / uid / f"1-{view}.dcm"
                _dicom(source, f"D1-{i:04d}", side)
                handle.write(source, f"CMMD/1-{view}.dcm")
        bodies[f"{cmmd.API}/getImage?SeriesInstanceUID={uid}"] = buffer.getvalue()
        series.append({"SeriesInstanceUID": uid, "PatientID": f"D1-{i:04d}", "ImageCount": 2})
    bodies[f"{cmmd.API}/getSeries?Collection=CMMD"] = json.dumps(series).encode()
    book = Workbook()
    sheet = book.active
    assert sheet
    sheet.append(["ID1", "LeftRight", "Age", "number", "abnormality", "classification", "subtype"])
    for i in range(10):
        sheet.append([f"D1-{i:04d}", "R", 50, 2, "mass", ["Benign", "Malignant"][i % 2], None])
    book.save(tmp_path / "clinical.xlsx")
    xlsx = (tmp_path / "clinical.xlsx").read_bytes()
    bodies[cmmd.CLINICAL[0]] = xlsx
    monkeypatch.setattr(cmmd, "CLINICAL", (cmmd.CLINICAL[0], len(xlsx)))
    server = Routes(bodies)
    monkeypatch.setattr("urllib.request.urlopen", server)
    root = tmp_path / "cmmd"
    monkeypatch.setattr("sys.argv", ["cmmd.py", str(root), "--delete-archives"])
    cmmd.main()
    with (root / "manifest.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert all(r["path"].endswith("1-2.dcm") for r in rows)
    assert {r["label"] for r in rows} == {"benign", "malignant"}
    requested = len(server.requested)
    cmmd.main()
    assert len(server.requested) == requested


def test_ham10000_leaves_out_a_few_labelled_images_with_no_file_but_not_many(script, tmp_path):
    ham = script("ham10000")
    metadata = [
        {
            "lesion_id": f"HAM_{i // 2:04d}",
            "image_id": f"ISIC_{i:04d}",
            "dx": ham.CLASSES[(i // 2) % 7],
        }
        for i in range(84)
    ]
    test_truth = [
        {"lesion_id": f"HAMTEST_{i:07d}", "image_id": f"ISIC_9{i:03d}", "dx": ham.CLASSES[i % 7]}
        for i in range(7)
    ]
    images = _jpegs(tmp_path / "images", [r["image_id"] for r in [*metadata, *test_truth]])
    del images["ISIC_9000"]
    rows = ham.manifest_rows(metadata, test_truth, images, tmp_path)
    assert len([r for r in rows if r["split"] == "test"]) == 6
    for name in list(images)[: ham.MAX_MISSING + 1]:
        del images[name]
    with pytest.raises(RuntimeError, match="labelled images are not on disk"):
        ham.manifest_rows(metadata, test_truth, images, tmp_path)
