import csv
import hashlib
import importlib
import io
import tarfile
import zipfile

import pytest
from PIL import Image
from xpt import expect

from opendiagnostics.data import CSVImageDataModule


@pytest.fixture
def scripts(root, monkeypatch):
    monkeypatch.syspath_prepend(str(root / "scripts/datasets"))
    return importlib.import_module


class Response(io.BytesIO):
    def __init__(self, data: bytes, status: int, drop_after: int | None = None):
        super().__init__(data)
        self.status = status
        self.drop_after = drop_after

    def read(self, size=-1):
        if self.drop_after is not None and self.tell() >= self.drop_after:
            raise ConnectionResetError("connection reset by peer")
        if self.drop_after is not None:
            size = min(size, self.drop_after - self.tell())
        return super().read(size)


class Server:
    """Serves files by URL, honouring Range headers unless told not to."""

    def __init__(self, files, ranges=True, drop_after=None):
        self.files = files
        self.ranges = ranges
        self.drop_after = drop_after
        self.requests = []

    def __call__(self, request):
        self.requests.append((request.full_url, request.get_header("Range")))
        data = self.files[request.full_url]
        header = request.get_header("Range")
        if header and self.ranges:
            start = int(header.removeprefix("bytes=").removesuffix("-"))
            return Response(data[start:], 206)
        drop, self.drop_after = self.drop_after, None
        return Response(data, 200, drop)


BLOB = bytes(range(256)) * 40


def test_fetch_resumes_a_partial_download_with_a_range_request(scripts, tmp_path):
    download = scripts("_download")
    (tmp_path / "blob.bin.part").write_bytes(BLOB[:1000])
    server = Server({"https://x/blob": BLOB})
    sha = hashlib.sha256(BLOB).hexdigest()
    download.fetch("https://x/blob", tmp_path / "blob.bin", len(BLOB), sha, opener=server)
    assert (tmp_path / "blob.bin").read_bytes() == BLOB
    assert not (tmp_path / "blob.bin.part").exists()
    assert server.requests == [("https://x/blob", "bytes=1000-")]
    download.fetch("https://x/blob", tmp_path / "blob.bin", len(BLOB), sha, opener=server)
    assert len(server.requests) == 1


def test_fetch_starts_over_when_the_server_ignores_the_range(scripts, tmp_path):
    download = scripts("_download")
    (tmp_path / "blob.bin.part").write_bytes(b"stale bytes")
    server = Server({"https://x/blob": BLOB}, ranges=False)
    download.fetch("https://x/blob", tmp_path / "blob.bin", len(BLOB), opener=server)
    assert (tmp_path / "blob.bin").read_bytes() == BLOB


def test_fetch_resumes_after_a_dropped_connection(scripts, tmp_path):
    download = scripts("_download")
    server = Server({"https://x/blob": BLOB}, drop_after=3000)
    download.fetch("https://x/blob", tmp_path / "blob.bin", len(BLOB), opener=server)
    assert (tmp_path / "blob.bin").read_bytes() == BLOB
    assert server.requests[-1] == ("https://x/blob", "bytes=3000-")


def test_fetch_discards_a_download_with_the_wrong_checksum(scripts, tmp_path):
    download = scripts("_download")
    server = Server({"https://x/blob": BLOB})
    with pytest.raises(download.DownloadError, match="SHA-256"):
        download.fetch("https://x/blob", tmp_path / "blob.bin", len(BLOB), "0" * 64, server)
    assert not (tmp_path / "blob.bin").exists()
    assert not (tmp_path / "blob.bin.part").exists()


def _png(shade=0, mode="L"):
    buffer = io.BytesIO()
    Image.new(mode, (8, 6), color=(shade,) * len(mode)).save(buffer, "PNG")
    return buffer.getvalue()


def _nih_files(nih, monkeypatch):
    images = {f"0000000{p}_00{n}.png": p for p in range(1, 5) for n in range(2)}
    tar = io.BytesIO()
    with tarfile.open(fileobj=tar, mode="w:gz") as bundle:
        for name in images:
            data = _png(40, "RGBA" if name.startswith("00000004") else "L")
            info = tarfile.TarInfo(f"images/{name}")
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    findings = ["No Finding", "Effusion|Mass", "Hernia", "Pleural_Thickening|Effusion"]
    entry = "Image Index,Finding Labels,Follow-up #,Patient ID\n" + "".join(
        f"{name},{findings[i % 4]},{i},{p}\n" for i, (name, p) in enumerate(images.items())
    )
    files = {
        "Data_Entry_2017_v2020.csv": entry.encode(),
        "train_val_list.txt": "\n".join(n for n, p in images.items() if p != 3).encode(),
        "test_list.txt": "\n".join(n for n, p in images.items() if p == 3).encode(),
        "images_001.tar.gz": tar.getvalue(),
    }
    ids = {name: i for i, name in enumerate(files)}
    sizes = {name: (ids[name], len(data)) for name, data in files.items()}
    monkeypatch.setattr(nih, "METADATA", {k: v for k, v in sizes.items() if "images" not in k})
    monkeypatch.setattr(nih, "ARCHIVES", {"images_001.tar.gz": sizes["images_001.tar.gz"]})
    return Server({nih.BOX.format(ids[name]): data for name, data in files.items()})


def test_nih_script_writes_a_manifest_the_data_module_reads(scripts, tmp_path, monkeypatch):
    nih = scripts("nih_chestxray14")
    server = _nih_files(nih, monkeypatch)
    out = tmp_path / "nih"
    manifest = nih.main([str(out), "--delete-archives"], opener=server)
    assert not (out / "images_001.tar.gz").exists()
    requests = len(server.requests)
    nih.main([str(out), "--delete-archives"], opener=server)
    assert len(server.requests) == requests
    expect(
        manifest.read_text(),
        """\
path,patient_id,label,split
images/00000001_000.png,1,,train
images/00000001_001.png,1,Effusion|Mass,train
images/00000002_000.png,2,Hernia,train
images/00000002_001.png,2,Pleural_Thickening|Effusion,train
images/00000003_000.png,3,,test
images/00000003_001.png,3,Effusion|Mass,test
images/00000004_000.png,4,Hernia,train
images/00000004_001.png,4,Pleural_Thickening|Effusion,train
""",
    )
    data = CSVImageDataModule(
        str(manifest),
        split_column="split",
        class_names=list(nih.FINDINGS),
        label_separator="|",
        image_mode="L",
        split="kfold",
        folds=3,
        stratify=False,
    )
    data.setup()
    assert {len(v) for v in data.splits.values()} == {2, 4}
    batch = next(iter(data.loader("test")))
    assert tuple(batch["image"].shape) == (2, 1, 6, 8)
    assert batch["target"].sum(1).tolist() == [0, 2]


def test_nih_images_outside_both_lists_are_an_error(scripts):
    nih = scripts("nih_chestxray14")
    entry = {"Image Index": "a.png", "Finding Labels": "Mass", "Patient ID": "1"}
    with pytest.raises(nih.DownloadError, match="neither"):
        nih.manifest_rows([entry], {"b.png"}, set())


def _jpeg():
    buffer = io.BytesIO()
    Image.new("L", (8, 6), color=60).save(buffer, "JPEG")
    return buffer.getvalue()


def test_kermany_scripts_unpack_their_own_subset_and_share_one_archive(
    scripts, tmp_path, monkeypatch
):
    kermany = scripts("_kermany")
    names = [
        "CellData/OCT/train/CNV/CNV-81630-1.jpeg",
        "CellData/OCT/train/NORMAL/NORMAL-81630-4.jpeg",
        "CellData/OCT/test/DME/DME-9925-2.jpeg",
        "CellData/chest_xray/train/PNEUMONIA/VIRUS-5628383-0014.jpeg",
        "CellData/chest_xray/test/NORMAL/NORMAL-1914424-0001.jpeg",
        "CellData/chest_xray/test/.DS_Store",
        "CellData/code/retrain.py",
    ]
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("CellData/OCT/", "")
        for name in names:
            bundle.writestr(name, _jpeg() if name.endswith(".jpeg") else b"x")
    data = archive.getvalue()
    monkeypatch.setattr(kermany, "SIZE", len(data))
    monkeypatch.setattr(kermany, "SHA256", hashlib.sha256(data).hexdigest())
    server = Server({kermany.URL: data})
    shared = tmp_path / "ZhangLabData.zip"
    oct_manifest = kermany.main(
        "OCT",
        "OCT",
        ("CNV", "DME", "DRUSEN", "NORMAL"),
        [str(tmp_path / "oct"), "--archive", str(shared)],
        server,
    )
    cxr_manifest = kermany.main(
        "CXR",
        "chest_xray",
        ("NORMAL", "PNEUMONIA"),
        [str(tmp_path / "cxr"), "--archive", str(shared)],
        server,
    )
    assert len(server.requests) == 1
    expect(
        oct_manifest.read_text() + cxr_manifest.read_text(),
        """\
path,patient_id,label,split
test/DME/DME-9925-2.jpeg,9925,DME,test
train/CNV/CNV-81630-1.jpeg,81630,CNV,train
train/NORMAL/NORMAL-81630-4.jpeg,81630,NORMAL,train
path,patient_id,label,split
test/NORMAL/NORMAL-1914424-0001.jpeg,1914424,NORMAL,test
train/PNEUMONIA/VIRUS-5628383-0014.jpeg,5628383,PNEUMONIA,train
""",
    )
    assert sorted(p.name for p in (tmp_path / "cxr").rglob("*") if p.is_file()) == [
        ".ZhangLabData.zip.extracted",
        "NORMAL-1914424-0001.jpeg",
        "VIRUS-5628383-0014.jpeg",
        "manifest.csv",
    ]
    with oct_manifest.open() as handle:
        rows = list(csv.DictReader(handle))
    assert {r["patient_id"] for r in rows if r["split"] == "train"}.isdisjoint(
        r["patient_id"] for r in rows if r["split"] == "test"
    )


def test_kermany_files_that_break_the_naming_scheme_are_an_error(scripts):
    kermany = scripts("_kermany")
    with pytest.raises(kermany.DownloadError, match=r"unexpected file train/CNV/scan\.jpeg"):
        kermany.manifest_rows(["train/CNV/scan.jpeg"], ("CNV",))
