import gzip
import hashlib
import io
import pickle
from email.message import Message
from urllib.error import HTTPError

import h5py
import numpy as np
import pytest

from opendiagnostics.data import LeakageError, PCamDataModule, pcam
from opendiagnostics.data.pcam import download, official_name

COUNTS = {"train": 6, "val": 2, "test": 4}


def _labels(n):
    return np.arange(n) % 2


def _slides(split, n):
    return [f"camelyon16_{split}_{i // 2:03d}" for i in range(n)]


def _meta(labels, slides, index=None):
    rows = [",coord_y,coord_x,tumor_patch,center_tumor_patch,wsi"]
    index = range(len(labels)) if index is None else index
    for i, label, slide in zip(index, labels, slides, strict=True):
        rows.append(f"{i},0,0,{bool(label)},{bool(label)},{slide}")
    return "\n".join(rows) + "\n"


def _fake_record(root, slides=_slides):
    """The nine files of the Zenodo record, in its layout, with a few tiny patches."""
    for split, n in COUNTS.items():
        images = np.zeros((n, 96, 96, 3), np.uint8)
        images[:, :, :, 0] = np.arange(n).reshape(-1, 1, 1) * 10
        labels = _labels(n)
        for key, array in (("x", images), ("y", labels.astype(np.uint8).reshape(n, 1, 1, 1))):
            buffer = io.BytesIO()
            with h5py.File(buffer, "w") as handle:
                handle.create_dataset(key, data=array)
            (root / official_name(split, key)).write_bytes(gzip.compress(buffer.getvalue()))
        (root / official_name(split, "meta")).write_text(_meta(labels, slides(split, n)))


def _module(root):
    return PCamDataModule(root=str(root), download=False, eval_batch_size=4)


def test_pcam_serves_the_official_split_with_slides_as_patients(tmp_path):
    _fake_record(tmp_path)
    data = _module(tmp_path)
    data.setup()
    assert {k: len(v) for k, v in data.splits.items()} == COUNTS
    assert data.info.label_mode == "binary"
    assert data.info.class_names == ("normal", "tumor")
    assert data.info.license == "CC0 1.0"
    assert data.info.patient_ids_available
    batch = next(iter(data.loader("test")))
    assert batch["image"].shape == (4, 3, 96, 96)
    assert batch["image"][:, 0, 0, 0].tolist() == pytest.approx([0, 10 / 255, 20 / 255, 30 / 255])
    assert batch["target"].tolist() == [0, 1, 0, 1]
    assert batch["id"] == ["test-0", "test-1", "test-2", "test-3"]
    assert batch["patient_id"] == ["camelyon16_test_000"] * 2 + ["camelyon16_test_001"] * 2


def test_pcam_refuses_a_slide_that_appears_in_two_splits(tmp_path):
    def shared(split, n):
        return ["camelyon16_train_tumor_001", *_slides(split, n - 1)]

    _fake_record(tmp_path, slides=shared)
    with pytest.raises(LeakageError, match="camelyon16_train_tumor_001"):
        _module(tmp_path).setup()


def test_pcam_refuses_metadata_out_of_order_with_the_labels(tmp_path):
    _fake_record(tmp_path)
    n = COUNTS["val"]
    reordered = _meta(_labels(n)[::-1], _slides("val", n), index=[1, 0])
    (tmp_path / official_name("val", "meta")).write_text(reordered)
    with pytest.raises(ValueError, match=r"row 0 of .* does not match patch 0"):
        _module(tmp_path).setup()


def test_pcam_refuses_metadata_whose_labels_disagree(tmp_path):
    _fake_record(tmp_path)
    n = COUNTS["test"]
    flipped = _meta(1 - _labels(n), _slides("test", n))
    (tmp_path / official_name("test", "meta")).write_text(flipped)
    with pytest.raises(ValueError, match="slide IDs cannot be trusted"):
        _module(tmp_path).setup()


def test_pcam_refuses_metadata_with_missing_rows(tmp_path):
    _fake_record(tmp_path)
    short = _meta(_labels(3), _slides("test", 3))
    (tmp_path / official_name("test", "meta")).write_text(short)
    with pytest.raises(ValueError, match="has 3 rows for 4 patches"):
        _module(tmp_path).setup()


def test_pcam_unpacks_to_npy_and_skips_what_is_already_unpacked(tmp_path):
    _fake_record(tmp_path)
    data = _module(tmp_path)
    data.prepare_data()
    assert np.load(tmp_path / "train_images.npy", mmap_mode="r").shape == (6, 96, 96, 3)
    assert np.load(tmp_path / "train_labels.npy").tolist() == [0, 1, 0, 1, 0, 1]
    assert not list(tmp_path.glob("*.partial")) + list(tmp_path.glob("*.h5"))
    labels = tmp_path / "val_labels.npy"
    before = labels.stat().st_mtime_ns
    (tmp_path / "test_images.npy").unlink()
    data.prepare_data()
    assert (tmp_path / "test_images.npy").exists()
    assert labels.stat().st_mtime_ns == before
    for archive in tmp_path.glob("*.h5.gz"):
        archive.unlink()
    data.setup()
    assert len(data.splits["train"]) == COUNTS["train"]


def test_pcam_dataset_pickles_without_its_images(tmp_path):
    _fake_record(tmp_path)
    data = _module(tmp_path)
    data.setup()
    data.source.dataset[0]
    assert len(pickle.dumps(data.source.dataset)) < 10_000


def test_pcam_without_files_or_download_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="set download: true"):
        _module(tmp_path).setup()


class _Response(io.BytesIO):
    def __init__(self, body: bytes, status: int, content_range: str = "") -> None:
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Range": content_range} if content_range else {}


def _server(content: bytes, honour_range: bool = True):
    requests = []

    def opener(request, timeout):
        requests.append(request)
        header = request.get_header("Range")
        if header and honour_range:
            start = int(header.removeprefix("bytes=").rstrip("-"))
            if start >= len(content):
                raise HTTPError(request.full_url, 416, "Range Not Satisfiable", Message(), None)
            return _Response(
                content[start:], 206, f"bytes {start}-{len(content) - 1}/{len(content)}"
            )
        return _Response(content, 200)

    return opener, requests


CONTENT = bytes(range(256)) * 40
MD5 = hashlib.md5(CONTENT).hexdigest()


def test_download_resumes_from_the_partial_file(tmp_path):
    dest = tmp_path / "file.h5.gz"
    (tmp_path / "file.h5.gz.partial").write_bytes(CONTENT[:1000])
    opener, requests = _server(CONTENT)
    download("https://example.test/file", dest, MD5, opener)
    assert requests[0].get_header("Range") == "bytes=1000-"
    assert dest.read_bytes() == CONTENT
    assert not (tmp_path / "file.h5.gz.partial").exists()


def test_download_starts_over_when_the_server_ignores_the_range(tmp_path):
    dest = tmp_path / "file.h5.gz"
    (tmp_path / "file.h5.gz.partial").write_bytes(b"stale bytes")
    opener, _ = _server(CONTENT, honour_range=False)
    download("https://example.test/file", dest, MD5, opener)
    assert dest.read_bytes() == CONTENT


def test_download_of_an_already_complete_partial_only_checks_it(tmp_path):
    dest = tmp_path / "file.h5.gz"
    (tmp_path / "file.h5.gz.partial").write_bytes(CONTENT)
    opener, _ = _server(CONTENT)
    download("https://example.test/file", dest, MD5, opener)
    assert dest.read_bytes() == CONTENT


def test_download_with_the_wrong_md5_deletes_the_file(tmp_path):
    dest = tmp_path / "file.h5.gz"
    opener, _ = _server(CONTENT)
    with pytest.raises(RuntimeError, match="does not match MD5"):
        download("https://example.test/file", dest, "0" * 32, opener)
    assert not dest.exists()
    assert not (tmp_path / "file.h5.gz.partial").exists()


def test_download_refuses_a_resume_that_starts_at_the_wrong_byte(tmp_path):
    (tmp_path / "file.h5.gz.partial").write_bytes(CONTENT[:1000])

    def opener(request, timeout):
        return _Response(CONTENT, 206, f"bytes 0-{len(CONTENT) - 1}/{len(CONTENT)}")

    with pytest.raises(RuntimeError, match="answered a resume from byte 1000"):
        download("https://example.test/file", tmp_path / "file.h5.gz", MD5, opener)


def test_pcam_downloads_each_missing_file_from_zenodo_with_its_md5(tmp_path, monkeypatch):
    source = tmp_path / "record"
    source.mkdir()
    _fake_record(source)
    fetched = []

    def fake_download(url, dest, md5):
        fetched.append((url, md5))
        dest.write_bytes((source / dest.name).read_bytes())

    monkeypatch.setattr(pcam, "download", fake_download)
    root = tmp_path / "data"
    PCamDataModule(root=str(root)).setup()
    assert sorted(fetched) == sorted((pcam.RECORD + name, md5) for name, md5 in pcam.MD5.items())
    assert len(fetched) == 9
    assert fetched[0][0] == (
        "https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_train_meta.csv"
    )
