import hashlib
import importlib.util
import io
import json
import sys

import pytest


@pytest.fixture(scope="module")
def s3_script(root):
    sys.path.insert(0, str(root / "scripts/datasets"))
    spec = importlib.util.spec_from_file_location(
        "camelyon_s3", root / "scripts/datasets/camelyon_s3.py"
    )
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.path.remove(str(root / "scripts/datasets"))


class Body(io.BytesIO):
    def iter_chunks(self, size):
        while block := self.read(size):
            yield block


class FakeBucket:
    """The bucket's list and ranged-get calls, served from memory."""

    def __init__(self, objects):
        self.objects = objects
        self.ranges = []

    def get_paginator(self, name):
        bucket = self

        class Pages:
            def paginate(self, Bucket, Prefix):
                return [
                    {
                        "Contents": [
                            {"Key": k, "Size": len(v)}
                            for k, v in bucket.objects.items()
                            if k.startswith(Prefix)
                        ]
                    }
                ]

        return Pages()

    def get_object(self, Bucket, Key, Range):
        start = int(Range.removeprefix("bytes=").rstrip("-"))
        self.ranges.append((Key, start))
        return {"Body": Body(self.objects[Key][start:])}


def _bucket(slide=b"slide-bytes" * 100, recorded=None):
    digest = recorded or hashlib.md5(slide).hexdigest()
    return FakeBucket(
        {
            "CAMELYON16/README.md": b"readme",
            "CAMELYON16/license.txt": b"cc0",
            "CAMELYON16/checksums.md5": f"{digest} *images/normal_001.tif\n".encode(),
            "CAMELYON16/images/normal_001.tif": slide,
            "CAMELYON16/masks/normal_001.tif": b"mask",
        }
    )


def test_download_resumes_a_partial_file_and_records_it_verified(s3_script, tmp_path, monkeypatch):
    bucket = _bucket()
    monkeypatch.setattr(s3_script, "client", lambda: bucket)
    slide = bucket.objects["CAMELYON16/images/normal_001.tif"]
    (tmp_path / "images").mkdir()
    (tmp_path / "images/normal_001.tif.partial").write_bytes(slide[:300])
    result = s3_script.download("CAMELYON16", tmp_path, lambda n: n.startswith("images/"))
    assert result == {"files": 1, "bytes": len(slide)}
    assert (tmp_path / "images/normal_001.tif").read_bytes() == slide
    assert ("CAMELYON16/images/normal_001.tif", 300) in bucket.ranges
    assert not (tmp_path / "masks").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["images/normal_001.tif"]["verified"] is True
    bucket.ranges.clear()
    s3_script.download("CAMELYON16", tmp_path, lambda n: n.startswith("images/"))
    assert all(
        key.endswith(("README.md", "license.txt", "checksums.md5")) for key, _ in bucket.ranges
    )


def test_a_file_with_the_wrong_md5_is_deleted_and_reported(s3_script, tmp_path, monkeypatch):
    bucket = _bucket(recorded="0" * 32)
    monkeypatch.setattr(s3_script, "client", lambda: bucket)
    with pytest.raises(RuntimeError, match="does not match"):
        s3_script.download("CAMELYON16", tmp_path, lambda n: n.startswith("images/"))
    assert not (tmp_path / "images/normal_001.tif").exists()
