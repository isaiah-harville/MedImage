import hashlib
import io
import json

import pytest
import torch
from torch import nn

from opendiagnostics.data import Camelyon16DataModule, SyntheticSlideDataModule
from opendiagnostics.data import camelyon_download as bucket
from opendiagnostics.data.synthetic import write_pyramid


class Body(io.BytesIO):
    def iter_chunks(self, size):
        while block := self.read(size):
            yield block


class FakeBucket:
    """The bucket's list, head and ranged-get calls, served from memory."""

    def __init__(self, objects):
        self.objects = objects
        self.ranges = []

    def get_paginator(self, name):
        objects = self.objects

        class Pages:
            def paginate(self, Bucket, Prefix):
                return [
                    {
                        "Contents": [
                            {"Key": k, "Size": len(v)}
                            for k, v in objects.items()
                            if k.startswith(Prefix)
                        ]
                    }
                ]

        return Pages()

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.objects[Key])}

    def get_object(self, Bucket, Key, Range):
        start = int(Range.removeprefix("bytes=").rstrip("-"))
        self.ranges.append((Key, start))
        return {"Body": Body(self.objects[Key][start:])}


def _bucket(slide=b"slide-bytes" * 100, recorded=None, extra=None):
    digest = recorded or hashlib.md5(slide).hexdigest()
    return FakeBucket(
        {
            "CAMELYON16/README.md": b"readme",
            "CAMELYON16/license.txt": b"cc0",
            "CAMELYON16/checksums.md5": f"{digest} *images/normal_001.tif\n".encode(),
            "CAMELYON16/images/normal_001.tif": slide,
            "CAMELYON16/masks/normal_001.tif": b"mask",
        }
        | (extra or {})
    )


def test_download_resumes_a_partial_file_and_records_it_verified(tmp_path, monkeypatch):
    fake = _bucket()
    monkeypatch.setattr(bucket, "client", lambda: fake)
    slide = fake.objects["CAMELYON16/images/normal_001.tif"]
    (tmp_path / "images").mkdir()
    (tmp_path / "images/normal_001.tif.partial").write_bytes(slide[:300])
    result = bucket.download("CAMELYON16", tmp_path, lambda n: n.startswith("images/"))
    assert result == {"files": 1, "bytes": len(slide)}
    assert (tmp_path / "images/normal_001.tif").read_bytes() == slide
    assert ("CAMELYON16/images/normal_001.tif", 300) in fake.ranges
    assert not (tmp_path / "masks").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["images/normal_001.tif"]["verified"] is True
    fake.ranges.clear()
    bucket.download("CAMELYON16", tmp_path, lambda n: n.startswith("images/"))
    assert fake.ranges == []


def test_a_file_with_the_wrong_md5_is_deleted_and_reported(tmp_path, monkeypatch):
    fake = _bucket(recorded="0" * 32)
    monkeypatch.setattr(bucket, "client", lambda: fake)
    with pytest.raises(RuntimeError, match="does not match"):
        bucket.download("CAMELYON16", tmp_path, lambda n: n.startswith("images/"))
    assert not (tmp_path / "images/normal_001.tif").exists()
    assert not (tmp_path / "images/normal_001.tif.partial").exists()


class MeanColour(nn.Module):
    def forward(self, images):
        return images.mean((2, 3))


def _tiny_slide(path):
    import numpy as np

    image = np.full((128, 128, 3), 240, np.uint8)
    image[16:112, 16:112] = (200, 100, 150)
    write_pyramid(path, image, levels=3)
    return path.read_bytes()


def test_camelyon16_streams_each_slide_from_the_bucket_and_deletes_it(tmp_path, monkeypatch):
    slide = _tiny_slide(tmp_path / "source.tif")
    reference = b"image,type,class,center\nnormal_001.tif,normal,negative,0\n"
    fake = _bucket(slide, extra={"CAMELYON16/evaluation/reference.csv": reference})
    monkeypatch.setattr(bucket, "client", lambda: fake)
    root = tmp_path / "camelyon16"
    data = Camelyon16DataModule(
        root=str(root),
        features_dir=str(root / "features"),
        encoder=MeanColour(),
        patch_size=16,
        mask_downsample=4,
        extract_workers=0,
        extract_device="cpu",
    )
    data.prepare_data()
    assert (root / "features/normal_001.pt").exists()
    assert not (root / "images/normal_001.tif").exists()
    assert (root / "evaluation/reference.csv").read_bytes() == reference
    fake.ranges.clear()
    data.prepare_data()
    assert fake.ranges == []


class RecordingSlides(SyntheticSlideDataModule):
    """Synthetic slides that log fetches and how many slide files are on disk at each step."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events = []

    def on_disk(self):
        return len(list((self.root / "images").glob("*.tif")))

    def fetch_slide(self, slide):
        super().fetch_slide(slide)
        self.events.append(("fetched", slide.slide_id, self.on_disk()))


def _recording(tmp_path, **kwargs):
    return RecordingSlides(
        root=str(tmp_path),
        features_dir=str(tmp_path / "features"),
        encoder=MeanColour(),
        num_slides=5,
        slide_size=128,
        patch_size=16,
        mask_downsample=4,
        extract_workers=0,
        extract_device="cpu",
        **kwargs,
    )


@pytest.mark.parametrize("prefetch", [0, 1, 2])
def test_streaming_keeps_at_most_one_plus_prefetch_slides_on_disk(tmp_path, prefetch):
    data = _recording(tmp_path, prefetch=prefetch)
    data.prepare_data()
    assert [e[1] for e in data.events] == [f"slide_{n:03d}" for n in range(5)]
    assert max(e[2] for e in data.events) <= 1 + prefetch
    assert data.on_disk() == 0
    assert len(list((tmp_path / "features").glob("*.pt"))) == 5


def test_slides_with_features_are_not_fetched_again_and_kept_slides_stay(tmp_path):
    data = _recording(tmp_path, keep_slides=True)
    data.prepare_data()
    assert data.on_disk() == 5
    (tmp_path / "features/slide_002.pt").unlink()
    (tmp_path / "images/slide_002.tif").unlink()
    first = torch.load(tmp_path / "features/slide_000.pt", weights_only=True)
    data.events.clear()
    data.prepare_data()
    assert [e[1] for e in data.events] == ["slide_002"]
    assert torch.equal(
        torch.load(tmp_path / "features/slide_000.pt", weights_only=True)["features"],
        first["features"],
    )
