import json
import urllib.error

import pytest
from conftest import MemoryBucket, smoke_project

from opendiagnostics.publish import WEIGHTS_URL
from opendiagnostics.serve.weights import COMPLETE, prune, sync

IN_CLUSTER = "http://seaweedfs-s3.seaweedfs.svc:8333/opendiagnostics-weights"


@pytest.fixture
def bucket(finished_run, tmp_path):
    bucket = MemoryBucket()
    smoke_project(finished_run, tmp_path / "models", bucket=bucket)
    bucket.served.clear()
    return bucket


def _rewrite(bucket):
    def open_url(url):
        return bucket.open_url(url.replace(IN_CLUSTER, WEIGHTS_URL))

    return open_url


def test_sync_downloads_each_cataloged_run_and_names_its_checkpoint_by_hash(
    finished_run, bucket, tmp_path
):
    seen = []
    opener = _rewrite(bucket)
    [folder] = sync(tmp_path / "cache", IN_CLUSTER, lambda url: seen.append(url) or opener(url))
    run = finished_run.name
    assert folder == tmp_path / "cache" / "models" / "smoke-project" / run
    assert seen == [
        f"{IN_CLUSTER}/catalog.json",
        f"{IN_CLUSTER}/smoke-project/{run}/demo.tar",
        f"{IN_CLUSTER}/smoke-project/{run}/checkpoint.ckpt",
    ]
    assert {"config.yaml", "metrics.json", "artifacts.yaml", "examples", COMPLETE} <= {
        p.name for p in folder.iterdir()
    }
    [checkpoint] = (tmp_path / "cache" / "checkpoints").iterdir()
    assert checkpoint.read_bytes() == (finished_run / "checkpoint.ckpt").read_bytes()
    assert len(checkpoint.stem) == 64


def test_sync_again_reads_only_the_catalog(bucket, tmp_path):
    sync(tmp_path / "cache", WEIGHTS_URL, bucket.open_url)
    bucket.served.clear()
    assert len(sync(tmp_path / "cache", WEIGHTS_URL, bucket.open_url)) == 1
    assert bucket.served == [f"{WEIGHTS_URL}/catalog.json"]


def test_prune_removes_runs_and_checkpoints_the_catalog_dropped(bucket, tmp_path):
    cache = tmp_path / "cache"
    folders = sync(cache, WEIGHTS_URL, bucket.open_url)
    (cache / "models" / "old-project" / "r1").mkdir(parents=True)
    (cache / "checkpoints" / f"{'0' * 64}.ckpt").write_bytes(b"old")
    prune(cache, folders)
    assert [p.name for p in (cache / "models").iterdir()] == ["smoke-project"]
    assert len(list((cache / "checkpoints").iterdir())) == 1
    prune(cache, [])
    assert list((cache / "models").iterdir()) == []
    assert list((cache / "checkpoints").iterdir()) == []


def test_a_checkpoint_whose_hash_does_not_match_is_skipped(finished_run, bucket, tmp_path):
    key = f"smoke-project/{finished_run.name}/checkpoint.ckpt"
    bucket.objects[key] = b"not the checkpoint"
    assert sync(tmp_path / "cache", WEIGHTS_URL, bucket.open_url) == []
    assert list((tmp_path / "cache" / "checkpoints").iterdir()) == []


def test_an_empty_bucket_serves_no_models(tmp_path):
    assert sync(tmp_path / "cache", WEIGHTS_URL, MemoryBucket().open_url) == []


def test_a_catalog_naming_an_unsafe_folder_is_refused(bucket, tmp_path):
    bucket.objects["catalog.json"] = json.dumps({"models": {"../escape": "r1"}}).encode()
    with pytest.raises(ValueError, match="not a safe folder name"):
        sync(tmp_path / "cache", WEIGHTS_URL, bucket.open_url)


def test_sync_waits_and_retries_when_rate_limited(bucket, tmp_path):
    waits, limited = [], []

    def open_url(url):
        if not limited:
            limited.append(url)
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)  # ty: ignore[invalid-argument-type]
        return bucket.open_url(url)

    assert len(sync(tmp_path / "cache", WEIGHTS_URL, open_url, sleep=waits.append)) == 1
    assert waits == [10.0]


def test_a_rescored_run_is_downloaded_again_when_its_bundle_changes(finished_run, bucket, tmp_path):
    cache = tmp_path / "cache"
    [folder] = sync(cache, WEIGHTS_URL, bucket.open_url)
    catalog = json.loads(bucket.objects["catalog.json"])
    assert (folder / COMPLETE).read_text() == catalog["bundles"]["smoke-project"]
    catalog["bundles"]["smoke-project"] = "0" * 64
    bucket.objects["catalog.json"] = json.dumps(catalog).encode()
    bucket.served.clear()
    sync(cache, WEIGHTS_URL, bucket.open_url)
    assert f"{WEIGHTS_URL}/smoke-project/{finished_run.name}/demo.tar" in bucket.served
    assert (folder / COMPLETE).read_text() == "0" * 64
