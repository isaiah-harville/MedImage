import pytest
from conftest import bundle_published, serve_checkpoint

from opendiagnostics.publish import WEIGHTS_URL, load_artifacts
from opendiagnostics.serve.weights import fetch

IN_CLUSTER = "http://seaweedfs-s3.seaweedfs.svc:8333/opendiagnostics-weights"


@pytest.fixture
def weights(finished_run, tmp_path):
    return bundle_published(finished_run, tmp_path)


def test_fetch_downloads_from_the_base_url_and_names_the_file_by_its_hash(
    finished_run, weights, tmp_path
):
    served: list[str] = []
    opener = serve_checkpoint(finished_run / "checkpoint.ckpt", served)
    assert fetch(weights, tmp_path / "cache", IN_CLUSTER, opener) == ["smoke-project"]
    [artifact] = [
        a
        for a in load_artifacts(weights / "smoke-project" / "artifacts.yaml").files
        if a.file == "checkpoint.ckpt"
    ]
    assert artifact.url
    assert served == [f"{IN_CLUSTER}/{artifact.url.removeprefix(WEIGHTS_URL + '/')}"]
    assert [p.name for p in (tmp_path / "cache").iterdir()] == [f"{artifact.sha256}.ckpt"]


def test_fetch_skips_a_cached_checkpoint_and_removes_unused_ones(finished_run, weights, tmp_path):
    cache = tmp_path / "cache"
    opener = serve_checkpoint(finished_run / "checkpoint.ckpt")
    fetch(weights, cache, WEIGHTS_URL, opener)
    (cache / f"{'0' * 64}.ckpt").write_bytes(b"old")
    served: list[str] = []
    again = serve_checkpoint(finished_run / "checkpoint.ckpt", served)
    assert fetch(weights, cache, WEIGHTS_URL, again) == []
    assert served == []
    assert len(list(cache.iterdir())) == 1


def test_fetch_refuses_a_checkpoint_whose_hash_does_not_match(weights, tmp_path):
    wrong = tmp_path / "wrong.ckpt"
    wrong.write_bytes(b"not the checkpoint")
    with pytest.raises(ValueError, match=r"artifacts\.yaml expects"):
        fetch(weights, tmp_path / "cache", WEIGHTS_URL, serve_checkpoint(wrong))
    assert list((tmp_path / "cache").iterdir()) == []
