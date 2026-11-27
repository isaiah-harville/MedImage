import base64
import gzip
import io

import nibabel as nib
import numpy as np
import pytest
from conftest import MemoryBucket, clean_run, fit_smoke, smoke_project
from fastapi import HTTPException

from opendiagnostics.publish import WEIGHTS_URL
from opendiagnostics.serve.app import ClassifierAPI, PredictRequest, describe
from opendiagnostics.serve.models import volume_input
from opendiagnostics.serve.volumes import MAX_VOLUME_BYTES, VolumeError, read_volume
from opendiagnostics.serve.weights import sync

VOLUME = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)


def _npy(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _nifti(array):
    return nib.Nifti1Image(array.transpose(), np.eye(4)).to_bytes()


@pytest.mark.parametrize(
    "encode",
    [_npy, lambda a: gzip.compress(_npy(a)), _nifti, lambda a: gzip.compress(_nifti(a))],
    ids=["npy", "npy.gz", "nii", "nii.gz"],
)
def test_a_volume_reads_back_depth_first_from_each_format(encode):
    assert np.array_equal(read_volume(encode(VOLUME)), VOLUME)


def test_a_channel_axis_of_one_is_dropped():
    assert read_volume(_npy(VOLUME[None])).shape == (2, 3, 4)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (_npy(np.zeros((4, 4))), "expected a 3D volume"),
        (_npy(np.array([np.nan] * 8).reshape(2, 2, 2)), "NaN"),
        (_npy(np.array(["a"] * 8).reshape(2, 2, 2)), "expected numbers"),
        (b"\x89PNG\r\n\x1a\n" + b"0" * 400, "upload a .npy array or a NIfTI"),
        (gzip.compress(b"\0" * (MAX_VOLUME_BYTES + 1)), "unpacks to more than"),
    ],
    ids=["2d", "nan", "text", "png", "gzip-bomb"],
)
def test_an_unreadable_volume_says_why(data, message):
    with pytest.raises(VolumeError, match=message):
        read_volume(data)


def test_volume_input_resizes_to_a_cube_and_scales_into_unit_range():
    floats = volume_input(np.linspace(-1000, 3000, 8 * 6 * 4).reshape(8, 6, 4), 16)
    assert floats.shape == (1, 16, 16, 16)
    assert floats.min() == 0
    assert floats.max() == 1
    bytes_ = volume_input(np.full((16, 16, 16), 51, np.uint8), 16)
    assert bytes_.unique().tolist() == [pytest.approx(0.2)]


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("serve3d")
    run = clean_run(fit_smoke("classification_3d", tmp / "runs"))
    bucket = MemoryBucket()
    smoke_project(run, tmp / "models", "classification_3d", "smoke-volumes", bucket=bucket)
    cache = tmp / "cache"
    folders = sync(cache, WEIGHTS_URL, bucket.open_url)
    api = ClassifierAPI(folders, cache, 2)
    api.setup("cpu")
    return describe(folders), api


def test_a_3d_classifier_is_listed_with_its_demo_volumes(served):
    listings, _ = served
    listing = listings["smoke-volumes"]
    assert listing.description["input"] == {"channels": 1, "size": 16, "dims": 3}
    assert len(listing.demos) == 8
    assert all(demo.volume is not None for demo in listing.demos.values())


def test_a_demo_volume_is_predicted_with_every_slice_explained(served):
    listings, api = served
    sample = next(iter(listings["smoke-volumes"].demos))
    request = PredictRequest(model="smoke-volumes", sample_id=sample)
    result = api.predict(api.decode_request(request))
    assert len(result["slices"]["input"]) == 16
    assert len(result["slices"]["attribution"]) == 16
    assert result["truth"] is not None
    assert sum(p["probability"] for p in result["probabilities"]) == pytest.approx(1)


def test_an_uploaded_volume_is_predicted_and_an_image_is_refused(served):
    _, api = served
    volume = base64.b64encode(
        gzip.compress(_nifti(np.random.default_rng(0).random((20, 12, 12))))
    ).decode()
    result = api.predict(api.decode_request(PredictRequest(model="smoke-volumes", image=volume)))
    assert result["truth"] is None
    assert len(result["slices"]["input"]) == 16
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 400).decode()
    with pytest.raises(HTTPException) as refused:
        api.decode_request(PredictRequest(model="smoke-volumes", image=png))
    assert refused.value.status_code == 422
