import pytest
import torch

from medimage.config import Component
from medimage.transforms import (
    NormalizeIntensity,
    RandomFlip,
    Resize,
    ScaleIntensityRange,
    build_transforms,
)


def _sample(shape=(1, 20, 40), boxes=None):
    sample = {"image": torch.rand(shape), "id": "a", "patient_id": "p"}
    if boxes is not None:
        sample["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        sample["labels"] = torch.zeros(len(boxes), dtype=torch.int64)
    return sample


def test_resize_scales_boxes():
    out = Resize([40, 80])(_sample(boxes=[[4, 2, 8, 10]]))
    assert out["image"].shape == (1, 40, 80)
    assert out["boxes"].tolist() == [[8, 4, 16, 20]]


def test_resize_3d():
    out = Resize(8)(_sample((2, 16, 16, 16)))
    assert out["image"].shape == (2, 8, 8, 8)


@pytest.mark.parametrize(("axis", "expected"), [(-1, [32, 2, 36, 10]), (-2, [4, 10, 8, 18])])
def test_flip_moves_boxes_with_pixels(axis, expected):
    sample = _sample(boxes=[[4, 2, 8, 10]])
    sample["image"] = torch.zeros(1, 20, 40)
    sample["image"][0, 2:10, 4:8] = 1
    out = RandomFlip(axes=[axis], p=1.0)(sample)
    assert out["boxes"].tolist() == [expected]
    x1, y1, x2, y2 = (int(v) for v in expected)
    assert out["image"][0, y1:y2, x1:x2].sum() == out["image"].sum() == 32


def test_spatial_transform_refuses_3d_boxes():
    sample = _sample((1, 8, 8, 8))
    sample["boxes"] = torch.zeros(1, 6)
    with pytest.raises(NotImplementedError):
        RandomFlip(p=1.0)(sample)


def test_normalize_intensity_nonzero_keeps_background():
    image = torch.zeros(1, 10, 10)
    image[0, 2:8, 2:8] = torch.arange(36, dtype=torch.float32).reshape(6, 6)
    out = NormalizeIntensity("zscore", nonzero=True)({"image": image})["image"]
    assert out[0, 0, 0] == 0
    foreground = out[image != 0]
    assert foreground.mean() == pytest.approx(0, abs=1e-5)


def test_ct_window():
    out = ScaleIntensityRange(-1000, 400)({"image": torch.tensor([[[-2000.0, -300.0, 1000.0]]])})[
        "image"
    ]
    assert out.flatten().tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_build_from_config_components():
    compose = build_transforms(
        [Component(name="resize", params={"size": 8}), Component(name="normalize_intensity")]
    )
    assert compose(_sample())["image"].shape == (1, 8, 8)
