import pytest
import torch

from medimage.registry import MODELS


@pytest.mark.parametrize(("dims", "shape"), [(2, (2, 1, 32, 32)), (3, (2, 1, 16, 16, 16))])
def test_simple_cnn(dims, shape):
    model = MODELS.build(
        "simple_cnn", in_channels=1, num_classes=3, spatial_dims=dims, width=4, depth=2
    )
    assert model(torch.randn(shape)).shape == (2, 3)


def test_timm_grayscale():
    model = MODELS.build("timm", in_channels=1, num_classes=1, model_name="resnet18")
    assert model(torch.randn(2, 1, 64, 64)).shape == (2, 1)


def test_timm_refuses_3d():
    with pytest.raises(ValueError, match="2D"):
        MODELS.build("timm", in_channels=1, num_classes=1, spatial_dims=3)


def test_tiny_fasterrcnn_predicts():
    model = MODELS.build(
        "fasterrcnn", in_channels=1, num_classes=3, backbone="tiny", min_size=64, max_size=64
    ).eval()
    with torch.no_grad():
        out = model([torch.rand(1, 64, 64)])
    assert set(out[0]) == {"boxes", "scores", "labels"}


@pytest.mark.parametrize("name", ["retinanet", "fcos", "fasterrcnn"])
def test_torchvision_detectors_build(name):
    model = MODELS.build(name, in_channels=3, num_classes=2, min_size=128, max_size=128).eval()
    with torch.no_grad():
        out = model([torch.rand(3, 128, 128)])
    assert "boxes" in out[0]


def test_pretrained_detector_needs_three_channels():
    with pytest.raises(ValueError, match="repeat_channels"):
        MODELS.build("fasterrcnn", in_channels=1, num_classes=2, weights_backbone="imagenet")


def test_monai_densenet_3d():
    pytest.importorskip("monai")
    model = MODELS.build("monai", in_channels=1, num_classes=2, spatial_dims=3, net="DenseNet121")
    with torch.no_grad():
        assert model.eval()(torch.randn(2, 1, 32, 32, 32)).shape == (2, 2)
