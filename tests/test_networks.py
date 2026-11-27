import pytest
import torch

from opendiagnostics.networks import KineticsVideoResNet


@pytest.mark.parametrize("architecture", ["r3d_18", "mc3_18", "r2plus1d_18"])
def test_a_video_resnet_classifies_one_channel_volumes(architecture):
    network = KineticsVideoResNet(architecture, num_classes=2, pretrained=False).eval()
    with torch.no_grad():
        logits = network(torch.rand(2, 1, 16, 32, 32))
    assert logits.shape == (2, 2)


def test_a_video_resnet_refuses_multichannel_volumes():
    network = KineticsVideoResNet("r3d_18", num_classes=1, pretrained=False)
    with pytest.raises(ValueError, match="one-channel"):
        network(torch.rand(1, 3, 8, 16, 16))
