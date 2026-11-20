from pathlib import Path

import pytest
import torch

from ultralytics.nn.modules import DualStreamViTFusion
from ultralytics.nn.tasks import DetectionModel


@pytest.mark.parametrize("height,width", [(16, 16), (15, 17)])
def test_dual_stream_vit_fusion_shape_and_gradient(height, width):
    module = DualStreamViTFusion(c1=32, c2=32, num_heads=4, window_size=8)
    rgb = torch.randn(2, 32, height, width, requires_grad=True)
    infrared = torch.randn(2, 32, height, width, requires_grad=True)

    output = module([rgb, infrared])

    assert output.shape == rgb.shape
    output.mean().backward()
    assert rgb.grad is not None
    assert infrared.grad is not None


def test_dual_stream_vit_fusion_rejects_mismatched_shapes():
    module = DualStreamViTFusion(c1=32, num_heads=4, window_size=8)

    with pytest.raises(ValueError, match="feature shapes must match"):
        module([torch.randn(1, 32, 16, 16), torch.randn(1, 32, 8, 8)])


def test_dual_stream_vit_yolo_model_forward():
    model_yaml = Path(__file__).parents[1] / "yaml" / "dual-stream-vit-yolov8.yaml"
    model = DetectionModel(model_yaml, verbose=False).eval()

    predictions, _ = model(torch.randn(1, 6, 128, 128))

    assert model.stride.tolist() == [8.0, 16.0, 32.0]
    assert predictions.shape == (1, 9, 336)
