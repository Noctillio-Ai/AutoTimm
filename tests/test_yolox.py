"""Tests for YOLOX detection head and ObjectDetector with YOLOX architecture."""

import pytest
import torch

from autotimm.heads import YOLOXHead
from autotimm.tasks.object_detection import ObjectDetector


def test_yolox_head_forward():
    """Test YOLOXHead forward pass."""
    head = YOLOXHead(in_channels=256, num_classes=80, num_convs=2)

    # Create dummy FPN features (5 levels)
    fpn_features = [
        torch.randn(2, 256, 80, 80),  # P3
        torch.randn(2, 256, 40, 40),  # P4
        torch.randn(2, 256, 20, 20),  # P5
        torch.randn(2, 256, 10, 10),  # P6
        torch.randn(2, 256, 5, 5),  # P7
    ]

    cls_outputs, reg_outputs = head(fpn_features)

    # Check outputs
    assert len(cls_outputs) == 5
    assert len(reg_outputs) == 5

    # Check shapes
    assert cls_outputs[0].shape == (2, 80, 80, 80)  # [B, C, H, W]
    assert reg_outputs[0].shape == (2, 4, 80, 80)  # [B, 4, H, W]

    # YOLOXHead should not return centerness


def test_yolox_head_with_depthwise():
    """Test YOLOXHead with depthwise convolutions."""
    head = YOLOXHead(
        in_channels=256,
        num_classes=80,
        num_convs=2,
        use_depthwise=True,
    )

    fpn_features = [torch.randn(1, 256, 20, 20)]
    cls_outputs, reg_outputs = head(fpn_features)

    assert cls_outputs[0].shape == (1, 80, 20, 20)
    assert reg_outputs[0].shape == (1, 4, 20, 20)


def test_object_detector_yolox():
    """Test ObjectDetector with YOLOX architecture."""
    model = ObjectDetector(
        backbone="resnet18",
        num_classes=80,
        detection_arch="yolox",
        metrics=None,
        compile_model=False,  # Disable for isinstance check
    )

    # Check that YOLOXHead was created
    assert isinstance(model.head, YOLOXHead)
    assert model.detection_arch == "yolox"

    # Forward pass
    images = torch.randn(2, 3, 640, 640)
    cls_outputs, reg_outputs, centerness_outputs = model(images)

    # YOLOXshould not return centerness
    assert centerness_outputs is None
    assert len(cls_outputs) == 5
    assert len(reg_outputs) == 5


def test_object_detector_yolox_training_step():
    """Test ObjectDetector training step with YOLOX."""
    model = ObjectDetector(
        backbone="resnet18",
        num_classes=10,
        detection_arch="yolox",
        metrics=None,
    )

    # Create dummy batch
    batch = {
        "image": torch.randn(2, 3, 640, 640),
        "boxes": [
            torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]]),
            torch.tensor([[150, 150, 250, 250]]),
        ],
        "labels": [
            torch.tensor([1, 2]),
            torch.tensor([3]),
        ],
    }

    # Training step should work
    loss = model.training_step(batch, batch_idx=0)

    assert loss.ndim == 0  # scalar
    assert loss.requires_grad
    assert loss.item() > 0


def test_object_detector_yolox_inference():
    """Test ObjectDetector inference with YOLOX."""
    model = ObjectDetector(
        backbone="resnet18",
        num_classes=10,
        detection_arch="yolox",
        metrics=None,
        score_thresh=0.05,
    )
    model.eval()

    images = torch.randn(2, 3, 640, 640)

    with torch.no_grad():
        detections = model.predict(images)

    # Should return list of detections per image
    assert len(detections) == 2
    assert "boxes" in detections[0]
    assert "scores" in detections[0]
    assert "labels" in detections[0]


def test_object_detector_fcos_vs_yolox():
    """Test that FCOS and YOLOX architectures behave differently."""
    # FCOS model
    model_fcos = ObjectDetector(
        backbone="resnet18",
        num_classes=80,
        detection_arch="fcos",
        metrics=None,
        compile_model=False,  # Disable for consistent testing
    )

    # YOLOX model
    model_yolox = ObjectDetector(
        backbone="resnet18",
        num_classes=80,
        detection_arch="yolox",
        metrics=None,
        compile_model=False,  # Disable for consistent testing
    )

    images = torch.randn(1, 3, 640, 640)

    # Forward pass
    with torch.no_grad():
        cls_f, reg_f, cent_f = model_fcos(images)
        cls_y, reg_y, cent_y = model_yolox(images)

    # FCOS should have centerness, YOLOX should not
    assert cent_f is not None
    assert cent_y is None

    # Both should have same number of outputs for cls and reg
    assert len(cls_f) == len(cls_y) == 5
    assert len(reg_f) == len(reg_y) == 5


def test_invalid_detection_arch():
    """Test that invalid detection architecture raises error."""
    with pytest.raises(ValueError, match="detection_arch must be"):
        ObjectDetector(
            backbone="resnet18",
            num_classes=80,
            detection_arch="invalid_arch",
        )
