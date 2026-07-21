"""Tests for official YOLOX detector."""

import pytest
import torch

from autotimm import YOLOXDetector
from autotimm.models import build_csp_darknet, build_yolox_pafpn


def test_csp_darknet_forward():
    """Test CSPDarknet backbone forward pass."""
    backbone = build_csp_darknet("yolox-s")
    x = torch.randn(2, 3, 640, 640)
    features = backbone(x)

    # Should return 3 features: dark3, dark4, dark5
    assert len(features) == 3
    assert "dark3" in features
    assert "dark4" in features
    assert "dark5" in features

    # Check shapes (stride 8, 16, 32)
    assert features["dark3"].shape == (2, 128, 80, 80)
    assert features["dark4"].shape == (2, 256, 40, 40)
    assert features["dark5"].shape == (2, 512, 20, 20)


def test_yolox_pafpn_forward():
    """Test YOLOXPAFPN neck forward pass."""
    pafpn = build_yolox_pafpn("yolox-s")

    # Create dummy features
    features = {
        "dark3": torch.randn(2, 128, 80, 80),
        "dark4": torch.randn(2, 256, 40, 40),
        "dark5": torch.randn(2, 512, 20, 20),
    }

    outputs = pafpn(features)

    # Should return 3 outputs: P3, N3, N4 with uniform channels
    assert len(outputs) == 3
    assert outputs[0].shape == (2, 128, 80, 80)  # P3
    assert outputs[1].shape == (2, 128, 40, 40)  # N3 (projected to uniform channels)
    assert outputs[2].shape == (2, 128, 20, 20)  # N4 (projected to uniform channels)


@pytest.mark.parametrize(
    "model_name",
    ["yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x"],
)
def test_yolox_detector_init(model_name):
    """Test YOLOXDetector initialization for all variants."""
    model = YOLOXDetector(model_name=model_name, num_classes=80, metrics=None)

    # Check components exist
    assert hasattr(model, "backbone")
    assert hasattr(model, "neck")
    assert hasattr(model, "head")


def test_yolox_detector_forward():
    """Test YOLOXDetector forward pass."""
    model = YOLOXDetector(model_name="yolox-s", num_classes=80, metrics=None)
    model.eval()

    x = torch.randn(2, 3, 640, 640)
    cls_outputs, reg_outputs = model(x)

    # Should have 3 feature levels
    assert len(cls_outputs) == 3
    assert len(reg_outputs) == 3

    # Check shapes
    # P3: stride 8
    assert cls_outputs[0].shape == (2, 80, 80, 80)
    assert reg_outputs[0].shape == (2, 4, 80, 80)

    # P4: stride 16
    assert cls_outputs[1].shape == (2, 80, 40, 40)
    assert reg_outputs[1].shape == (2, 4, 40, 40)

    # P5: stride 32
    assert cls_outputs[2].shape == (2, 80, 20, 20)
    assert reg_outputs[2].shape == (2, 4, 20, 20)


def test_yolox_detector_training_step():
    """Test YOLOXDetector training step."""
    model = YOLOXDetector(model_name="yolox-s", num_classes=80, metrics=None)

    # Create dummy batch
    batch = {
        "image": torch.randn(2, 3, 640, 640),
        "boxes": [
            torch.tensor([[10, 10, 50, 50], [100, 100, 200, 200]], dtype=torch.float32),
            torch.tensor([[20, 20, 60, 60]], dtype=torch.float32),
        ],
        "labels": [
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([2], dtype=torch.int64),
        ],
    }

    # Training step
    loss = model.training_step(batch, batch_idx=0)

    # Check loss is valid (can be negative due to GIoU loss)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert not torch.isnan(loss)  # Just check it's not NaN
    assert not torch.isinf(loss)  # And not infinite


def test_yolox_detector_validation_step():
    """Test YOLOXDetector validation step."""
    model = YOLOXDetector(model_name="yolox-s", num_classes=80, metrics=None)

    # Create dummy batch
    batch = {
        "image": torch.randn(2, 3, 640, 640),
        "boxes": [
            torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
            torch.tensor([[20, 20, 60, 60]], dtype=torch.float32),
        ],
        "labels": [
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int64),
        ],
    }

    # Validation step
    model.validation_step(batch, batch_idx=0)

    # Just verify it runs without errors


def test_yolox_detector_predict():
    """Test YOLOXDetector inference."""
    model = YOLOXDetector(model_name="yolox-s", num_classes=80, metrics=None)
    model.eval()

    images = torch.randn(2, 3, 640, 640)

    with torch.no_grad():
        predictions = model.predict(images)

    # Should return predictions for each image
    assert len(predictions) == 2

    # Each prediction should have boxes, scores, labels
    for pred in predictions:
        assert "boxes" in pred
        assert "scores" in pred
        assert "labels" in pred
        assert isinstance(pred["boxes"], torch.Tensor)
        assert isinstance(pred["scores"], torch.Tensor)
        assert isinstance(pred["labels"], torch.Tensor)


def test_yolox_detector_output_channels():
    """Test that different YOLOX variants have correct output channels."""
    configs = {
        "yolox-nano": (64, 128, 256),
        "yolox-tiny": (96, 192, 384),
        "yolox-s": (128, 256, 512),
        "yolox-m": (192, 384, 768),
        "yolox-l": (256, 512, 1024),
        "yolox-x": (320, 640, 1280),
    }

    for model_name, expected_channels in configs.items():
        model = YOLOXDetector(model_name=model_name, num_classes=80, metrics=None)

        # Forward pass
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            features = model.backbone(x)

        # Check channel dimensions
        assert features["dark3"].shape[1] == expected_channels[0]
        assert features["dark4"].shape[1] == expected_channels[1]
        assert features["dark5"].shape[1] == expected_channels[2]


def test_yolox_detector_invalid_model():
    """Test that invalid model name raises error."""
    with pytest.raises(ValueError, match="Unknown model"):
        YOLOXDetector(model_name="invalid-model", num_classes=80)


def test_yolox_detector_configure_optimizers():
    """Test optimizer configuration with official YOLOX settings."""
    model = YOLOXDetector(
        model_name="yolox-s", num_classes=80, lr=0.01, metrics=None, total_epochs=300
    )

    optimizer_config = model.configure_optimizers()

    # Check optimizer (YOLOX official uses SGD)
    assert "optimizer" in optimizer_config
    optimizer = optimizer_config["optimizer"]
    assert optimizer.__class__.__name__ == "SGD"

    # Check momentum and nesterov
    assert optimizer.param_groups[0]["momentum"] == 0.9
    assert optimizer.param_groups[0]["nesterov"] is True

    # Check scheduler (YOLOX scheduler)
    assert "lr_scheduler" in optimizer_config
    scheduler_config = optimizer_config["lr_scheduler"]
    assert "scheduler" in scheduler_config
    scheduler = scheduler_config["scheduler"]
    assert scheduler.__class__.__name__ == "YOLOXLRScheduler"

    # Check base learning rate (scheduler starts with warmup from 0)
    assert scheduler.base_lrs[0] == 0.01
