"""Tests for loss function registry and loss_fn parameter support."""

import pytest
import torch
from torch import nn

from autotimm.losses import (
    DiceLoss,
    FocalLoss,
    get_loss_registry,
    list_available_losses,
    register_custom_loss,
)
from autotimm.tasks import ImageClassifier, ObjectDetector, SemanticSegmentor


class TestLossRegistry:
    """Test the loss registry functionality."""

    def test_get_registry(self):
        """Test getting the global registry instance."""
        registry = get_loss_registry()
        assert registry is not None

    def test_list_all_losses(self):
        """Test listing all available losses."""
        losses = list_available_losses()
        assert isinstance(losses, list)
        assert len(losses) > 0
        assert "dice" in losses
        assert "focal" in losses
        assert "cross_entropy" in losses

    def test_list_losses_by_task(self):
        """Test listing losses by task category."""
        # Classification losses
        cls_losses = list_available_losses(task="classification")
        assert "cross_entropy" in cls_losses
        assert "bce_with_logits" in cls_losses

        # Detection losses
        det_losses = list_available_losses(task="detection")
        assert "focal" in det_losses
        assert "giou" in det_losses

        # Segmentation losses
        seg_losses = list_available_losses(task="segmentation")
        assert "dice" in seg_losses
        assert "tversky" in seg_losses

    def test_get_loss_by_name(self):
        """Test creating loss instances from registry."""
        registry = get_loss_registry()

        # Get Dice loss
        dice_loss = registry.get_loss("dice", num_classes=10)
        assert isinstance(dice_loss, DiceLoss)

        # Get Focal loss
        focal_loss = registry.get_loss("focal", alpha=0.25, gamma=2.0)
        assert isinstance(focal_loss, FocalLoss)

    def test_get_loss_invalid_name(self):
        """Test getting a non-existent loss raises error."""
        registry = get_loss_registry()
        with pytest.raises(ValueError, match="not found in registry"):
            registry.get_loss("invalid_loss_name")

    def test_has_loss(self):
        """Test checking if a loss exists."""
        registry = get_loss_registry()
        assert registry.has_loss("dice") is True
        assert registry.has_loss("focal") is True
        assert registry.has_loss("invalid_name") is False

    def test_register_custom_loss(self):
        """Test registering a custom loss function."""

        class CustomTestLoss(nn.Module):
            def __init__(self, scale: float = 1.0):
                super().__init__()
                self.scale = scale

            def forward(self, input, target):
                return torch.tensor(0.0)

        # Register the loss
        register_custom_loss("custom_test", CustomTestLoss, alias="ct")

        # Verify it's registered
        registry = get_loss_registry()
        assert registry.has_loss("custom_test")
        assert registry.has_loss("ct")  # Alias should work

        # Create instance
        loss = registry.get_loss("custom_test", scale=2.0)
        assert isinstance(loss, CustomTestLoss)
        assert loss.scale == 2.0

    def test_get_loss_info(self):
        """Test getting organized loss information."""
        registry = get_loss_registry()
        info = registry.get_loss_info()

        assert "classification" in info
        assert "detection" in info
        assert "segmentation" in info
        assert isinstance(info["classification"], list)
        assert isinstance(info["detection"], list)
        assert isinstance(info["segmentation"], list)


class TestImageClassifierWithLossFn:
    """Test ImageClassifier with loss_fn parameter."""

    def test_with_string_loss_fn(self):
        """Test creating classifier with string loss name."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            loss_fn="cross_entropy",
            compile_model=False,
        )
        assert model.criterion is not None
        assert isinstance(model.criterion, nn.CrossEntropyLoss)

    def test_with_custom_loss_instance(self):
        """Test creating classifier with custom loss instance."""

        class CustomClassLoss(nn.Module):
            def forward(self, input, target):
                return nn.functional.cross_entropy(input, target)

        custom_loss = CustomClassLoss()
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            loss_fn=custom_loss,
            compile_model=False,
        )
        assert model.criterion is custom_loss

    def test_backward_compatibility_no_loss_fn(self):
        """Test that models work without loss_fn (backward compatibility)."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            compile_model=False,
        )
        assert model.criterion is not None
        assert isinstance(model.criterion, nn.CrossEntropyLoss)

    def test_multilabel_with_bce(self):
        """Test multi-label classification with BCE loss."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=5,
            multi_label=True,
            loss_fn="bce_with_logits",
            compile_model=False,
        )
        assert isinstance(model.criterion, nn.BCEWithLogitsLoss)

    def test_forward_with_custom_loss(self):
        """Test forward pass works with custom loss."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            loss_fn="cross_entropy",
            compile_model=False,
        )

        # Forward pass
        x = torch.randn(2, 3, 224, 224)
        y = torch.randint(0, 10, (2,))

        output = model(x)
        assert output.shape == (2, 10)

        # Training step
        batch = (x, y)
        loss = model.training_step(batch, 0)
        assert loss.item() >= 0


class TestSemanticSegmentorWithLossFn:
    """Test SemanticSegmentor with loss_fn parameter."""

    def test_with_string_loss_fn(self):
        """Test creating segmentor with string loss name."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=10,
            loss_fn="dice",
            compile_model=False,
        )
        assert model.criterion is not None
        assert isinstance(model.criterion, DiceLoss)

    def test_with_custom_loss_instance(self):
        """Test creating segmentor with custom loss instance."""
        custom_loss = DiceLoss(num_classes=10)
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=10,
            loss_fn=custom_loss,
            compile_model=False,
        )
        assert model.criterion is custom_loss

    def test_backward_compatibility_loss_type(self):
        """Test that loss_type parameter still works."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=10,
            loss_type="combined",
            compile_model=False,
        )
        assert model.criterion is not None

    def test_loss_fn_overrides_loss_type(self):
        """Test that loss_fn takes priority over loss_type."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=10,
            loss_fn="dice",
            loss_type="combined",  # Should be ignored
            compile_model=False,
        )
        assert isinstance(model.criterion, DiceLoss)

    def test_focal_pixelwise_loss(self):
        """Test using focal pixelwise loss for segmentation."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=5,
            loss_fn="focal_pixelwise",
            compile_model=False,
        )
        assert model.criterion is not None

    def test_forward_with_custom_loss(self):
        """Test forward pass with custom loss."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=5,
            loss_fn="dice",
            compile_model=False,
        )

        # Forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape[0] == 2
        assert output.shape[1] == 5


class TestObjectDetectorWithLossFn:
    """Test ObjectDetector with loss_fn parameters."""

    def test_with_string_loss_functions(self):
        """Test creating detector with string loss names."""
        model = ObjectDetector(
            backbone="resnet18",
            num_classes=80,
            cls_loss_fn="focal",
            reg_loss_fn="giou",
            compile_model=False,
        )
        assert model.focal_loss is not None
        assert model.giou_loss is not None

    def test_with_custom_loss_instances(self):
        """Test creating detector with custom loss instances."""
        focal = FocalLoss(alpha=0.3, gamma=2.5)
        from autotimm.losses import GIoULoss

        giou = GIoULoss()

        model = ObjectDetector(
            backbone="resnet18",
            num_classes=80,
            cls_loss_fn=focal,
            reg_loss_fn=giou,
            compile_model=False,
        )
        assert model.focal_loss is focal
        assert model.giou_loss is giou

    def test_backward_compatibility_no_loss_fn(self):
        """Test that detector works without loss_fn parameters."""
        model = ObjectDetector(
            backbone="resnet18",
            num_classes=80,
            compile_model=False,
        )
        assert model.focal_loss is not None
        assert model.giou_loss is not None

    def test_forward_with_custom_losses(self):
        """Test forward pass with custom losses."""
        model = ObjectDetector(
            backbone="resnet18",
            num_classes=80,
            cls_loss_fn="focal",
            reg_loss_fn="giou",
            compile_model=False,
        )

        # Forward pass
        x = torch.randn(2, 3, 640, 640)
        cls_outputs, reg_outputs, _centerness_outputs = model(x)
        assert len(cls_outputs) == 5  # 5 FPN levels
        assert len(reg_outputs) == 5


class TestLossRegistryIntegration:
    """Integration tests for loss registry with training."""

    def test_loss_fn_with_training_step_classification(self):
        """Test that loss_fn works in training step for classification."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            loss_fn="cross_entropy",
            compile_model=False,
        )

        x = torch.randn(4, 3, 224, 224)
        y = torch.randint(0, 10, (4,))
        batch = (x, y)

        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_loss_fn_with_training_step_segmentation(self):
        """Test that loss_fn works in training step for segmentation."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=5,
            loss_fn="dice",
            compile_model=False,
        )

        images = torch.randn(2, 3, 256, 256)
        masks = torch.randint(0, 5, (2, 256, 256))
        batch = {"image": images, "mask": masks}

        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_multiple_models_with_different_losses(self):
        """Test creating multiple models with different losses."""
        model1 = SemanticSegmentor(
            backbone="resnet18",
            num_classes=10,
            loss_fn="dice",
            compile_model=False,
        )

        model2 = SemanticSegmentor(
            backbone="resnet18",
            num_classes=10,
            loss_fn="focal_pixelwise",
            compile_model=False,
        )

        model3 = SemanticSegmentor(
            backbone="resnet18",
            num_classes=10,
            loss_fn="combined_segmentation",
            compile_model=False,
        )

        # Verify they have different loss types
        assert type(model1.criterion).__name__ != type(model2.criterion).__name__
        assert type(model2.criterion).__name__ != type(model3.criterion).__name__
