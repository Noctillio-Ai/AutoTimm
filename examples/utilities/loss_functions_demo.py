"""Demonstration of loss function registry and custom loss usage in AutoTimm.

This example shows:
1. How to list available loss functions
2. How to use built-in losses from the registry
3. How to create and register custom loss functions
4. How to pass losses to different model types
"""

import torch
from torch import nn

from autotimm import (
    ImageClassifier,
    ObjectDetector,
    SemanticSegmentor,
)
from autotimm.losses import (
    get_loss_registry,
    list_available_losses,
    register_custom_loss,
)


def example_1_list_available_losses():
    """Example 1: List all available loss functions."""
    print("\n" + "=" * 70)
    print("Example 1: Available Loss Functions")
    print("=" * 70)

    # Get the global registry
    registry = get_loss_registry()

    # List all losses
    print("\nAll available losses:")
    print(list_available_losses())

    # List by task
    print("\nClassification losses:")
    print(list_available_losses(task="classification"))

    print("\nDetection losses:")
    print(list_available_losses(task="detection"))

    print("\nSegmentation losses:")
    print(list_available_losses(task="segmentation"))

    # Get loss info organized by task
    print("\nLoss info by task:")
    info = registry.get_loss_info()
    for task, losses in info.items():
        print(f"  {task}: {losses}")


def example_2_use_registry_losses_for_classification():
    """Example 2: Use losses from registry for classification."""
    print("\n" + "=" * 70)
    print("Example 2: Using Registry Losses for Classification")
    print("=" * 70)

    # Option 1: Use loss name directly in model
    print("\n--- Using string loss name ---")
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        loss_fn="cross_entropy",  # Use from registry
        compile_model=False,
    )
    print("✓ Created ImageClassifier with 'cross_entropy' loss")
    print(f"  Loss type: {type(model.criterion)}")

    # Option 2: Create loss from registry and pass to model
    print("\n--- Creating loss from registry ---")
    registry = get_loss_registry()
    bce_loss = registry.get_loss("bce_with_logits")
    print("✓ Created BCEWithLogitsLoss from registry")
    print(f"  Loss type: {type(bce_loss)}")

    model_multilabel = ImageClassifier(
        backbone="resnet18",
        num_classes=5,
        multi_label=True,
        loss_fn=bce_loss,  # Pass instance
        compile_model=False,
    )
    print("✓ Created multi-label classifier with BCEWithLogitsLoss")


def example_3_use_registry_losses_for_segmentation():
    """Example 3: Use losses from registry for segmentation."""
    print("\n" + "=" * 70)
    print("Example 3: Using Registry Losses for Segmentation")
    print("=" * 70)

    # Use different segmentation losses
    losses_to_try = ["dice", "focal_pixelwise", "combined_segmentation"]

    for loss_name in losses_to_try:
        print(f"\n--- Using '{loss_name}' loss ---")
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=19,
            head_type="fcn",
            loss_fn=loss_name,  # Use from registry
            compile_model=False,
        )
        print(f"✓ Created SemanticSegmentor with '{loss_name}' loss")
        print(f"  Loss type: {type(model.criterion)}")


def example_4_use_registry_losses_for_detection():
    """Example 4: Use losses from registry for detection."""
    print("\n" + "=" * 70)
    print("Example 4: Using Registry Losses for Detection")
    print("=" * 70)

    # Use detection losses from registry
    print("\n--- Using losses from registry ---")
    model = ObjectDetector(
        backbone="resnet18",
        num_classes=80,
        cls_loss_fn="focal",  # Classification loss
        reg_loss_fn="giou",  # Regression loss
        compile_model=False,
    )
    print("✓ Created ObjectDetector with registry losses")
    print(f"  Classification loss: {type(model.focal_loss)}")
    print(f"  Regression loss: {type(model.giou_loss)}")


def example_5_custom_loss_function():
    """Example 5: Create and register a custom loss function."""
    print("\n" + "=" * 70)
    print("Example 5: Custom Loss Functions")
    print("=" * 70)

    # Define a custom loss
    class CustomFocalTverskyLoss(nn.Module):
        """Custom loss combining Focal and Tversky concepts."""

        def __init__(
            self,
            num_classes: int,
            alpha: float = 0.7,
            beta: float = 0.3,
            gamma: float = 0.75,
            ignore_index: int = 255,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.ignore_index = ignore_index
            print("  Initialized CustomFocalTverskyLoss with:")
            print(f"    alpha={alpha}, beta={beta}, gamma={gamma}")

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            # Simplified implementation for demo
            # In practice, you would implement the full loss logic
            return torch.tensor(0.0, requires_grad=True)

    # Register the custom loss
    print("\n--- Registering custom loss ---")
    register_custom_loss(
        name="focal_tversky",
        loss_class=CustomFocalTverskyLoss,
        alias="ft",
    )
    print("✓ Registered 'focal_tversky' loss with alias 'ft'")

    # Verify it's registered
    registry = get_loss_registry()
    print(f"✓ Loss is registered: {registry.has_loss('focal_tversky')}")
    print(f"✓ Alias works: {registry.has_loss('ft')}")

    # Use the custom loss
    print("\n--- Using custom loss ---")
    custom_loss = registry.get_loss("focal_tversky", num_classes=10)
    print(f"✓ Created custom loss instance: {type(custom_loss)}")

    # Use in a model
    model = SemanticSegmentor(
        backbone="resnet18",
        num_classes=10,
        loss_fn=custom_loss,
        compile_model=False,
    )
    print("✓ Created SemanticSegmentor with custom loss")


def example_6_backward_compatibility():
    """Example 6: Backward compatibility with loss_type parameter."""
    print("\n" + "=" * 70)
    print("Example 6: Backward Compatibility")
    print("=" * 70)

    # Old way (still works)
    print("\n--- Old way: using loss_type ---")
    model_old = SemanticSegmentor(
        backbone="resnet18",
        num_classes=10,
        loss_type="combined",  # Old parameter
        dice_weight=1.5,
        ce_weight=1.0,
        compile_model=False,
    )
    print("✓ Old way still works: loss_type='combined'")
    print(f"  Loss type: {type(model_old.criterion)}")

    # New way (preferred)
    print("\n--- New way: using loss_fn ---")
    model_new = SemanticSegmentor(
        backbone="resnet18",
        num_classes=10,
        loss_fn="combined_segmentation",  # New parameter
        dice_weight=1.5,
        ce_weight=1.0,
        compile_model=False,
    )
    print("✓ New way: loss_fn='combined_segmentation'")
    print(f"  Loss type: {type(model_new.criterion)}")


def example_7_training_with_custom_loss():
    """Example 7: Complete training workflow with custom loss."""
    print("\n" + "=" * 70)
    print("Example 7: Training Workflow with Custom Loss")
    print("=" * 70)

    # Define a simple custom loss
    class WeightedCrossEntropyLoss(nn.Module):
        """Cross-entropy with custom class weighting."""

        def __init__(self, class_weights: torch.Tensor, ignore_index: int = 255):
            super().__init__()
            self.loss_fn = nn.CrossEntropyLoss(
                weight=class_weights, ignore_index=ignore_index
            )

        def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return self.loss_fn(input, target)

    # Create class weights for imbalanced data
    class_weights = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.5])
    custom_loss = WeightedCrossEntropyLoss(class_weights=class_weights)

    print("\n--- Model with custom weighted loss ---")
    model = SemanticSegmentor(
        backbone="resnet18",
        num_classes=5,
        loss_fn=custom_loss,
        compile_model=False,
    )
    print("✓ Created model with custom weighted CrossEntropy")
    print(f"  Class weights: {class_weights.tolist()}")

    # This model is now ready for training with AutoTrainer
    print("\n--- Ready for training ---")
    print("  trainer = AutoTrainer(max_epochs=10)")
    print("  trainer.fit(model, datamodule=data)")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("AutoTimm Loss Functions Demonstration")
    print("=" * 70)

    example_1_list_available_losses()
    example_2_use_registry_losses_for_classification()
    example_3_use_registry_losses_for_segmentation()
    example_4_use_registry_losses_for_detection()
    example_5_custom_loss_function()
    example_6_backward_compatibility()
    example_7_training_with_custom_loss()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Use list_available_losses() to see all available losses")
    print("  2. Pass loss_fn='loss_name' to use losses from registry")
    print("  3. Pass loss_fn=custom_loss_instance for custom losses")
    print("  4. Register custom losses with register_custom_loss()")
    print("  5. Old loss_type parameter still works for backward compatibility")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
