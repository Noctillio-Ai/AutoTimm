"""Example: Using custom loss functions with AutoTimm models.

This example demonstrates:
1. Using built-in losses from the registry
2. Creating and using custom loss functions
3. Training with custom losses
"""

import torch
from torch import nn

from autotimm import (
    ImageClassifier,
    SemanticSegmentor,
)
from autotimm.losses import get_loss_registry, list_available_losses


def example_1_list_and_use_built_in_losses():
    """Example 1: Browse and use built-in losses."""
    print("\n" + "=" * 70)
    print("Example 1: Built-in Losses")
    print("=" * 70)

    # List all available losses
    print("\nAvailable losses:")
    print(list_available_losses())

    # List by task
    print("\nSegmentation losses:")
    print(list_available_losses(task="segmentation"))

    # Use a loss by name
    print("\n--- Using Dice loss for segmentation ---")
    model = SemanticSegmentor(
        backbone="resnet18",
        num_classes=19,
        loss_fn="dice",  # Use dice loss from registry
        compile_model=False,
    )
    print(f"✓ Created model with Dice loss: {type(model.criterion)}")


def example_2_custom_weighted_loss():
    """Example 2: Custom weighted cross-entropy for class imbalance."""
    print("\n" + "=" * 70)
    print("Example 2: Custom Weighted Loss for Imbalanced Data")
    print("=" * 70)

    class WeightedCrossEntropyLoss(nn.Module):
        """Cross-entropy with custom class weights."""

        def __init__(self, class_weights: torch.Tensor):
            super().__init__()
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return self.loss_fn(input, target)

    # Define class weights (higher weight for rare classes)
    class_weights = torch.tensor([1.0, 2.0, 3.0, 2.5, 1.5])

    # Create custom loss
    custom_loss = WeightedCrossEntropyLoss(class_weights=class_weights)

    # Use in model
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=5,
        loss_fn=custom_loss,  # Pass custom loss instance
        compile_model=False,
    )

    print("✓ Created model with custom weighted loss")
    print(f"  Class weights: {class_weights.tolist()}")
    print(f"  Loss type: {type(model.criterion)}")


def example_3_focal_tversky_loss():
    """Example 3: Custom Focal-Tversky loss for medical imaging."""
    print("\n" + "=" * 70)
    print("Example 3: Focal-Tversky Loss for Medical Imaging")
    print("=" * 70)

    class FocalTverskyLoss(nn.Module):
        """Focal Tversky loss - good for highly imbalanced segmentation."""

        def __init__(
            self,
            num_classes: int,
            alpha: float = 0.7,  # Weight for false negatives
            beta: float = 0.3,  # Weight for false positives
            gamma: float = 0.75,  # Focal parameter
            ignore_index: int = 255,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.ignore_index = ignore_index

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            import torch.nn.functional as F

            # Get probabilities
            probs = F.softmax(logits, dim=1)

            # Create valid mask
            valid_mask = targets != self.ignore_index

            # One-hot encode targets
            targets_one_hot = F.one_hot(
                targets.clamp(0, self.num_classes - 1), num_classes=self.num_classes
            )
            targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

            # Apply valid mask
            valid_mask = valid_mask.unsqueeze(1).float()
            probs = probs * valid_mask
            targets_one_hot = targets_one_hot * valid_mask

            # Compute Tversky index
            tp = (probs * targets_one_hot).sum(dim=(2, 3))
            fp = (probs * (1 - targets_one_hot)).sum(dim=(2, 3))
            fn = ((1 - probs) * targets_one_hot).sum(dim=(2, 3))

            tversky_index = tp / (tp + self.alpha * fn + self.beta * fp + 1e-7)

            # Apply focal modulation
            focal_tversky = torch.pow(1 - tversky_index, self.gamma)

            return focal_tversky.mean()

    # Create custom loss
    ft_loss = FocalTverskyLoss(
        num_classes=2,  # Binary segmentation
        alpha=0.7,  # Emphasize recall (reduce false negatives)
        beta=0.3,
        gamma=0.75,
    )

    # Use in segmentation model
    model = SemanticSegmentor(
        backbone="resnet18",
        num_classes=2,
        loss_fn=ft_loss,
        compile_model=False,
    )

    print("✓ Created model with Focal-Tversky loss")
    print(f"  Alpha (FN weight): {ft_loss.alpha}")
    print(f"  Beta (FP weight): {ft_loss.beta}")
    print(f"  Gamma (focal): {ft_loss.gamma}")


def example_4_comparing_losses():
    """Example 4: Compare different losses for the same task."""
    print("\n" + "=" * 70)
    print("Example 4: Comparing Different Losses")
    print("=" * 70)

    # Create models with different losses
    losses_to_compare = ["dice", "focal_pixelwise", "combined_segmentation"]

    for loss_name in losses_to_compare:
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=10,
            loss_fn=loss_name,
            compile_model=False,
        )
        print(f"  {loss_name:25s} -> {type(model.criterion).__name__}")

    print("\nYou can train each model and compare val/test metrics to find")
    print("the best loss function for your specific dataset!")


def example_5_loss_with_parameters():
    """Example 5: Using registry losses with custom parameters."""
    print("\n" + "=" * 70)
    print("Example 5: Registry Losses with Custom Parameters")
    print("=" * 70)

    # Get registry
    registry = get_loss_registry()

    # Create Dice loss with custom parameters
    dice_loss = registry.get_loss("dice", num_classes=19, smooth=2.0, ignore_index=255)

    # Create Combined loss with custom weights
    combined_loss = registry.get_loss(
        "combined_segmentation",
        num_classes=19,
        ce_weight=0.5,
        dice_weight=1.5,  # Emphasize Dice more
        ignore_index=255,
    )

    print(f"✓ Created custom Dice loss: {type(dice_loss)}")
    print(f"✓ Created custom Combined loss: {type(combined_loss)}")


def example_6_training_workflow():
    """Example 6: Complete training workflow with custom loss."""
    print("\n" + "=" * 70)
    print("Example 6: Training Workflow")
    print("=" * 70)

    # Define custom loss
    class BalancedCELoss(nn.Module):
        """Balanced cross-entropy for handling class imbalance."""

        def __init__(self, class_weights: torch.Tensor):
            super().__init__()
            self.ce = nn.CrossEntropyLoss(weight=class_weights)

        def forward(self, input, target):
            return self.ce(input, target)

    # Create loss
    weights = torch.tensor([1.0, 1.5, 2.0, 1.5, 1.0, 2.5])
    custom_loss = BalancedCELoss(class_weights=weights)

    # Create model
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=6,
        loss_fn=custom_loss,
        lr=1e-3,
        compile_model=False,
    )

    print("✓ Created model with custom balanced loss")
    print("\n--- Ready for training ---")
    print("# trainer = AutoTrainer(max_epochs=10)")
    print("# trainer.fit(model, datamodule=data)")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Custom Loss Functions with AutoTimm")
    print("=" * 70)

    example_1_list_and_use_built_in_losses()
    example_2_custom_weighted_loss()
    example_3_focal_tversky_loss()
    example_4_comparing_losses()
    example_5_loss_with_parameters()
    example_6_training_workflow()

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print("1. List losses: list_available_losses()")
    print("2. Use by name: loss_fn='dice'")
    print("3. Custom loss: loss_fn=MyLoss()")
    print("4. Get from registry: registry.get_loss('dice', num_classes=10)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
