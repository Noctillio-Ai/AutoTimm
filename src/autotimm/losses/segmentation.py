"""Loss functions for semantic and instance segmentation tasks."""

import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    """Dice loss for multi-class segmentation.

    Formula: 1 - (2 * |X ∩ Y|) / (|X| + |Y|)

    Args:
        num_classes: Number of segmentation classes
        smooth: Smoothing constant to avoid division by zero (default: 1.0)
        ignore_index: Index to ignore in loss computation (default: 255)
        reduction: Reduction method ('mean', 'sum', 'none') (default: 'mean')
    """

    def __init__(
        self,
        num_classes: int,
        smooth: float = 1.0,
        ignore_index: int = 255,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            logits: Predicted logits [B, C, H, W]
            targets: Ground truth masks [B, H, W] with class indices

        Returns:
            Dice loss value
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # Create mask for valid pixels
        valid_mask = targets != self.ignore_index  # [B, H, W]

        # One-hot encode targets
        targets_one_hot = F.one_hot(
            targets.clamp(0, self.num_classes - 1), num_classes=self.num_classes
        )  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Apply valid mask
        valid_mask = valid_mask.unsqueeze(1).float()  # [B, 1, H, W]
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        # Compute Dice coefficient per class
        # Flatten spatial dimensions
        probs = probs.flatten(2)  # [B, C, H*W]
        targets_one_hot = targets_one_hot.flatten(2)  # [B, C, H*W]

        # Intersection and union
        intersection = (probs * targets_one_hot).sum(dim=2)  # [B, C]
        cardinality = probs.sum(dim=2) + targets_one_hot.sum(dim=2)  # [B, C]

        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )  # [B, C]

        # Dice loss
        dice_loss = 1.0 - dice  # [B, C]

        # Apply reduction
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class FocalLossPixelwise(nn.Module):
    """Focal loss for dense pixel-wise prediction.

    Handles class imbalance by down-weighting easy examples.

    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
        gamma: Focusing parameter for modulating loss (default: 2.0)
        ignore_index: Index to ignore in loss computation (default: 255)
        reduction: Reduction method ('mean', 'sum', 'none') (default: 'mean')
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = 255,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Predicted logits [B, C, H, W]
            targets: Ground truth masks [B, H, W] with class indices

        Returns:
            Focal loss value
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # Flatten
        _B, C, _H, _W = logits.shape
        logits = logits.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        targets = targets.reshape(-1)  # [B*H*W]
        probs = probs.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]

        # Create valid mask
        valid_mask = targets != self.ignore_index

        # Filter valid pixels
        logits = logits[valid_mask]
        targets = targets[valid_mask]
        probs = probs[valid_mask]

        # Get probabilities of target classes
        targets_one_hot = F.one_hot(targets, num_classes=C).float()
        pt = (probs * targets_one_hot).sum(dim=1)  # [N]

        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Compute cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Apply focal weight and alpha
        loss = self.alpha * focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class TverskyLoss(nn.Module):
    """Tversky loss for segmentation.

    Generalization of Dice loss with separate control over false positives and false negatives.

    Args:
        num_classes: Number of segmentation classes
        alpha: Weight for false positives (default: 0.5)
        beta: Weight for false negatives (default: 0.5)
        smooth: Smoothing constant (default: 1.0)
        ignore_index: Index to ignore in loss computation (default: 255)
        reduction: Reduction method ('mean', 'sum', 'none') (default: 'mean')
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        ignore_index: int = 255,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Tversky loss.

        Args:
            logits: Predicted logits [B, C, H, W]
            targets: Ground truth masks [B, H, W] with class indices

        Returns:
            Tversky loss value
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # Create mask for valid pixels
        valid_mask = targets != self.ignore_index  # [B, H, W]

        # One-hot encode targets
        targets_one_hot = F.one_hot(
            targets.clamp(0, self.num_classes - 1), num_classes=self.num_classes
        )  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Apply valid mask
        valid_mask = valid_mask.unsqueeze(1).float()  # [B, 1, H, W]
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        # Flatten spatial dimensions
        probs = probs.flatten(2)  # [B, C, H*W]
        targets_one_hot = targets_one_hot.flatten(2)  # [B, C, H*W]

        # True positives, false positives, false negatives
        tp = (probs * targets_one_hot).sum(dim=2)  # [B, C]
        fp = (probs * (1 - targets_one_hot)).sum(dim=2)  # [B, C]
        fn = ((1 - probs) * targets_one_hot).sum(dim=2)  # [B, C]

        # Tversky index
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        # Tversky loss
        tversky_loss = 1.0 - tversky  # [B, C]

        # Apply reduction
        if self.reduction == "mean":
            return tversky_loss.mean()
        elif self.reduction == "sum":
            return tversky_loss.sum()
        else:
            return tversky_loss


class MaskLoss(nn.Module):
    """Binary cross-entropy loss for instance segmentation masks.

    Args:
        reduction: Reduction method ('mean', 'sum', 'none') (default: 'mean')
        pos_weight: Weight for positive examples (default: None)
    """

    def __init__(
        self,
        reduction: str = "mean",
        pos_weight: float | None = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(
        self, pred_masks: torch.Tensor, target_masks: torch.Tensor
    ) -> torch.Tensor:
        """Compute binary cross-entropy loss for masks.

        Args:
            pred_masks: Predicted masks [N, H, W] or [N, 1, H, W] (logits)
            target_masks: Ground truth binary masks [N, H, W] or [N, 1, H, W]

        Returns:
            Mask loss value
        """
        # Ensure same shape
        if pred_masks.dim() == 4:
            pred_masks = pred_masks.squeeze(1)
        if target_masks.dim() == 4:
            target_masks = target_masks.squeeze(1)

        # Flatten
        pred_masks = pred_masks.flatten()
        target_masks = target_masks.flatten()

        # Compute BCE loss
        pos_weight_tensor = None
        if self.pos_weight is not None:
            pos_weight_tensor = torch.tensor(
                [self.pos_weight], device=pred_masks.device, dtype=pred_masks.dtype
            )

        loss = F.binary_cross_entropy_with_logits(
            pred_masks,
            target_masks,
            reduction=self.reduction,
            pos_weight=pos_weight_tensor,
        )

        return loss


class CombinedSegmentationLoss(nn.Module):
    """Combined cross-entropy and Dice loss for semantic segmentation.

    Args:
        num_classes: Number of segmentation classes
        ce_weight: Weight for cross-entropy loss (default: 1.0)
        dice_weight: Weight for Dice loss (default: 1.0)
        ignore_index: Index to ignore in loss computation (default: 255)
        class_weights: Optional per-class weights for CE loss (default: None)
    """

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        ignore_index: int = 255,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

        # Cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction="mean",
        )

        # Dice loss
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            reduction="mean",
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits: Predicted logits [B, C, H, W]
            targets: Ground truth masks [B, H, W] with class indices

        Returns:
            Combined loss value
        """
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        return self.ce_weight * ce + self.dice_weight * dice
