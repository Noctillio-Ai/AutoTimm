"""Detection losses for FCOS-style object detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in object detection.

    Focal Loss reduces the relative loss for well-classified examples,
    focusing training on hard negatives.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters:
        alpha: Weighting factor for positive examples. Default 0.25.
        gamma: Focusing parameter. Higher values give more weight to hard
            examples. Default 2.0.
        reduction: Reduction method: 'none', 'mean', or 'sum'.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Predicted logits of shape [N, C] or [N, C, H, W].
            targets: Ground truth class indices of shape [N] or [N, H, W].
                Use -1 to ignore samples.

        Returns:
            Focal loss value.
        """
        # Handle spatial dimensions
        if inputs.dim() == 4:
            # [N, C, H, W] -> [N, H, W, C] -> [N*H*W, C]
            n, c, h, w = inputs.shape
            inputs = inputs.permute(0, 2, 3, 1).reshape(-1, c)
            targets = targets.reshape(-1)

        # Filter out ignored samples (targets == -1)
        valid_mask = targets >= 0
        if not valid_mask.any():
            return inputs.sum() * 0.0

        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        # Compute probabilities
        p = torch.sigmoid(inputs)
        num_classes = inputs.shape[-1]

        # Create one-hot encoded targets
        targets_one_hot = F.one_hot(targets, num_classes).float()

        # Compute focal weights
        pt = p * targets_one_hot + (1 - p) * (1 - targets_one_hot)
        focal_weight = (1 - pt) ** self.gamma

        # Compute binary cross entropy
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets_one_hot, reduction="none"
        )

        # Apply focal weight and alpha
        alpha_t = self.alpha * targets_one_hot + (1 - self.alpha) * (
            1 - targets_one_hot
        )
        focal_loss = alpha_t * focal_weight * bce

        # Sum over classes, then reduce
        focal_loss = focal_loss.sum(dim=-1)

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class GIoULoss(nn.Module):
    """Generalized IoU Loss for bounding box regression.

    GIoU provides better gradients than standard IoU loss when boxes
    don't overlap, making it more suitable for training.

    GIoU = IoU - (C - U) / C

    where C is the smallest enclosing box and U is the union.

    Parameters:
        reduction: Reduction method: 'none', 'mean', or 'sum'.
        eps: Small value for numerical stability.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute GIoU loss.

        Args:
            pred_boxes: Predicted boxes [N, 4] in (x1, y1, x2, y2) format.
            target_boxes: Target boxes [N, 4] in (x1, y1, x2, y2) format.
            weights: Optional per-box weights [N].

        Returns:
            GIoU loss value.
        """
        if pred_boxes.numel() == 0:
            return pred_boxes.sum() * 0.0

        giou = self._compute_giou(pred_boxes, target_boxes)
        loss = 1 - giou

        if weights is not None:
            loss = loss * weights

        if self.reduction == "mean":
            return (
                loss.mean()
                if weights is None
                else loss.sum() / weights.sum().clamp(min=self.eps)
            )
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _compute_giou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute GIoU between two sets of boxes."""
        # Intersection
        inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Union
        union_area = area1 + area2 - inter_area

        # IoU
        iou = inter_area / union_area.clamp(min=self.eps)

        # Enclosing box
        enclose_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
        enclose_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
        enclose_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
        enclose_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])

        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

        # GIoU
        giou = iou - (enclose_area - union_area) / enclose_area.clamp(min=self.eps)

        return giou


class CenternessLoss(nn.Module):
    """Binary cross-entropy loss for FCOS centerness prediction.

    Parameters:
        reduction: Reduction method: 'none', 'mean', or 'sum'.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute centerness loss.

        Args:
            pred: Predicted centerness logits [N] or [N, 1].
            target: Target centerness values [N] in [0, 1].
            weights: Optional per-sample weights [N].

        Returns:
            BCE loss value.
        """
        if pred.numel() == 0:
            return pred.sum() * 0.0

        pred = pred.view(-1)
        target = target.view(-1)

        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        if weights is not None:
            weights = weights.view(-1)
            loss = loss * weights

        if self.reduction == "mean":
            if weights is not None:
                return loss.sum() / weights.sum().clamp(min=1e-7)
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FCOSLoss(nn.Module):
    """Combined FCOS loss: Focal Loss + IoU Loss + Centerness Loss.

    The regression term is a simplified IoU loss computed directly from
    predicted/target LTRB distances (see ``_compute_reg_loss``) — it has no
    enclosing-box penalty term, so it is not true GIoU. Computing an actual
    GIoU loss would require converting LTRB distances to absolute boxes using
    the grid center of each location, which isn't threaded through this
    wrapper's inputs.

    This wrapper computes and combines the three loss components used
    in FCOS-style object detection.

    Parameters:
        num_classes: Number of object classes.
        focal_alpha: Alpha parameter for focal loss.
        focal_gamma: Gamma parameter for focal loss.
        cls_weight: Weight for classification loss.
        reg_weight: Weight for regression loss.
        centerness_weight: Weight for centerness loss.
    """

    def __init__(
        self,
        num_classes: int,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        centerness_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.centerness_weight = centerness_weight

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.centerness_loss = CenternessLoss()

    def forward(
        self,
        cls_preds: list[torch.Tensor],
        reg_preds: list[torch.Tensor],
        centerness_preds: list[torch.Tensor],
        cls_targets: list[torch.Tensor],
        reg_targets: list[torch.Tensor],
        centerness_targets: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute combined FCOS loss.

        Args:
            cls_preds: List of classification predictions per level [B, C, H, W].
            reg_preds: List of regression predictions per level [B, 4, H, W].
            centerness_preds: List of centerness predictions per level [B, 1, H, W].
            cls_targets: List of classification targets per level [B, H, W].
            reg_targets: List of regression targets per level [B, 4, H, W].
            centerness_targets: List of centerness targets per level [B, H, W].

        Returns:
            Dict with 'cls_loss', 'reg_loss', 'centerness_loss', and 'total_loss'.
        """
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        total_centerness_loss = 0.0
        num_pos = 0

        for level_idx in range(len(cls_preds)):
            cls_pred = cls_preds[level_idx]
            reg_pred = reg_preds[level_idx]
            centerness_pred = centerness_preds[level_idx]
            cls_target = cls_targets[level_idx]
            reg_target = reg_targets[level_idx]
            centerness_target = centerness_targets[level_idx]

            # Classification loss (all locations)
            total_cls_loss = total_cls_loss + self.focal_loss(cls_pred, cls_target)

            # Find positive samples (cls_target >= 0 and not background)
            # Background is typically the last class or handled via -1
            pos_mask = cls_target >= 0

            if pos_mask.any():
                # Regression loss (positive samples only)
                pos_reg_pred = reg_pred.permute(0, 2, 3, 1)[pos_mask]  # [N_pos, 4]
                pos_reg_target = reg_target.permute(0, 2, 3, 1)[pos_mask]  # [N_pos, 4]

                # Convert LTRB distances to boxes for GIoU
                # This requires knowing the grid locations - simplified here
                # In practice, you'd compute boxes from distances + grid centers
                total_reg_loss = total_reg_loss + self._compute_reg_loss(
                    pos_reg_pred, pos_reg_target
                )

                # Centerness loss (positive samples only)
                pos_centerness_pred = centerness_pred.permute(0, 2, 3, 1)[pos_mask]
                pos_centerness_target = centerness_target[pos_mask]
                total_centerness_loss = total_centerness_loss + self.centerness_loss(
                    pos_centerness_pred, pos_centerness_target
                )

                num_pos += pos_mask.sum()

        # Normalize by number of positive samples
        num_pos = max(num_pos, 1)

        cls_loss = self.cls_weight * total_cls_loss
        reg_loss = self.reg_weight * total_reg_loss / num_pos
        centerness_loss = self.centerness_weight * total_centerness_loss / num_pos
        total_loss = cls_loss + reg_loss + centerness_loss

        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "centerness_loss": centerness_loss,
            "total_loss": total_loss,
        }

    def _compute_reg_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute regression loss using IoU-based loss.

        For FCOS, predictions are (left, top, right, bottom) distances.
        We can compute IoU loss directly from these.
        """
        # Compute areas from LTRB distances
        pred_area = (pred[:, 0] + pred[:, 2]) * (pred[:, 1] + pred[:, 3])
        target_area = (target[:, 0] + target[:, 2]) * (target[:, 1] + target[:, 3])

        # Intersection
        inter_w = torch.min(pred[:, 0], target[:, 0]) + torch.min(
            pred[:, 2], target[:, 2]
        )
        inter_h = torch.min(pred[:, 1], target[:, 1]) + torch.min(
            pred[:, 3], target[:, 3]
        )
        inter_area = inter_w * inter_h

        # Union
        union_area = pred_area + target_area - inter_area

        # IoU loss
        iou = inter_area / union_area.clamp(min=1e-7)
        loss = -torch.log(iou.clamp(min=1e-7))

        return loss.sum()
