"""Shared FCOS target-assignment, loss, and box-decoding helpers.

Extracted from :class:`autotimm.tasks.object_detection.ObjectDetector` (the
one task with a validated, tested FCOS implementation) so the same logic can
be reused by any other FCOS-headed task — e.g.
:class:`autotimm.tasks.instance_segmentation.InstanceSegmentor`, whose
detection head shares the same architecture and per-level (points, strides,
regress_ranges) convention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision import ops


def compute_targets_per_level(
    points: torch.Tensor,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    regress_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute FCOS targets for a single image at one FPN level.

    Args:
        points: Grid points [H, W, 2], in image coordinates (cell centers).
        boxes: Target boxes [N, 4] in xyxy format.
        labels: Target labels [N].
        regress_range: (min, max) allowed regression distance for this level.

    Returns:
        cls_target: [H, W] with class labels or -1 for ignore.
        reg_target: [H, W, 4] with (l, t, r, b) distances.
        centerness_target: [H, W] with centerness values.
    """
    device = points.device
    feat_h, feat_w = points.shape[:2]

    points_flat = points.reshape(-1, 2)  # [H*W, 2]
    num_points = points_flat.shape[0]

    boxes_exp = boxes.unsqueeze(0)  # [1, N, 4]
    points_exp = points_flat.unsqueeze(1)  # [H*W, 1, 2]

    left = points_exp[..., 0] - boxes_exp[..., 0]  # [H*W, N]
    top = points_exp[..., 1] - boxes_exp[..., 1]
    right = boxes_exp[..., 2] - points_exp[..., 0]
    bottom = boxes_exp[..., 3] - points_exp[..., 1]

    reg_targets_per_box = torch.stack([left, top, right, bottom], dim=-1)  # [H*W, N, 4]

    inside_box = (left > 0) & (top > 0) & (right > 0) & (bottom > 0)  # [H*W, N]

    max_reg = reg_targets_per_box.max(dim=-1)[0]  # [H*W, N]
    min_range, max_range = regress_range
    in_range = (max_reg >= min_range) & (max_reg < max_range)

    valid = inside_box & in_range  # [H*W, N]

    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # [N]
    box_areas_exp = box_areas.unsqueeze(0).expand(num_points, -1)  # [H*W, N]

    box_areas_masked = torch.where(
        valid, box_areas_exp, torch.tensor(float("inf"), device=device)
    )

    min_areas, best_box_idx = box_areas_masked.min(dim=1)  # [H*W]
    has_assignment = min_areas < float("inf")

    cls_target = torch.full((num_points,), -1, dtype=torch.long, device=device)
    reg_target = torch.zeros(num_points, 4, device=device)
    centerness_target = torch.zeros(num_points, device=device)

    if has_assignment.any():
        assigned_idx = best_box_idx[has_assignment]
        cls_target[has_assignment] = labels[assigned_idx]

        point_indices = torch.arange(num_points, device=device)[has_assignment]
        reg_target[has_assignment] = reg_targets_per_box[point_indices, assigned_idx]

        lr = reg_target[has_assignment]
        left_right_min = torch.min(lr[:, 0], lr[:, 2])
        left_right_max = torch.max(lr[:, 0], lr[:, 2])
        top_bottom_min = torch.min(lr[:, 1], lr[:, 3])
        top_bottom_max = torch.max(lr[:, 1], lr[:, 3])

        centerness = torch.sqrt(
            (left_right_min / left_right_max.clamp(min=1e-7))
            * (top_bottom_min / top_bottom_max.clamp(min=1e-7))
        )
        centerness_target[has_assignment] = centerness

    cls_target = cls_target.reshape(feat_h, feat_w)
    reg_target = reg_target.reshape(feat_h, feat_w, 4)
    centerness_target = centerness_target.reshape(feat_h, feat_w)

    return cls_target, reg_target, centerness_target


def compute_iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """IoU-based regression loss for LTRB predictions (summed, not reduced)."""
    pred_area = (pred[:, 0] + pred[:, 2]) * (pred[:, 1] + pred[:, 3])
    target_area = (target[:, 0] + target[:, 2]) * (target[:, 1] + target[:, 3])

    inter_w = torch.min(pred[:, 0], target[:, 0]) + torch.min(pred[:, 2], target[:, 2])
    inter_h = torch.min(pred[:, 1], target[:, 1]) + torch.min(pred[:, 3], target[:, 3])
    inter_area = inter_w * inter_h

    union_area = pred_area + target_area - inter_area

    iou = inter_area / union_area.clamp(min=1e-7)
    loss = -torch.log(iou.clamp(min=1e-7))

    return loss.sum()


def _grid_points(feat_h: int, feat_w: int, stride: int, device, dtype=torch.float32):
    grid_y, grid_x = torch.meshgrid(
        torch.arange(feat_h, device=device, dtype=dtype),
        torch.arange(feat_w, device=device, dtype=dtype),
        indexing="ij",
    )
    points_x = (grid_x + 0.5) * stride
    points_y = (grid_y + 0.5) * stride
    return points_x, points_y


def _ltrb_to_relative_boxes(ltrb: torch.Tensor) -> torch.Tensor:
    """Convert LTRB distances to boxes relative to their grid point.

    A point with distances (l, t, r, b) corresponds to the box
    (-l, -t, r, b) in a coordinate frame centered on the point. IoU/GIoU
    are translation-invariant, so box losses computed on these relative
    boxes equal those on the absolute boxes.
    """
    return torch.stack([-ltrb[:, 0], -ltrb[:, 1], ltrb[:, 2], ltrb[:, 3]], dim=-1)


def compute_fcos_detection_loss(
    cls_outputs: list[torch.Tensor],
    reg_outputs: list[torch.Tensor],
    centerness_outputs: list[torch.Tensor] | None,
    target_boxes: list[torch.Tensor],
    target_labels: list[torch.Tensor],
    strides: tuple[int, ...],
    regress_ranges: tuple[tuple[float, float], ...],
    focal_loss_fn,
    num_classes: int,
    reg_loss_fn=None,
) -> dict[str, torch.Tensor]:
    """Compute FCOS classification/regression/centerness losses across all levels.

    Mirrors the per-level target assignment used in
    ``ObjectDetector.training_step`` for ``detection_arch="fcos"``.

    ``centerness_outputs`` may be ``None`` (e.g. YOLOX heads don't predict
    centerness), in which case ``centerness_loss`` is always zero.

    ``reg_loss_fn``, when provided, is called with predicted/target boxes in
    xyxy format (relative to each grid point) and should return a summed
    loss — e.g. ``GIoULoss(reduction="sum")``. When ``None``, the default
    ``-log(IoU)`` loss on LTRB distances is used.

    Returns:
        Dict with ``cls_loss``, ``reg_loss``, ``centerness_loss`` (each already
        normalized by the number of positive samples) and ``num_pos``.
    """
    device = cls_outputs[0].device
    batch_size = cls_outputs[0].shape[0]

    total_cls_loss = torch.tensor(0.0, device=device)
    total_reg_loss = torch.tensor(0.0, device=device)
    total_centerness_loss = torch.tensor(0.0, device=device)
    num_pos = 0

    centerness_iter = (
        [None] * len(cls_outputs) if centerness_outputs is None else centerness_outputs
    )

    for level_idx, (cls_out, reg_out, cent_out) in enumerate(
        zip(cls_outputs, reg_outputs, centerness_iter)
    ):
        stride = strides[level_idx]
        feat_h, feat_w = cls_out.shape[-2:]
        points_x, points_y = _grid_points(feat_h, feat_w, stride, device)
        points = torch.stack([points_x, points_y], dim=-1)  # [H, W, 2]

        level_cls_targets = []
        level_reg_targets = []
        level_centerness_targets = []

        for b in range(batch_size):
            boxes = target_boxes[b]
            labels = target_labels[b]

            if len(boxes) == 0:
                cls_target = torch.full(
                    (feat_h, feat_w), -1, dtype=torch.long, device=device
                )
                reg_target = torch.zeros(feat_h, feat_w, 4, device=device)
                cent_target = torch.zeros(feat_h, feat_w, device=device)
            else:
                cls_target, reg_target, cent_target = compute_targets_per_level(
                    points, boxes, labels, regress_ranges[level_idx]
                )

            level_cls_targets.append(cls_target)
            level_reg_targets.append(reg_target)
            level_centerness_targets.append(cent_target)

        cls_targets = torch.stack(level_cls_targets)  # [B, H, W]
        reg_targets = torch.stack(level_reg_targets)  # [B, H, W, 4]
        cent_targets = torch.stack(level_centerness_targets)  # [B, H, W]

        cls_out_flat = cls_out.permute(0, 2, 3, 1).reshape(-1, num_classes)
        cls_targets_flat = cls_targets.reshape(-1)
        pos_cls_mask = cls_targets_flat >= 0

        # FCOS uses independent sigmoid classifiers. Every unassigned point is
        # a background negative (an all-zero target), not an ignored sample.
        # Ignoring them means the detector never learns to suppress false
        # positives and an all-background batch has no classification gradient.
        dense_cls_targets = torch.zeros_like(cls_out_flat)
        if pos_cls_mask.any():
            positive_labels = cls_targets_flat[pos_cls_mask]
            if (positive_labels >= num_classes).any():
                raise ValueError(
                    "Target class index exceeds num_classes: "
                    f"max label={positive_labels.max().item()}, "
                    f"num_classes={num_classes}"
                )
            dense_cls_targets[pos_cls_mask, positive_labels] = 1.0

        total_cls_loss = total_cls_loss + focal_loss_fn(cls_out_flat, dense_cls_targets)

        pos_mask = cls_targets >= 0  # [B, H, W]

        if pos_mask.any():
            pos_reg_pred = reg_out.permute(0, 2, 3, 1)[pos_mask]
            pos_reg_target = reg_targets[pos_mask]
            if reg_loss_fn is not None:
                total_reg_loss = total_reg_loss + reg_loss_fn(
                    _ltrb_to_relative_boxes(pos_reg_pred),
                    _ltrb_to_relative_boxes(pos_reg_target),
                )
            else:
                total_reg_loss = total_reg_loss + compute_iou_loss(
                    pos_reg_pred, pos_reg_target
                )

            if cent_out is not None:
                pos_cent_pred = cent_out.squeeze(1)[pos_mask]
                pos_cent_target = cent_targets[pos_mask]
                cent_loss = F.binary_cross_entropy_with_logits(
                    pos_cent_pred, pos_cent_target, reduction="sum"
                )
                total_centerness_loss = total_centerness_loss + cent_loss

            num_pos += pos_mask.sum().item()

    num_pos = max(num_pos, 1)

    return {
        "cls_loss": total_cls_loss / num_pos,
        "reg_loss": total_reg_loss / num_pos,
        "centerness_loss": total_centerness_loss / num_pos,
        "num_pos": num_pos,
    }


def decode_fcos_detections(
    cls_outputs: list[torch.Tensor],
    reg_outputs: list[torch.Tensor],
    centerness_outputs: list[torch.Tensor] | None,
    strides: tuple[int, ...],
    num_classes: int,
    img_size: tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    max_detections_per_image: int,
) -> list[dict[str, torch.Tensor]]:
    """Decode raw FCOS head outputs into per-image boxes/scores/labels (post-NMS).

    Mirrors ``ObjectDetector.predict()``.
    """
    batch_size = cls_outputs[0].shape[0]
    img_h, img_w = img_size
    device = cls_outputs[0].device

    if centerness_outputs is None:
        centerness_iter = [None] * len(cls_outputs)
    else:
        centerness_iter = centerness_outputs

    all_detections = []

    for b in range(batch_size):
        all_boxes = []
        all_scores = []
        all_labels = []

        for level_idx, (cls_out, reg_out, cent_out) in enumerate(
            zip(cls_outputs, reg_outputs, centerness_iter)
        ):
            stride = strides[level_idx]
            feat_h, feat_w = cls_out.shape[-2:]

            cls_logits = cls_out[b]  # [C, H, W]
            reg_pred = reg_out[b]  # [4, H, W]

            points_x, points_y = _grid_points(feat_h, feat_w, stride, device)

            cls_logits = cls_logits.permute(1, 2, 0).reshape(-1, num_classes)
            reg_pred = reg_pred.permute(1, 2, 0).reshape(-1, 4)
            points_x = points_x.reshape(-1)
            points_y = points_y.reshape(-1)

            cls_scores = cls_logits.sigmoid()
            if cent_out is not None:
                cent_pred = cent_out[b, 0].reshape(-1)
                centerness = cent_pred.sigmoid()
                scores = cls_scores * centerness.unsqueeze(-1)
            else:
                scores = cls_scores

            max_scores, class_ids = scores.max(dim=-1)

            keep = max_scores > score_thresh
            if not keep.any():
                continue

            max_scores = max_scores[keep]
            class_ids = class_ids[keep]
            reg_pred = reg_pred[keep]
            points_x = points_x[keep]
            points_y = points_y[keep]

            left, top, right, bottom = (
                reg_pred[:, 0],
                reg_pred[:, 1],
                reg_pred[:, 2],
                reg_pred[:, 3],
            )
            x1 = (points_x - left).clamp(min=0, max=img_w)
            y1 = (points_y - top).clamp(min=0, max=img_h)
            x2 = (points_x + right).clamp(min=0, max=img_w)
            y2 = (points_y + bottom).clamp(min=0, max=img_h)

            boxes = torch.stack([x1, y1, x2, y2], dim=-1)

            all_boxes.append(boxes)
            all_scores.append(max_scores)
            all_labels.append(class_ids)

        if len(all_boxes) > 0:
            boxes = torch.cat(all_boxes, dim=0)
            scores = torch.cat(all_scores, dim=0)
            labels = torch.cat(all_labels, dim=0)

            keep_indices = ops.batched_nms(boxes, scores, labels, nms_thresh)
            keep_indices = keep_indices[:max_detections_per_image]

            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
        else:
            boxes = torch.zeros((0, 4), device=device)
            scores = torch.zeros((0,), device=device)
            labels = torch.zeros((0,), dtype=torch.long, device=device)

        all_detections.append({"boxes": boxes, "scores": scores, "labels": labels})

    return all_detections
