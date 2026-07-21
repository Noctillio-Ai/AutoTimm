"""Tests for the FCOS-specific detection path.

test_yolox.py exercises ObjectDetector mostly with detection_arch="yolox";
this file covers the FCOS-only pieces that had no dedicated coverage:
target assignment (autotimm.tasks._fcos_targets), FCOSLoss, and
ObjectDetector's training/validation/test/predict steps under
detection_arch="fcos".
"""

import pytest
import torch

from autotimm.heads import DetectionHead
from autotimm.losses import FCOSLoss
from autotimm.tasks.object_detection import ObjectDetector
from autotimm.tasks._fcos_targets import (
    compute_fcos_detection_loss,
    compute_iou_loss,
    compute_targets_per_level,
    decode_fcos_detections,
)

# ---------------------------------------------------------------------------
# compute_targets_per_level
# ---------------------------------------------------------------------------


def _grid(feat_h, feat_w, stride):
    grid_y, grid_x = torch.meshgrid(
        torch.arange(feat_h, dtype=torch.float32),
        torch.arange(feat_w, dtype=torch.float32),
        indexing="ij",
    )
    points_x = (grid_x + 0.5) * stride
    points_y = (grid_y + 0.5) * stride
    return torch.stack([points_x, points_y], dim=-1)


def test_compute_targets_per_level_assigns_center_point():
    """A point inside a single box, within the regress range, should be assigned
    that box's label and a centerness close to 1 near the box center."""
    stride = 8
    points = _grid(10, 10, stride)  # grid cell (4, 4) sits at (36, 36)

    # Box exactly centered on grid point (4, 4) = (36, 36), half-extent 20
    boxes = torch.tensor([[16.0, 16.0, 56.0, 56.0]])
    labels = torch.tensor([3])

    cls_target, reg_target, centerness_target = compute_targets_per_level(
        points, boxes, labels, regress_range=(-1, 64)
    )

    assert cls_target[4, 4] == 3
    # Point is exactly centered -> l == t == r == b -> centerness == 1
    assert centerness_target[4, 4] == pytest.approx(1.0, abs=1e-5)

    # Points far outside the box should be ignored (-1)
    assert cls_target[0, 0] == -1
    assert reg_target[0, 0].abs().sum() == 0


def test_compute_targets_per_level_no_assignment_outside_regress_range():
    """A box whose max LTRB distance exceeds the level's regress range should
    not be assigned, even though the point is inside the box."""
    points = _grid(4, 4, stride=32)  # points span 0..128
    boxes = torch.tensor([[0.0, 0.0, 128.0, 128.0]])  # huge box
    labels = torch.tensor([1])

    # regress range (-1, 8) is far too small for this box at any point
    cls_target, _, _ = compute_targets_per_level(points, boxes, labels, (-1, 8))
    assert (cls_target == -1).all()


def test_compute_targets_per_level_picks_smaller_box_on_overlap():
    """When two boxes overlap a point, the smaller-area box wins."""
    points = _grid(10, 10, stride=8)

    # Big box covers most of the grid; small box is nested inside it
    big_box = [0.0, 0.0, 80.0, 80.0]
    small_box = [30.0, 30.0, 50.0, 50.0]
    boxes = torch.tensor([big_box, small_box])
    labels = torch.tensor([1, 2])

    cls_target, _, _ = compute_targets_per_level(
        points, boxes, labels, regress_range=(-1, float("inf"))
    )

    # Center of the small box should be assigned to the smaller box's label
    assert cls_target[4, 4] == 2


def test_compute_iou_loss_perfect_overlap_near_zero():
    pred = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    target = pred.clone()
    loss = compute_iou_loss(pred, target)
    assert loss.item() < 1e-3


def test_compute_iou_loss_partial_overlap_positive():
    pred = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    target = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
    loss = compute_iou_loss(pred, target)
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# compute_fcos_detection_loss / decode_fcos_detections (shared module)
# ---------------------------------------------------------------------------


def test_compute_fcos_detection_loss_zero_targets_is_zero():
    from autotimm.losses import FocalLoss

    cls_outputs = [torch.randn(2, 5, 8, 8, requires_grad=True)]
    reg_outputs = [torch.randn(2, 4, 8, 8, requires_grad=True)]
    centerness_outputs = [torch.randn(2, 1, 8, 8, requires_grad=True)]

    losses = compute_fcos_detection_loss(
        cls_outputs,
        reg_outputs,
        centerness_outputs,
        target_boxes=[torch.zeros(0, 4), torch.zeros(0, 4)],
        target_labels=[
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        ],
        strides=(8,),
        regress_ranges=((-1, float("inf")),),
        focal_loss_fn=FocalLoss(reduction="sum"),
        num_classes=5,
    )
    assert losses["cls_loss"].item() == pytest.approx(0.0)
    assert losses["reg_loss"].item() == pytest.approx(0.0)
    assert losses["centerness_loss"].item() == pytest.approx(0.0)
    assert losses["num_pos"] == 1  # max(num_pos, 1) floor


def test_compute_fcos_detection_loss_with_targets_nonzero_and_differentiable():
    from autotimm.losses import FocalLoss

    cls_outputs = [torch.randn(1, 5, 8, 8, requires_grad=True)]
    reg_outputs = [(torch.rand(1, 4, 8, 8) * 20 + 5).requires_grad_()]
    centerness_outputs = [torch.randn(1, 1, 8, 8, requires_grad=True)]

    boxes = [torch.tensor([[10.0, 10.0, 40.0, 40.0]])]
    labels = [torch.tensor([2])]

    losses = compute_fcos_detection_loss(
        cls_outputs,
        reg_outputs,
        centerness_outputs,
        boxes,
        labels,
        strides=(8,),
        regress_ranges=((-1, float("inf")),),
        focal_loss_fn=FocalLoss(reduction="sum"),
        num_classes=5,
    )

    total = losses["cls_loss"] + losses["reg_loss"] + losses["centerness_loss"]
    assert total.item() > 0
    assert losses["num_pos"] >= 1

    total.backward()
    assert cls_outputs[0].grad is not None
    assert cls_outputs[0].grad.abs().sum().item() > 0
    assert reg_outputs[0].grad is not None
    assert reg_outputs[0].grad.abs().sum().item() > 0


def test_compute_fcos_detection_loss_no_centerness_outputs():
    """centerness_outputs=None (YOLOX-style) should yield zero centerness loss."""
    from autotimm.losses import FocalLoss

    cls_outputs = [torch.randn(1, 5, 8, 8, requires_grad=True)]
    reg_outputs = [torch.rand(1, 4, 8, 8, requires_grad=True) * 20 + 5]

    losses = compute_fcos_detection_loss(
        cls_outputs,
        reg_outputs,
        None,
        [torch.tensor([[10.0, 10.0, 40.0, 40.0]])],
        [torch.tensor([1])],
        strides=(8,),
        regress_ranges=((-1, float("inf")),),
        focal_loss_fn=FocalLoss(reduction="sum"),
        num_classes=5,
    )
    assert losses["centerness_loss"].item() == 0.0


def test_decode_fcos_detections_empty_when_below_threshold():
    cls_outputs = [torch.full((1, 5, 4, 4), -10.0)]  # very low logits -> low sigmoid
    reg_outputs = [torch.rand(1, 4, 4, 4) * 10]
    centerness_outputs = [torch.full((1, 1, 4, 4), -10.0)]

    detections = decode_fcos_detections(
        cls_outputs,
        reg_outputs,
        centerness_outputs,
        strides=(8,),
        num_classes=5,
        img_size=(32, 32),
        score_thresh=0.5,
        nms_thresh=0.5,
        max_detections_per_image=100,
    )
    assert len(detections) == 1
    assert detections[0]["boxes"].shape == (0, 4)


def test_decode_fcos_detections_finds_high_confidence_box():
    cls_outputs = [torch.full((1, 5, 4, 4), -10.0)]
    cls_outputs[0][0, 2, 1, 1] = 10.0  # very confident class-2 prediction at (1,1)
    reg_outputs = [torch.full((1, 4, 4, 4), 5.0)]  # small symmetric LTRB box
    centerness_outputs = [torch.full((1, 1, 4, 4), 10.0)]

    detections = decode_fcos_detections(
        cls_outputs,
        reg_outputs,
        centerness_outputs,
        strides=(8,),
        num_classes=5,
        img_size=(32, 32),
        score_thresh=0.5,
        nms_thresh=0.5,
        max_detections_per_image=100,
    )
    assert detections[0]["boxes"].shape[0] == 1
    assert detections[0]["labels"][0].item() == 2


# ---------------------------------------------------------------------------
# FCOSLoss (public API, previously untested)
# ---------------------------------------------------------------------------


def test_fcos_loss_runs_and_returns_expected_keys():
    loss_fn = FCOSLoss(num_classes=5)

    cls_preds = [torch.randn(1, 5, 4, 4)]
    reg_preds = [torch.rand(1, 4, 4, 4) * 10 + 1]
    centerness_preds = [torch.randn(1, 1, 4, 4)]

    cls_targets = [torch.full((1, 4, 4), -1, dtype=torch.long)]
    cls_targets[0][0, 1, 1] = 2
    reg_targets = [torch.rand(1, 4, 4, 4) * 10 + 1]
    centerness_targets = [torch.rand(1, 4, 4)]

    result = loss_fn(
        cls_preds,
        reg_preds,
        centerness_preds,
        cls_targets,
        reg_targets,
        centerness_targets,
    )

    assert set(result.keys()) == {
        "cls_loss",
        "reg_loss",
        "centerness_loss",
        "total_loss",
    }
    for v in result.values():
        assert torch.isfinite(v)
    assert result["total_loss"] == pytest.approx(
        (result["cls_loss"] + result["reg_loss"] + result["centerness_loss"]).item()
    )


def test_fcos_loss_no_positive_samples_gives_zero_reg_and_centerness():
    loss_fn = FCOSLoss(num_classes=3)

    cls_preds = [torch.randn(1, 3, 4, 4)]
    reg_preds = [torch.rand(1, 4, 4, 4)]
    centerness_preds = [torch.randn(1, 1, 4, 4)]

    # All background (-1) => no positive samples
    cls_targets = [torch.full((1, 4, 4), -1, dtype=torch.long)]
    reg_targets = [torch.zeros(1, 4, 4, 4)]
    centerness_targets = [torch.zeros(1, 4, 4)]

    result = loss_fn(
        cls_preds,
        reg_preds,
        centerness_preds,
        cls_targets,
        reg_targets,
        centerness_targets,
    )
    # With no positive samples, FCOSLoss never accumulates a tensor for these
    # terms and they stay as the plain python float they were initialized to.
    assert float(result["reg_loss"]) == 0.0
    assert float(result["centerness_loss"]) == 0.0


# ---------------------------------------------------------------------------
# ObjectDetector end-to-end, detection_arch="fcos"
# ---------------------------------------------------------------------------


@pytest.fixture
def fcos_model():
    return ObjectDetector(
        backbone="resnet18",
        num_classes=10,
        detection_arch="fcos",
        metrics=None,
        compile_model=False,
    )


def _fcos_batch():
    return {
        "image": torch.randn(2, 3, 640, 640),
        "boxes": [
            torch.tensor([[100.0, 100.0, 200.0, 200.0], [300.0, 300.0, 400.0, 400.0]]),
            torch.tensor([[150.0, 150.0, 250.0, 250.0]]),
        ],
        "labels": [torch.tensor([1, 2]), torch.tensor([3])],
    }


def test_fcos_head_is_detection_head(fcos_model):
    assert isinstance(fcos_model.head, DetectionHead)


def test_fcos_training_step_produces_gradients(fcos_model):
    batch = _fcos_batch()
    loss = fcos_model.training_step(batch, batch_idx=0)

    assert loss.ndim == 0
    assert loss.item() > 0

    loss.backward()
    head_grad = sum(
        p.grad.abs().sum().item()
        for p in fcos_model.head.parameters()
        if p.grad is not None
    )
    assert head_grad > 0


def test_fcos_training_step_no_targets_still_runs(fcos_model):
    batch = {
        "image": torch.randn(1, 3, 640, 640),
        "boxes": [torch.zeros(0, 4)],
        "labels": [torch.zeros(0, dtype=torch.long)],
    }
    loss = fcos_model.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)


def test_fcos_predict_returns_correct_structure(fcos_model):
    fcos_model.eval()
    images = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        detections = fcos_model.predict(images)

    assert len(detections) == 2
    for det in detections:
        assert set(det.keys()) == {"boxes", "scores", "labels"}
        assert det["boxes"].shape[-1] == 4
        assert det["boxes"].shape[0] == det["scores"].shape[0] == det["labels"].shape[0]


def test_fcos_validation_and_test_step_run(fcos_model):
    fcos_model.eval()
    batch = _fcos_batch()
    with torch.no_grad():
        fcos_model.validation_step(batch, batch_idx=0)
        fcos_model.test_step(batch, batch_idx=0)
