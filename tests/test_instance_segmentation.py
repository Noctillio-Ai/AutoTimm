"""Tests for InstanceSegmentor.

Previously this task had no dedicated test file at all. That gap hid a real
bug: `_compute_detection_loss` was a hardcoded placeholder that always
returned zero loss regardless of the detection head's predictions (so the
detection head never received a training signal), and `predict()`
unconditionally returned empty boxes/scores/labels/masks. Both are now real
implementations built on the shared FCOS logic in
`autotimm.tasks._fcos_targets` (the same code ObjectDetector uses). These
tests assert on the specific failure mode that was present before the fix:
gradients must actually reach the detection head, and predict() must be able
to return non-empty, correctly-shaped detections.
"""

import pytest
import torch

from autotimm.tasks.instance_segmentation import InstanceSegmentor


@pytest.fixture
def model():
    return InstanceSegmentor(
        backbone="resnet18",
        num_classes=5,
        metrics=None,
        compile_model=False,
    )


def _batch(img_size=256):
    return {
        "image": torch.randn(2, 3, img_size, img_size),
        "boxes": [
            torch.tensor([[20.0, 20.0, 100.0, 100.0], [130.0, 130.0, 200.0, 200.0]]),
            torch.tensor([[50.0, 50.0, 150.0, 150.0]]),
        ],
        "labels": [torch.tensor([1, 2]), torch.tensor([3])],
        "masks": [
            torch.zeros(2, img_size, img_size),
            torch.zeros(1, img_size, img_size),
        ],
    }


def test_forward_shapes(model):
    images = torch.randn(2, 3, 256, 256)
    cls_outputs, reg_outputs, centerness_outputs = model(images)

    # 3 backbone levels + 2 extra FPN levels = 5 (P3-P7)
    assert len(cls_outputs) == len(reg_outputs) == len(centerness_outputs) == 5
    assert cls_outputs[0].shape[1] == 5  # num_classes
    assert reg_outputs[0].shape[1] == 4
    assert centerness_outputs[0].shape[1] == 1


def test_training_step_detection_head_receives_gradients(model):
    """Regression test for the zero-loss placeholder bug: cls_loss/reg_loss/
    centerness_loss must be real, differentiable losses tied to the
    detection head's actual predictions, not constants."""
    batch = _batch()
    loss = model.training_step(batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0

    loss.backward()

    det_head_grad = sum(
        p.grad.abs().sum().item()
        for p in model.detection_head.parameters()
        if p.grad is not None
    )
    mask_head_grad = sum(
        p.grad.abs().sum().item()
        for p in model.mask_head.parameters()
        if p.grad is not None
    )

    assert det_head_grad > 0, "detection_head got no gradient from training_step"
    assert mask_head_grad > 0, "mask_head got no gradient from training_step"


def test_training_step_logs_nonzero_component_losses(model):
    batch = _batch()
    logged = {}
    model.log = lambda name, value, **kwargs: logged.__setitem__(
        name, value.detach() if isinstance(value, torch.Tensor) else value
    )

    model.training_step(batch, batch_idx=0)

    # cls_loss must depend on real predictions, not be hardcoded to 0.
    assert float(logged["train/cls_loss"]) != 0.0


def test_training_step_no_targets_still_runs(model):
    batch = {
        "image": torch.randn(1, 3, 256, 256),
        "boxes": [torch.zeros(0, 4)],
        "labels": [torch.zeros(0, dtype=torch.long)],
        "masks": [torch.zeros(0, 256, 256)],
    }
    loss = model.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)


def test_predict_returns_correct_structure(model):
    """Regression test for the always-empty predict() placeholder: the
    return shape/keys must be correct regardless of how many detections
    survive score thresholding."""
    model.eval()
    images = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        predictions = model.predict(images)

    assert len(predictions) == 2
    for pred in predictions:
        assert set(pred.keys()) == {"boxes", "labels", "scores", "masks"}
        n = pred["boxes"].shape[0]
        assert pred["labels"].shape[0] == n
        assert pred["scores"].shape[0] == n
        assert pred["masks"].shape == (n, 256, 256)
        assert pred["masks"].dtype == torch.bool


def test_predict_can_produce_nonempty_detections(model):
    """With score_thresh=0, every location passes threshold, so at least one
    of the two images should have a detection after NMS — proving predict()
    is no longer hardcoded to always return empty results."""
    model.score_thresh = 0.0
    model.eval()
    images = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        predictions = model.predict(images)

    total_detections = sum(p["boxes"].shape[0] for p in predictions)
    assert total_detections > 0


def test_validation_step_runs(model):
    model.eval()
    batch = _batch()
    with torch.no_grad():
        model.validation_step(batch, batch_idx=0)


def test_test_step_runs(model):
    model.eval()
    batch = _batch()
    with torch.no_grad():
        model.test_step(batch, batch_idx=0)


def test_invalid_backbone_still_uses_three_feature_levels():
    """strides/regress_ranges assume 5 FPN levels (P3-P7); backbone construction
    must be pinned to 3 output levels (C3-C5) to match, regardless of the
    default out_indices a plain model-name string would otherwise resolve to."""
    m = InstanceSegmentor(
        backbone="resnet18",
        num_classes=3,
        metrics=None,
        compile_model=False,
    )
    assert len(m.strides) == 5
    assert len(m.regress_ranges) == 5
    images = torch.randn(1, 3, 128, 128)
    cls_outputs, _, _ = m(images)
    assert len(cls_outputs) == 5
