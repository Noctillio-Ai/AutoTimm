"""Tests for YOLOX learning rate scheduler."""

import torch
from torch import nn

from autotimm.models.yolox_scheduler import YOLOXLRScheduler, YOLOXWarmupLR


def test_yolox_scheduler_warmup():
    """Test YOLOX scheduler warmup phase."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = YOLOXLRScheduler(
        optimizer,
        total_epochs=300,
        warmup_epochs=5,
        no_aug_epochs=15,
        min_lr_ratio=0.05,
        scheduler_type="cosine",
    )

    # Epoch 0 already trains at 1/5 of base_lr (not 0)
    assert abs(optimizer.param_groups[0]["lr"] - 0.01 / 5) < 1e-6

    # Epoch 1: 2/5 of base_lr
    scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"] - 0.01 * 2 / 5) < 1e-6

    # Epoch 4 (last warmup epoch) reaches base_lr
    for _ in range(3):
        scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"] - 0.01) < 1e-6


def test_yolox_scheduler_cosine_decay():
    """Test YOLOX scheduler cosine decay phase."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = YOLOXLRScheduler(
        optimizer,
        total_epochs=300,
        warmup_epochs=5,
        no_aug_epochs=15,
        min_lr_ratio=0.05,
        scheduler_type="cosine",
    )

    # Skip warmup
    for _ in range(5):
        scheduler.step()

    # During cosine decay, LR should decrease smoothly
    lr_after_warmup = optimizer.param_groups[0]["lr"]
    scheduler.step()
    lr_after_first_decay = optimizer.param_groups[0]["lr"]

    assert lr_after_first_decay < lr_after_warmup
    assert lr_after_first_decay > 0.01 * 0.05  # Above minimum


def test_yolox_scheduler_no_aug_phase():
    """Test YOLOX scheduler no augmentation phase."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = YOLOXLRScheduler(
        optimizer,
        total_epochs=300,
        warmup_epochs=5,
        no_aug_epochs=15,
        min_lr_ratio=0.05,
        scheduler_type="cosine",
    )

    # Go to no-aug phase (last 15 epochs)
    for _ in range(300 - 15):
        scheduler.step()

    # LR should be at minimum during no-aug phase
    assert abs(optimizer.param_groups[0]["lr"] - 0.01 * 0.05) < 1e-6

    # Should stay at minimum for remaining epochs
    for _ in range(14):
        scheduler.step()
        assert abs(optimizer.param_groups[0]["lr"] - 0.01 * 0.05) < 1e-6


def test_yolox_scheduler_linear_decay():
    """Test YOLOX scheduler with linear decay."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = YOLOXLRScheduler(
        optimizer,
        total_epochs=300,
        warmup_epochs=5,
        no_aug_epochs=15,
        min_lr_ratio=0.05,
        scheduler_type="linear",
    )

    # Skip warmup
    for _ in range(5):
        scheduler.step()

    # During linear decay, LR should decrease linearly
    lr_values = []
    for _ in range(10):
        scheduler.step()
        lr_values.append(optimizer.param_groups[0]["lr"])

    # Check that LR is decreasing
    for i in range(len(lr_values) - 1):
        assert lr_values[i] > lr_values[i + 1]


def test_yolox_warmup_lr():
    """Test simple YOLOX warmup scheduler."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = YOLOXWarmupLR(
        optimizer,
        warmup_epochs=5,
        warmup_lr_start=0.0,
    )

    # First epoch is already non-zero
    assert abs(optimizer.param_groups[0]["lr"] - 0.01 / 5) < 1e-6

    # After warmup, should be at base LR
    for _ in range(5):
        scheduler.step()

    assert abs(optimizer.param_groups[0]["lr"] - 0.01) < 1e-6


def test_yolox_scheduler_multiple_param_groups():
    """Test YOLOX scheduler with multiple parameter groups."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(
        [
            {"params": model.weight, "lr": 0.01},
            {"params": model.bias, "lr": 0.001},
        ]
    )
    scheduler = YOLOXLRScheduler(
        optimizer,
        total_epochs=300,
        warmup_epochs=5,
    )

    # Both groups start at 1/5 of their own base LR
    assert abs(optimizer.param_groups[0]["lr"] - 0.01 / 5) < 1e-6
    assert abs(optimizer.param_groups[1]["lr"] - 0.001 / 5) < 1e-6

    # After warmup, should be at respective base LRs
    for _ in range(5):
        scheduler.step()

    assert abs(optimizer.param_groups[0]["lr"] - 0.01) < 1e-6
    assert abs(optimizer.param_groups[1]["lr"] - 0.001) < 1e-6


def test_yolox_scheduler_shrinks_phases_on_short_runs():
    """warmup + no_aug >= total must not leave the run at min LR.

    Regression: total_epochs=10 with the default warmup=5 / no_aug=15 ran
    epoch 0 at LR 0 and every epoch from 5 on at min LR.
    """
    import pytest

    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.warns(UserWarning, match="no main training phase"):
        scheduler = YOLOXLRScheduler(
            optimizer, total_epochs=10, warmup_epochs=5, no_aug_epochs=15
        )

    assert scheduler.warmup_epochs + scheduler.no_aug_epochs <= 5
    assert scheduler.warmup_epochs >= 1

    lrs = []
    for _ in range(10):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    assert all(lr > 0 for lr in lrs)
    assert max(lrs) == pytest.approx(0.01)
    # At least half of training is spent above the minimum LR
    assert sum(lr > 0.01 * 0.05 + 1e-9 for lr in lrs) >= 5


def test_yolox_scheduler_keeps_phases_that_fit():
    """The official 300-epoch schedule is left untouched (no warning)."""
    import warnings

    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scheduler = YOLOXLRScheduler(
            optimizer, total_epochs=300, warmup_epochs=5, no_aug_epochs=15
        )
    assert (scheduler.warmup_epochs, scheduler.no_aug_epochs) == (5, 15)


def test_yolox_scheduler_single_epoch():
    """Degenerate 1-epoch runs train at base LR without dividing by zero."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    import pytest

    with pytest.warns(UserWarning):
        YOLOXLRScheduler(optimizer, total_epochs=1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
