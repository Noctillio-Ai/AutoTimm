"""YOLOX Learning Rate Scheduler with Warmup.

Based on the official YOLOX implementation:
https://github.com/Megvii-BaseDetection/YOLOX
"""

import math
import warnings

from torch.optim.lr_scheduler import _LRScheduler


def _fit_phases(
    total_epochs: int, warmup_epochs: int, no_aug_epochs: int
) -> tuple[int, int]:
    """Shrink warmup / no-aug phases that don't fit in ``total_epochs``.

    The official schedule (5 warmup + 15 no-aug) assumes ~300 epochs. On a
    short run the two phases can cover every epoch, so the main phase never
    happens and the LR sits at its minimum right after warmup. When that
    would happen, both phases are scaled down proportionally so they take at
    most half of training.
    """
    warmup_epochs = max(warmup_epochs, 0)
    no_aug_epochs = max(no_aug_epochs, 0)
    if warmup_epochs + no_aug_epochs < total_epochs:
        return warmup_epochs, no_aug_epochs

    budget = total_epochs // 2
    scale = budget / (warmup_epochs + no_aug_epochs)
    new_warmup = int(warmup_epochs * scale)
    if warmup_epochs > 0 and total_epochs > 1:
        new_warmup = max(new_warmup, 1)
    new_no_aug = min(int(no_aug_epochs * scale), max(budget - new_warmup, 0))
    warnings.warn(
        f"YOLOX schedule: warmup_epochs={warmup_epochs} + "
        f"no_aug_epochs={no_aug_epochs} >= total_epochs={total_epochs}, which "
        f"leaves no main training phase. Using warmup_epochs={new_warmup}, "
        f"no_aug_epochs={new_no_aug} instead.",
        UserWarning,
        stacklevel=3,
    )
    return new_warmup, new_no_aug


def _warmup_lr(base_lr: float, start: float, epoch: int, warmup_epochs: int) -> float:
    """Linear warmup that is already non-zero in the first epoch.

    Epoch ``e`` (0-based) trains at ``(e + 1) / warmup_epochs`` of the way
    from ``start`` to ``base_lr``, so the last warmup epoch reaches
    ``base_lr``. (Using ``e / warmup_epochs`` makes the whole first epoch
    run at LR 0 when ``start`` is 0, wasting it.)
    """
    return start + (base_lr - start) * (epoch + 1) / warmup_epochs


class YOLOXLRScheduler(_LRScheduler):
    """YOLOX learning rate scheduler with warmup and cosine/linear decay.

    This scheduler implements the official YOLOX training strategy:
    1. Warmup phase: Linear warmup from 0 to base_lr
    2. Main phase: Cosine annealing or linear decay

    Args:
        optimizer: Wrapped optimizer
        total_epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs (default: 5)
        warmup_lr_start: Starting LR for warmup (default: 0)
        no_aug_epochs: Number of epochs without augmentation at the end (default: 15)
        min_lr_ratio: Minimum LR ratio (default: 0.05)
        scheduler_type: 'cosine' or 'linear' (default: 'cosine')
    """

    def __init__(
        self,
        optimizer,
        total_epochs: int,
        warmup_epochs: int = 5,
        warmup_lr_start: float = 0.0,
        no_aug_epochs: int = 15,
        min_lr_ratio: float = 0.05,
        scheduler_type: str = "cosine",
        last_epoch: int = -1,
    ):
        self.total_epochs = total_epochs
        self.warmup_epochs, self.no_aug_epochs = _fit_phases(
            total_epochs, warmup_epochs, no_aug_epochs
        )
        self.warmup_lr_start = warmup_lr_start
        self.min_lr_ratio = min_lr_ratio
        self.scheduler_type = scheduler_type

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for current epoch."""
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Warmup phase: linear warmup
            return [
                _warmup_lr(base_lr, self.warmup_lr_start, epoch, self.warmup_epochs)
                for base_lr in self.base_lrs
            ]

        elif epoch >= self.total_epochs - self.no_aug_epochs:
            # No augmentation phase: use minimum LR
            return [base_lr * self.min_lr_ratio for base_lr in self.base_lrs]

        else:
            # Main training phase
            progress = (epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs - self.no_aug_epochs
            )

            if self.scheduler_type == "cosine":
                # Cosine annealing
                return [
                    self.min_lr_ratio * base_lr
                    + (base_lr - self.min_lr_ratio * base_lr)
                    * (1 + math.cos(math.pi * progress))
                    / 2
                    for base_lr in self.base_lrs
                ]
            else:
                # Linear decay
                return [
                    base_lr * (1 - progress * (1 - self.min_lr_ratio))
                    for base_lr in self.base_lrs
                ]


class YOLOXWarmupLR(_LRScheduler):
    """Simple warmup scheduler for YOLOX.

    Linearly increases learning rate from warmup_lr_start to base_lr
    over warmup_epochs.

    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        warmup_lr_start: Starting LR for warmup (default: 0)
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        warmup_lr_start: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_start = warmup_lr_start
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            return [
                _warmup_lr(
                    base_lr, self.warmup_lr_start, self.last_epoch, self.warmup_epochs
                )
                for base_lr in self.base_lrs
            ]
        return self.base_lrs
