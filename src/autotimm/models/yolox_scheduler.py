"""YOLOX Learning Rate Scheduler with Warmup.

Based on the official YOLOX implementation:
https://github.com/Megvii-BaseDetection/YOLOX
"""

import math

from torch.optim.lr_scheduler import _LRScheduler


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
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_start = warmup_lr_start
        self.no_aug_epochs = no_aug_epochs
        self.min_lr_ratio = min_lr_ratio
        self.scheduler_type = scheduler_type

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for current epoch."""
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Warmup phase: linear warmup
            return [
                self.warmup_lr_start
                + (base_lr - self.warmup_lr_start) * epoch / self.warmup_epochs
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
                self.warmup_lr_start
                + (base_lr - self.warmup_lr_start)
                * self.last_epoch
                / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        return self.base_lrs
