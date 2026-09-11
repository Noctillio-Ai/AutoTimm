"""Lightning data module for multi-label image classification."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

from autotimm.data.transform_config import TransformConfig
from autotimm.data.transforms import (
    albu_default_eval_transforms,
    albu_default_train_transforms,
    default_eval_transforms,
    default_train_transforms,
    get_train_transforms,
)


class MultiLabelImageDataModule(pl.LightningDataModule):
    """Lightning data module for multi-label image classification.

    Reads CSV files where each row contains an image path and binary label
    columns. See :class:`~autotimm.data.dataset.MultiLabelImageDataset`
    for the expected CSV format.

    Parameters:
        train_csv: Path to training CSV file.
        image_dir: Root directory for image paths.
        val_csv: Optional path to validation CSV. If ``None``, splits
            from ``train_csv`` using ``val_split``.
        test_csv: Optional path to test CSV.
        label_columns: List of label column names. If ``None``,
            auto-detected from CSV (all columns except the image column).
        image_column: Column name for image paths. Default: first column.
        image_size: Target image size (square).
        batch_size: Batch size for all dataloaders.
        num_workers: Number of data-loading workers. Defaults to ``os.cpu_count()``.
        val_split: Fraction for validation split when ``val_csv`` is ``None``.
        train_transforms: Custom training transforms.
        eval_transforms: Custom eval transforms.
        augmentation_preset: Built-in preset name.
        transform_backend: ``"torchvision"`` or ``"albumentations"``.
        transform_config: Optional :class:`TransformConfig` for model-specific
            normalization.
        backbone: Backbone name or module for timm transform resolution.
        pin_memory: Pin memory for GPU transfer.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Number of batches prefetched per worker.
    """

    def __init__(
        self,
        train_csv: str | Path,
        image_dir: str | Path = ".",
        val_csv: str | Path | None = None,
        test_csv: str | Path | None = None,
        label_columns: list[str] | None = None,
        image_column: str | None = None,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = min(os.cpu_count() or 4, 4),
        val_split: float = 0.1,
        train_transforms: Callable | None = None,
        eval_transforms: Callable | None = None,
        augmentation_preset: str | None = None,
        transform_backend: str = "torchvision",
        transform_config: TransformConfig | None = None,
        backbone: str | nn.Module | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.train_csv = Path(train_csv)
        self.image_dir = Path(image_dir)
        self.val_csv = Path(val_csv) if val_csv is not None else None
        self.test_csv = Path(test_csv) if test_csv is not None else None
        self.label_columns = label_columns
        self.image_column = image_column
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor

        if transform_backend not in ("torchvision", "albumentations"):
            raise ValueError(
                f"Unknown transform_backend '{transform_backend}'. "
                f"Choose from: torchvision, albumentations."
            )
        self.transform_backend = transform_backend
        self.transform_config = transform_config
        self.backbone = backbone

        # Resolve transforms - TransformConfig takes precedence
        if transform_config is not None and backbone is not None:
            from autotimm.data.timm_transforms import get_transforms_from_backbone

            self.train_transforms = get_transforms_from_backbone(
                backbone=backbone,
                transform_config=transform_config,
                is_train=True,
                task="classification",
            )
            self.eval_transforms = get_transforms_from_backbone(
                backbone=backbone,
                transform_config=transform_config,
                is_train=False,
                task="classification",
            )
        elif train_transforms is not None:
            self.train_transforms = train_transforms
        elif augmentation_preset is not None:
            self.train_transforms = get_train_transforms(
                augmentation_preset,
                backend=transform_backend,
                image_size=image_size,
            )
        else:
            self.train_transforms = self._default_train_transforms()

        if eval_transforms is not None:
            self.eval_transforms = eval_transforms
        elif not (transform_config is not None and backbone is not None):
            self.eval_transforms = self._default_eval_transforms()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._num_labels: int | None = None
        self._label_names: list[str] | None = None

    @property
    def num_labels(self) -> int | None:
        """Number of labels. Available after ``setup()``."""
        return self._num_labels

    @property
    def label_names(self) -> list[str] | None:
        """Label column names. Available after ``setup()``."""
        return self._label_names

    def _default_train_transforms(self) -> Callable:
        if self.transform_backend == "albumentations":
            return albu_default_train_transforms(self.image_size)
        return default_train_transforms(self.image_size)

    def _default_eval_transforms(self) -> Callable:
        if self.transform_backend == "albumentations":
            return albu_default_eval_transforms(self.image_size)
        return default_eval_transforms(self.image_size)

    def setup(self, stage: str | None = None) -> None:
        from autotimm.data.dataset import MultiLabelImageDataset

        use_albu = self.transform_backend == "albumentations"

        if stage in ("fit", None):
            full_train = MultiLabelImageDataset(
                csv_path=self.train_csv,
                image_dir=self.image_dir,
                label_columns=self.label_columns,
                image_column=self.image_column,
                transform=self.train_transforms,
                use_albumentations=use_albu,
            )
            self._num_labels = full_train.num_labels
            self._label_names = full_train.label_names

            if self.val_csv is not None:
                self.train_dataset = full_train
                self.val_dataset = MultiLabelImageDataset(
                    csv_path=self.val_csv,
                    image_dir=self.image_dir,
                    label_columns=self.label_columns,
                    image_column=self.image_column,
                    transform=self.eval_transforms,
                    use_albumentations=use_albu,
                )
            else:
                # Separate instance with eval transforms so the validation
                # split is not augmented.
                full_eval = MultiLabelImageDataset(
                    csv_path=self.train_csv,
                    image_dir=self.image_dir,
                    label_columns=self.label_columns,
                    image_column=self.image_column,
                    transform=self.eval_transforms,
                    use_albumentations=use_albu,
                )
                n_val = int(len(full_train) * self.val_split)
                n_train = len(full_train) - n_val
                self.train_dataset, val_subset = random_split(
                    full_train, [n_train, n_val]
                )
                self.val_dataset = Subset(full_eval, val_subset.indices)

        if stage in ("test", None) and self.test_csv is not None:
            self.test_dataset = MultiLabelImageDataset(
                csv_path=self.test_csv,
                image_dir=self.image_dir,
                label_columns=self.label_columns,
                image_column=self.image_column,
                transform=self.eval_transforms,
                use_albumentations=use_albu,
            )

    def _loader_kwargs(self) -> dict:
        kwargs: dict = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }
        if self.prefetch_factor is not None and self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError(
                "No test split found. Provide test_csv to enable testing."
            )
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            **self._loader_kwargs(),
        )
