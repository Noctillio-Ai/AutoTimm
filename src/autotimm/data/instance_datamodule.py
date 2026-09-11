"""Lightning DataModule for instance segmentation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader

from autotimm.data.instance_dataset import (
    COCOInstanceDataset,
    CSVInstanceDataset,
    collate_instance_segmentation,
)
from autotimm.data.segmentation_transforms import instance_segmentation_transforms
from autotimm.data.transform_config import TransformConfig


class InstanceSegmentationDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for instance segmentation.

    Supports two modes:

    1. **COCO mode** (default) -- expects COCO-format annotations.
    2. **CSV mode** -- provide ``train_csv`` pointing to a CSV file with
       columns ``image_path,x_min,y_min,x_max,y_max,label,mask_path``.

    Args:
        data_dir: Root directory of COCO dataset
        image_size: Target image size (square)
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers. Defaults to ``os.cpu_count()``.
        augmentation_preset: Augmentation strength ('default', 'strong', 'light')
        custom_train_transforms: Optional custom training transforms
        custom_val_transforms: Optional custom validation transforms
        min_keypoints: Minimum number of keypoints for an instance to be valid
        min_area: Minimum area for an instance to be valid
        transform_config: Optional TransformConfig for unified transform configuration.
            When provided along with backbone, uses model-specific normalization.
        backbone: Optional backbone name or module for model-specific normalization.
        train_csv: Path to training CSV file (CSV mode).
        val_csv: Path to validation CSV file (CSV mode).
        test_csv: Path to test CSV file (CSV mode).
        image_dir: Root directory for resolving image/mask paths in CSV mode.
        image_column: CSV column name for image paths.
        bbox_columns: CSV column names for bbox coordinates.
        label_column: CSV column name for class labels.
        mask_column: CSV column name for mask file paths.
    """

    def __init__(
        self,
        data_dir: str | Path = ".",
        image_size: int = 640,
        batch_size: int = 4,
        num_workers: int = min(os.cpu_count() or 4, 4),
        augmentation_preset: str = "default",
        custom_train_transforms: Any = None,
        custom_val_transforms: Any = None,
        min_keypoints: int = 0,
        min_area: float = 0.0,
        transform_config: TransformConfig | None = None,
        backbone: str | nn.Module | None = None,
        train_csv: str | Path | None = None,
        val_csv: str | Path | None = None,
        test_csv: str | Path | None = None,
        image_dir: str | Path | None = None,
        image_column: str = "image_path",
        bbox_columns: list[str] | None = None,
        label_column: str = "label",
        mask_column: str = "mask_path",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.data_dir = Path(data_dir)
        self.train_csv = Path(train_csv) if train_csv else None
        self.val_csv = Path(val_csv) if val_csv else None
        self.test_csv = Path(test_csv) if test_csv else None
        self.image_dir = Path(image_dir) if image_dir else None
        self.image_column = image_column
        self.bbox_columns = bbox_columns
        self.label_column = label_column
        self.mask_column = mask_column
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation_preset = augmentation_preset
        self.custom_train_transforms = custom_train_transforms
        self.custom_val_transforms = custom_val_transforms
        self.min_keypoints = min_keypoints
        self.min_area = min_area
        self.transform_config = transform_config
        self.backbone = backbone

        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None):
        """Setup datasets for each stage.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Get transforms - TransformConfig takes precedence
        if self.transform_config is not None and self.backbone is not None:
            from autotimm.data.timm_transforms import get_transforms_from_backbone

            train_transforms = get_transforms_from_backbone(
                backbone=self.backbone,
                transform_config=self.transform_config,
                is_train=True,
                task="instance_segmentation",
            )
            val_transforms = get_transforms_from_backbone(
                backbone=self.backbone,
                transform_config=self.transform_config,
                is_train=False,
                task="instance_segmentation",
            )
        elif self.custom_train_transforms is not None:
            train_transforms = self.custom_train_transforms
            if self.custom_val_transforms is not None:
                val_transforms = self.custom_val_transforms
            else:
                val_transforms = instance_segmentation_transforms(
                    image_size=self.image_size,
                    train=False,
                )
        else:
            train_transforms = instance_segmentation_transforms(
                image_size=self.image_size,
                train=True,
            )
            if self.custom_val_transforms is not None:
                val_transforms = self.custom_val_transforms
            else:
                val_transforms = instance_segmentation_transforms(
                    image_size=self.image_size,
                    train=False,
                )

        # Setup datasets based on stage
        if self.train_csv is not None:
            self._setup_csv(stage, train_transforms, val_transforms)
        else:
            self._setup_coco(stage, train_transforms, val_transforms)

    def _setup_csv(self, stage, train_transforms, val_transforms):
        img_dir = self.image_dir or self.train_csv.parent

        if stage == "fit" or stage is None:
            self.train_dataset = CSVInstanceDataset(
                csv_path=self.train_csv,
                image_dir=img_dir,
                image_column=self.image_column,
                bbox_columns=self.bbox_columns,
                label_column=self.label_column,
                mask_column=self.mask_column,
                transform=train_transforms,
            )

            if self.val_csv is not None:
                self.val_dataset = CSVInstanceDataset(
                    csv_path=self.val_csv,
                    image_dir=img_dir,
                    image_column=self.image_column,
                    bbox_columns=self.bbox_columns,
                    label_column=self.label_column,
                    mask_column=self.mask_column,
                    transform=val_transforms,
                )

        if stage == "validate" and self.val_csv is not None:
            self.val_dataset = CSVInstanceDataset(
                csv_path=self.val_csv,
                image_dir=img_dir,
                image_column=self.image_column,
                bbox_columns=self.bbox_columns,
                label_column=self.label_column,
                mask_column=self.mask_column,
                transform=val_transforms,
            )

        if stage == "test" and self.test_csv is not None:
            self.test_dataset = CSVInstanceDataset(
                csv_path=self.test_csv,
                image_dir=img_dir,
                image_column=self.image_column,
                bbox_columns=self.bbox_columns,
                label_column=self.label_column,
                mask_column=self.mask_column,
                transform=val_transforms,
            )

    def _setup_coco(self, stage, train_transforms, val_transforms):
        if stage == "fit" or stage is None:
            self.train_dataset = COCOInstanceDataset(
                data_dir=self.data_dir,
                split="train",
                transforms=train_transforms,
                min_keypoints=self.min_keypoints,
                min_area=self.min_area,
            )

            try:
                self.val_dataset = COCOInstanceDataset(
                    data_dir=self.data_dir,
                    split="val",
                    transforms=val_transforms,
                    min_keypoints=self.min_keypoints,
                    min_area=self.min_area,
                )
            except FileNotFoundError:
                self.val_dataset = None

        if stage == "validate":
            self.val_dataset = COCOInstanceDataset(
                data_dir=self.data_dir,
                split="val",
                transforms=val_transforms,
                min_keypoints=self.min_keypoints,
                min_area=self.min_area,
            )

        if stage == "test":
            self.test_dataset = COCOInstanceDataset(
                data_dir=self.data_dir,
                split="test",
                transforms=val_transforms,
                min_keypoints=self.min_keypoints,
                min_area=self.min_area,
            )

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_instance_segmentation,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader | None:
        """Get validation dataloader."""
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_instance_segmentation,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_instance_segmentation,
            pin_memory=True,
        )
