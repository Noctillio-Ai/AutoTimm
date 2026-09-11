"""Detection data module for COCO-format datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader

from autotimm.data.detection_dataset import (
    COCODetectionDataset,
    CSVDetectionDataset,
    detection_collate_fn,
)
from autotimm.data.detection_transforms import (
    detection_eval_transforms,
    get_detection_transforms,
)
from autotimm.data.transform_config import TransformConfig


class DetectionDataModule(pl.LightningDataModule):
    """Lightning data module for object detection.

    Supports two modes:

    1. **COCO mode** (default) -- expects COCO-style directory structure::

        data_dir/
          train2017/           # Training images
          val2017/             # Validation images
          annotations/
            instances_train2017.json
            instances_val2017.json

    2. **CSV mode** -- provide ``train_csv`` pointing to a CSV file with
       columns ``image_path,x_min,y_min,x_max,y_max,label``.

    Parameters:
        data_dir: Root directory containing images and annotations.
        train_images_dir: Path to training images. Defaults to data_dir/train2017.
        val_images_dir: Path to validation images. Defaults to data_dir/val2017.
        train_ann_file: Path to train annotations. Defaults to
            data_dir/annotations/instances_train2017.json.
        val_ann_file: Path to val annotations. Defaults to
            data_dir/annotations/instances_val2017.json.
        test_images_dir: Optional path to test images.
        test_ann_file: Optional path to test annotations.
        train_csv: Path to training CSV file (CSV mode).
        val_csv: Path to validation CSV file (CSV mode).
        test_csv: Path to test CSV file (CSV mode).
        image_dir: Root directory for resolving image paths in CSV mode.
        image_column: CSV column name for image paths.
        bbox_columns: CSV column names for bbox coordinates.
        label_column: CSV column name for class labels.
        image_size: Target image size (square).
        batch_size: Batch size for all dataloaders.
        num_workers: Number of data-loading workers. Defaults to ``os.cpu_count()``.
        train_transforms: Custom training transforms. Must include bbox_params.
        eval_transforms: Custom eval transforms. Must include bbox_params.
        augmentation_preset: Preset name (``"default"``, ``"strong"``).
            Ignored when train_transforms is provided.
        transform_config: Optional :class:`TransformConfig` for unified transform
            configuration. When provided along with ``backbone``, uses model-specific
            normalization from timm. Takes precedence over individual transform args.
        backbone: Optional backbone name or module. Used with ``transform_config``
            to resolve model-specific normalization (mean, std, input_size).
        pin_memory: Pin memory for GPU transfer.
        persistent_workers: Keep worker processes alive between epochs.
        prefetch_factor: Number of batches prefetched per worker.
        min_bbox_area: Minimum bbox area to include in training.
        class_ids: Optional list of class IDs to filter.
    """

    def __init__(
        self,
        data_dir: str | Path = "./coco",
        train_images_dir: str | Path | None = None,
        val_images_dir: str | Path | None = None,
        train_ann_file: str | Path | None = None,
        val_ann_file: str | Path | None = None,
        test_images_dir: str | Path | None = None,
        test_ann_file: str | Path | None = None,
        train_csv: str | Path | None = None,
        val_csv: str | Path | None = None,
        test_csv: str | Path | None = None,
        image_dir: str | Path | None = None,
        image_column: str = "image_path",
        bbox_columns: list[str] | None = None,
        label_column: str = "label",
        image_size: int = 640,
        batch_size: int = 16,
        num_workers: int = min(os.cpu_count() or 4, 4),
        train_transforms: Callable | None = None,
        eval_transforms: Callable | None = None,
        augmentation_preset: str = "default",
        transform_config: TransformConfig | None = None,
        backbone: str | nn.Module | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        min_bbox_area: float = 0.0,
        class_ids: list[int] | None = None,
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
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor
        self.min_bbox_area = min_bbox_area
        self.class_ids = class_ids

        # Set default paths
        self.train_images_dir = (
            Path(train_images_dir) if train_images_dir else self.data_dir / "train2017"
        )
        self.val_images_dir = (
            Path(val_images_dir) if val_images_dir else self.data_dir / "val2017"
        )
        self.train_ann_file = (
            Path(train_ann_file)
            if train_ann_file
            else self.data_dir / "annotations" / "instances_train2017.json"
        )
        self.val_ann_file = (
            Path(val_ann_file)
            if val_ann_file
            else self.data_dir / "annotations" / "instances_val2017.json"
        )
        self.test_images_dir = Path(test_images_dir) if test_images_dir else None
        self.test_ann_file = Path(test_ann_file) if test_ann_file else None
        self.transform_config = transform_config
        self.backbone = backbone

        # Resolve transforms - TransformConfig takes precedence
        if transform_config is not None and backbone is not None:
            from autotimm.data.timm_transforms import get_transforms_from_backbone

            self.train_transforms = get_transforms_from_backbone(
                backbone=backbone,
                transform_config=transform_config,
                is_train=True,
                task="detection",
            )
            self.eval_transforms = get_transforms_from_backbone(
                backbone=backbone,
                transform_config=transform_config,
                is_train=False,
                task="detection",
            )
        elif train_transforms is not None:
            self.train_transforms = train_transforms
        else:
            self.train_transforms = get_detection_transforms(
                preset=augmentation_preset,
                image_size=image_size,
                is_train=True,
            )

        if eval_transforms is not None:
            self.eval_transforms = eval_transforms
        elif not (transform_config is not None and backbone is not None):
            self.eval_transforms = detection_eval_transforms(image_size=image_size)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes: int | None = None
        self.class_names: list[str] | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_csv is not None:
            self._setup_csv(stage)
        else:
            self._setup_coco(stage)

    def _setup_csv(self, stage: str | None) -> None:
        img_dir = self.image_dir or self.train_csv.parent

        if stage in ("fit", None):
            self.train_dataset = CSVDetectionDataset(
                csv_path=self.train_csv,
                image_dir=img_dir,
                image_column=self.image_column,
                bbox_columns=self.bbox_columns,
                label_column=self.label_column,
                transform=self.train_transforms,
                min_bbox_area=self.min_bbox_area,
            )
            self.num_classes = self.train_dataset.num_classes
            self.class_names = self.train_dataset.class_names

            if self.val_csv is not None:
                self.val_dataset = CSVDetectionDataset(
                    csv_path=self.val_csv,
                    image_dir=img_dir,
                    image_column=self.image_column,
                    bbox_columns=self.bbox_columns,
                    label_column=self.label_column,
                    transform=self.eval_transforms,
                    min_bbox_area=0.0,
                )

        if stage in ("test", None) and self.test_csv is not None:
            self.test_dataset = CSVDetectionDataset(
                csv_path=self.test_csv,
                image_dir=img_dir,
                image_column=self.image_column,
                bbox_columns=self.bbox_columns,
                label_column=self.label_column,
                transform=self.eval_transforms,
                min_bbox_area=0.0,
            )
            if self.num_classes is None:
                self.num_classes = self.test_dataset.num_classes
                self.class_names = self.test_dataset.class_names

    def _setup_coco(self, stage: str | None) -> None:
        if stage in ("fit", None):
            self.train_dataset = COCODetectionDataset(
                images_dir=self.train_images_dir,
                annotations_file=self.train_ann_file,
                transform=self.train_transforms,
                min_bbox_area=self.min_bbox_area,
                class_ids=self.class_ids,
            )
            self.val_dataset = COCODetectionDataset(
                images_dir=self.val_images_dir,
                annotations_file=self.val_ann_file,
                transform=self.eval_transforms,
                min_bbox_area=0.0,  # Keep all boxes for evaluation
                class_ids=self.class_ids,
            )
            self.num_classes = self.train_dataset.num_classes
            self.class_names = self.train_dataset.class_names

        if stage in ("test", None) and self.test_images_dir and self.test_ann_file:
            self.test_dataset = COCODetectionDataset(
                images_dir=self.test_images_dir,
                annotations_file=self.test_ann_file,
                transform=self.eval_transforms,
                min_bbox_area=0.0,
                class_ids=self.class_ids,
            )
            if self.num_classes is None:
                self.num_classes = self.test_dataset.num_classes
                self.class_names = self.test_dataset.class_names

        if stage == "validate" and self.val_dataset is None:
            self.val_dataset = COCODetectionDataset(
                images_dir=self.val_images_dir,
                annotations_file=self.val_ann_file,
                transform=self.eval_transforms,
                min_bbox_area=0.0,
                class_ids=self.class_ids,
            )
            self.num_classes = self.val_dataset.num_classes
            self.class_names = self.val_dataset.class_names

    def _loader_kwargs(self) -> dict:
        kwargs: dict = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": detection_collate_fn,
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

    def val_dataloader(self) -> DataLoader | None:
        # CSV mode without val_csv has no validation split; returning None
        # lets Lightning skip the validation loop instead of crashing.
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError(
                "No test dataset configured. Provide test_images_dir and test_ann_file."
            )
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            **self._loader_kwargs(),
        )
