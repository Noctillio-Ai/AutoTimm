"""Image data module for folder-based and torchvision built-in datasets."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Callable

import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from torchvision import datasets

from autotimm.data.transform_config import TransformConfig
from autotimm.data.transforms import (
    albu_default_eval_transforms,
    albu_default_train_transforms,
    default_eval_transforms,
    default_train_transforms,
    get_train_transforms,
)


class ImageDataModule(pl.LightningDataModule):
    """Lightning data module for image classification.

    Supports three modes:

    1. **Folder mode** -- point ``data_dir`` at a directory with ``train/``,
       ``val/``, and optionally ``test/`` subdirectories, each containing one
       sub-folder per class (ImageFolder layout).
    2. **Built-in dataset mode** -- set ``dataset_name`` to a torchvision
       dataset (``"CIFAR10"``, ``"CIFAR100"``, ``"FashionMNIST"``, ``"MNIST"``)
       and ``data_dir`` to the download root.
    3. **CSV mode** -- provide ``train_csv`` pointing to a CSV file with
       ``image_path,label`` columns. Optionally provide ``val_csv`` and
       ``test_csv`` for separate splits.

    Parameters:
        data_dir: Root directory for image data or download root.
        dataset_name: Optional name of a torchvision dataset class.
        train_csv: Path to training CSV file (CSV mode).
        val_csv: Path to validation CSV file (CSV mode).
        test_csv: Path to test CSV file (CSV mode).
        image_dir: Root directory for resolving image paths in CSV mode.
            Defaults to the parent directory of ``train_csv``.
        image_column: Name of the CSV column containing image paths.
        label_column: Name of the CSV column containing class labels.
        image_size: Target image size (square).
        batch_size: Batch size for all dataloaders.
        num_workers: Number of data-loading workers. Defaults to ``os.cpu_count()``.
        val_split: Fraction of training data used for validation when
            no explicit val set exists.
        train_transforms: Custom training transforms; defaults used if None.
            Mutually exclusive with ``augmentation_preset``.
        eval_transforms: Custom eval transforms; defaults used if None.
        augmentation_preset: Name of a built-in augmentation preset.
            For ``torchvision``: ``"default"``, ``"autoaugment"``,
            ``"randaugment"``, ``"trivialaugment"``.
            For ``albumentations``: ``"default"``, ``"strong"``.
            Ignored when ``train_transforms`` is provided.
        transform_backend: ``"torchvision"`` (PIL-based) or
            ``"albumentations"`` (OpenCV-based). Defaults to
            ``"torchvision"``. When ``"albumentations"`` is selected,
            folder-mode datasets load images with OpenCV and built-in
            datasets convert PIL images to numpy for the augmentation
            pipeline.
        transform_config: Optional :class:`TransformConfig` for unified transform
            configuration. When provided along with ``backbone``, uses model-specific
            normalization from timm. Takes precedence over individual transform args.
        backbone: Optional backbone name or module. Used with ``transform_config``
            to resolve model-specific normalization (mean, std, input_size).
        pin_memory: Pin memory for GPU transfer.
        persistent_workers: Keep worker processes alive between epochs.
            Reduces overhead when ``num_workers > 0``.
        prefetch_factor: Number of batches prefetched per worker.
        balanced_sampling: Use ``WeightedRandomSampler`` to counter
            class imbalance in the training set.
    """

    BUILTIN_DATASETS: dict[str, type] = {
        "CIFAR10": datasets.CIFAR10,
        "CIFAR100": datasets.CIFAR100,
        "FashionMNIST": datasets.FashionMNIST,
        "MNIST": datasets.MNIST,
    }

    def __init__(
        self,
        data_dir: str | Path = "./data",
        dataset_name: str | None = None,
        train_csv: str | Path | None = None,
        val_csv: str | Path | None = None,
        test_csv: str | Path | None = None,
        image_dir: str | Path | None = None,
        image_column: str | None = None,
        label_column: str | None = None,
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
        balanced_sampling: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.train_csv = Path(train_csv) if train_csv else None
        self.val_csv = Path(val_csv) if val_csv else None
        self.test_csv = Path(test_csv) if test_csv else None
        self.image_dir = Path(image_dir) if image_dir else None
        self.image_column = image_column
        self.label_column = label_column
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor
        self.balanced_sampling = balanced_sampling

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
        self.num_classes: int | None = None
        self.class_names: list[str] | None = None
        self._train_targets: list[int] | None = None

    def _default_train_transforms(self) -> Callable:
        if self.transform_backend == "albumentations":
            return albu_default_train_transforms(self.image_size)
        return default_train_transforms(self.image_size)

    def _default_eval_transforms(self) -> Callable:
        if self.transform_backend == "albumentations":
            return albu_default_eval_transforms(self.image_size)
        return default_eval_transforms(self.image_size)

    def prepare_data(self) -> None:
        if self.dataset_name and self.dataset_name in self.BUILTIN_DATASETS:
            cls = self.BUILTIN_DATASETS[self.dataset_name]
            cls(str(self.data_dir), train=True, download=True)
            cls(str(self.data_dir), train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        if self.train_csv is not None:
            self._setup_csv(stage)
        elif self.dataset_name and self.dataset_name in self.BUILTIN_DATASETS:
            self._setup_builtin(stage)
        elif self.transform_backend == "albumentations":
            self._setup_folder_cv2(stage)
        else:
            self._setup_folder(stage)

    def _setup_builtin(self, stage: str | None) -> None:
        cls = self.BUILTIN_DATASETS[self.dataset_name]

        if self.transform_backend == "albumentations":
            wrapper_train = _AlbumentationsBuiltinWrapper(self.train_transforms)
            wrapper_eval = _AlbumentationsBuiltinWrapper(self.eval_transforms)
        else:
            wrapper_train = self.train_transforms
            wrapper_eval = self.eval_transforms

        if stage in ("fit", None):
            full_train = cls(str(self.data_dir), train=True, transform=wrapper_train)
            # Separate instance with eval transforms so the validation split
            # is not augmented.
            full_eval = cls(str(self.data_dir), train=True, transform=wrapper_eval)
            n_val = int(len(full_train) * self.val_split)
            n_train = len(full_train) - n_val
            self.train_dataset, val_subset = random_split(
                full_train, [n_train, n_val]
            )
            self.val_dataset = Subset(full_eval, val_subset.indices)
            self.num_classes = (
                len(full_train.classes) if hasattr(full_train, "classes") else 10
            )
            self.class_names = (
                list(full_train.classes) if hasattr(full_train, "classes") else None
            )
            # int() so tensor targets (e.g. MNIST) hash by value, not identity
            self._train_targets = [
                int(full_train.targets[i]) for i in self.train_dataset.indices
            ]
        if stage in ("test", None):
            self.test_dataset = cls(
                str(self.data_dir), train=False, transform=wrapper_eval
            )

    def _setup_folder(self, stage: str | None) -> None:
        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"
        test_dir = self.data_dir / "test"

        if stage in ("fit", None):
            full_train = datasets.ImageFolder(
                str(train_dir), transform=self.train_transforms
            )
            self.train_dataset = full_train
            self.num_classes = len(full_train.classes)
            self.class_names = list(full_train.classes)
            all_targets = [s[1] for s in full_train.samples]
            self._train_targets = all_targets

            if val_dir.exists():
                self.val_dataset = datasets.ImageFolder(
                    str(val_dir), transform=self.eval_transforms
                )
            else:
                # Separate instance with eval transforms so the validation
                # split is not augmented.
                full_eval = datasets.ImageFolder(
                    str(train_dir), transform=self.eval_transforms
                )
                n_val = int(len(full_train) * self.val_split)
                n_train = len(full_train) - n_val
                self.train_dataset, val_subset = random_split(
                    full_train, [n_train, n_val]
                )
                self.val_dataset = Subset(full_eval, val_subset.indices)
                # Recompute targets for the subset so balanced sampling
                # weights align with the sampled indices.
                self._train_targets = [
                    all_targets[i] for i in self.train_dataset.indices
                ]
        if stage in ("test", None) and test_dir.exists():
            self.test_dataset = datasets.ImageFolder(
                str(test_dir), transform=self.eval_transforms
            )

    def _setup_folder_cv2(self, stage: str | None) -> None:
        from autotimm.data.dataset import ImageFolderCV2

        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"
        test_dir = self.data_dir / "test"

        if stage in ("fit", None):
            full_train = ImageFolderCV2(
                str(train_dir), transform=self.train_transforms
            )
            self.train_dataset = full_train
            self.num_classes = len(full_train.classes)
            self.class_names = list(full_train.classes)
            all_targets = [s[1] for s in full_train.samples]
            self._train_targets = all_targets

            if val_dir.exists():
                self.val_dataset = ImageFolderCV2(
                    str(val_dir), transform=self.eval_transforms
                )
            else:
                # Separate instance with eval transforms so the validation
                # split is not augmented.
                full_eval = ImageFolderCV2(
                    str(train_dir), transform=self.eval_transforms
                )
                n_val = int(len(full_train) * self.val_split)
                n_train = len(full_train) - n_val
                self.train_dataset, val_subset = random_split(
                    full_train, [n_train, n_val]
                )
                self.val_dataset = Subset(full_eval, val_subset.indices)
                # Recompute targets for the subset so balanced sampling
                # weights align with the sampled indices.
                self._train_targets = [
                    all_targets[i] for i in self.train_dataset.indices
                ]
        if stage in ("test", None) and test_dir.exists():
            self.test_dataset = ImageFolderCV2(
                str(test_dir), transform=self.eval_transforms
            )

    def _setup_csv(self, stage: str | None) -> None:
        from autotimm.data.dataset import CSVImageDataset

        use_albu = self.transform_backend == "albumentations"
        # Default image_dir to parent of train_csv
        img_dir = self.image_dir or self.train_csv.parent

        if stage in ("fit", None):
            full_train = CSVImageDataset(
                csv_path=self.train_csv,
                image_dir=img_dir,
                image_column=self.image_column,
                label_column=self.label_column,
                transform=self.train_transforms,
                use_albumentations=use_albu,
            )
            self.train_dataset = full_train
            self.num_classes = full_train.num_classes
            self.class_names = list(full_train.classes)
            all_targets = [s[1] for s in full_train.samples]
            self._train_targets = all_targets

            if self.val_csv is not None:
                self.val_dataset = CSVImageDataset(
                    csv_path=self.val_csv,
                    image_dir=img_dir,
                    image_column=self.image_column,
                    label_column=self.label_column,
                    transform=self.eval_transforms,
                    use_albumentations=use_albu,
                )
            else:
                # Separate instance with eval transforms so the validation
                # split is not augmented.
                full_eval = CSVImageDataset(
                    csv_path=self.train_csv,
                    image_dir=img_dir,
                    image_column=self.image_column,
                    label_column=self.label_column,
                    transform=self.eval_transforms,
                    use_albumentations=use_albu,
                )
                n_val = int(len(full_train) * self.val_split)
                n_train = len(full_train) - n_val
                self.train_dataset, val_subset = random_split(
                    full_train, [n_train, n_val]
                )
                self.val_dataset = Subset(full_eval, val_subset.indices)
                # Recompute targets for the subset so balanced sampling
                # weights align with the sampled indices.
                self._train_targets = [
                    all_targets[i] for i in self.train_dataset.indices
                ]

        if stage in ("test", None) and self.test_csv is not None:
            self.test_dataset = CSVImageDataset(
                csv_path=self.test_csv,
                image_dir=img_dir,
                image_column=self.image_column,
                label_column=self.label_column,
                transform=self.eval_transforms,
                use_albumentations=use_albu,
            )

    def _make_sampler(self) -> WeightedRandomSampler | None:
        if not self.balanced_sampling or self._train_targets is None:
            return None

        counts = Counter(self._train_targets)
        weight_per_class = {cls: 1.0 / cnt for cls, cnt in counts.items()}
        sample_weights = [weight_per_class[t] for t in self._train_targets]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
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
        sampler = self._make_sampler()
        return DataLoader(
            self.train_dataset,
            shuffle=sampler is None,
            sampler=sampler,
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
                "No test split found. Provide a 'test/' directory or use a built-in dataset."
            )
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            **self._loader_kwargs(),
        )


class _AlbumentationsBuiltinWrapper:
    """Wraps an albumentations ``Compose`` so it works as a torchvision transform.

    Built-in torchvision datasets yield PIL images. This wrapper converts
    them to numpy arrays (RGB, uint8) before applying the albumentations
    pipeline, so users can pass ``transform_backend="albumentations"``
    with built-in datasets seamlessly.
    """

    def __init__(self, albu_transform):
        self.albu_transform = albu_transform

    def __call__(self, pil_image):
        import numpy as np

        image = np.array(pil_image)
        # Grayscale → 3-channel
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return self.albu_transform(image=image)["image"]
