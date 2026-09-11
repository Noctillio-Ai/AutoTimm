"""Image datasets for classification tasks."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class ImageFolderCV2(Dataset):
    """ImageFolder-style dataset that loads images with OpenCV.

    Expects the same directory layout as ``torchvision.datasets.ImageFolder``::

        root/
          class_a/
            img1.jpg
          class_b/
            img2.jpg

    Images are loaded as RGB numpy arrays (H, W, C) with dtype ``uint8``,
    which is the format expected by albumentations transforms.

    Parameters:
        root: Root directory containing class sub-folders.
        transform: Callable applied to the image array (e.g. an
            albumentations ``Compose``). Must return a dict with
            an ``"image"`` key containing a torch Tensor.
        extensions: Tuple of valid file extensions.
    """

    VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        extensions: tuple[str, ...] | None = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.extensions = extensions or self.VALID_EXTENSIONS

        self.classes: list[str] = sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        )
        self.class_to_idx: dict[str, int] = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        self.samples: list[tuple[str, int]] = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            cls_idx = self.class_to_idx[cls_name]
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append((str(img_path), cls_idx))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No images found in '{self.root}'. Expected sub-folders "
                f"with image files ({', '.join(self.extensions)})."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        import cv2

        path, target = self.samples[index]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to load image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, target


class CSVImageDataset(Dataset):
    """Dataset for single-label classification from CSV.

    CSV format::

        image_path,label
        img001.jpg,cat
        img002.jpg,dog

    Parameters:
        csv_path: Path to CSV file.
        image_dir: Root directory for resolving image paths.
        image_column: Name of the column containing image paths.
            If ``None``, uses the first column.
        label_column: Name of the column containing class labels.
            If ``None``, uses the second column.
        transform: Transform to apply to images. Supports both
            torchvision transforms (PIL input) and albumentations
            transforms (numpy input with ``image`` key).
        use_albumentations: If ``True``, load images with OpenCV and
            pass as numpy arrays to an albumentations transform.
            Default is ``False`` (PIL + torchvision).
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path = ".",
        image_column: str | None = None,
        label_column: str | None = None,
        transform: Callable | None = None,
        use_albumentations: bool = False,
    ):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.use_albumentations = use_albumentations

        # Parse CSV
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            if not fieldnames:
                raise ValueError(f"CSV file has no columns: {self.csv_path}")

            # Determine image column
            self._image_column = image_column or fieldnames[0]
            if self._image_column not in fieldnames:
                raise ValueError(
                    f"Image column '{self._image_column}' not found in CSV. "
                    f"Available columns: {fieldnames}"
                )

            # Determine label column
            if label_column is not None:
                if label_column not in fieldnames:
                    raise ValueError(
                        f"Label column '{label_column}' not found in CSV. "
                        f"Available columns: {fieldnames}"
                    )
                self._label_column = label_column
            else:
                non_image = [c for c in fieldnames if c != self._image_column]
                if not non_image:
                    raise ValueError(
                        "No label column found. Provide label_column explicitly "
                        "or ensure the CSV has a column beyond the image column."
                    )
                self._label_column = non_image[0]

            # Read rows
            self._image_paths: list[str] = []
            self._labels_raw: list[str] = []
            for row in reader:
                self._image_paths.append(row[self._image_column])
                self._labels_raw.append(row[self._label_column])

        if len(self._image_paths) == 0:
            raise ValueError(f"CSV file has no data rows: {self.csv_path}")

        # Build class mapping from unique label values
        unique_labels = sorted(set(self._labels_raw))
        self.classes: list[str] = unique_labels
        self.class_to_idx: dict[str, int] = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        # Convert string labels to indices
        self.samples: list[tuple[str, int]] = [
            (img, self.class_to_idx[lbl])
            for img, lbl in zip(self._image_paths, self._labels_raw)
        ]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        rel_path, target = self.samples[index]
        img_path = self.image_dir / rel_path

        if self.use_albumentations:
            import cv2

            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]
        else:
            from PIL import Image

            image = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        return image, target


class MultiLabelImageDataset(Dataset):
    """Dataset for multi-label classification from CSV.

    CSV format::

        image_path,label_0,label_1,...,label_N
        img1.jpg,1,0,1,...,0
        img2.jpg,0,1,0,...,1

    Parameters:
        csv_path: Path to CSV file.
        image_dir: Root directory for resolving image paths.
        label_columns: List of column names to use as labels.
            If ``None``, uses all columns except the image column.
        image_column: Name of the column containing image paths.
            If ``None``, uses the first column.
        transform: Transform to apply to images. Supports both
            torchvision transforms (PIL input) and albumentations
            transforms (numpy input with ``image`` key).
        use_albumentations: If ``True``, load images with OpenCV and
            pass as numpy arrays to an albumentations transform.
            Default is ``False`` (PIL + torchvision).
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path = ".",
        label_columns: list[str] | None = None,
        image_column: str | None = None,
        transform: Callable | None = None,
        use_albumentations: bool = False,
    ):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.use_albumentations = use_albumentations

        # Parse CSV
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            if not fieldnames:
                raise ValueError(f"CSV file has no columns: {self.csv_path}")

            # Determine image column
            self._image_column = image_column or fieldnames[0]
            if self._image_column not in fieldnames:
                raise ValueError(
                    f"Image column '{self._image_column}' not found in CSV. "
                    f"Available columns: {fieldnames}"
                )

            # Determine label columns
            if label_columns is not None:
                for col in label_columns:
                    if col not in fieldnames:
                        raise ValueError(
                            f"Label column '{col}' not found in CSV. "
                            f"Available columns: {fieldnames}"
                        )
                self._label_columns = label_columns
            else:
                self._label_columns = [c for c in fieldnames if c != self._image_column]

            if not self._label_columns:
                raise ValueError(
                    "No label columns found. Provide label_columns explicitly "
                    "or ensure the CSV has columns beyond the image column."
                )

            # Read rows
            self._image_paths: list[str] = []
            self._labels: list[list[float]] = []
            for row in reader:
                self._image_paths.append(row[self._image_column])
                self._labels.append([float(row[col]) for col in self._label_columns])

        if len(self._image_paths) == 0:
            raise ValueError(f"CSV file has no data rows: {self.csv_path}")

    @property
    def num_labels(self) -> int:
        """Number of label columns."""
        return len(self._label_columns)

    @property
    def label_names(self) -> list[str]:
        """Names of the label columns."""
        return list(self._label_columns)

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> tuple[Any, torch.Tensor]:
        img_path = self.image_dir / self._image_paths[index]
        label_tensor = torch.tensor(self._labels[index], dtype=torch.float32)

        if self.use_albumentations:
            import cv2

            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]
        else:
            from PIL import Image

            image = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        return image, label_tensor
