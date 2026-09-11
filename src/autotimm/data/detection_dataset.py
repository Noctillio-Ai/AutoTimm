"""Detection datasets (COCO-format and CSV-format)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset


class COCODetectionDataset(Dataset):
    """Dataset for COCO-format object detection annotations.

    Expects COCO JSON annotation format with images and annotations.
    Bounding boxes are in COCO format: [x_min, y_min, width, height].

    Parameters:
        images_dir: Directory containing image files.
        annotations_file: Path to COCO JSON annotations file.
        transform: Albumentations transform with bbox_params.
        min_bbox_area: Minimum bbox area to include. Default 0.
        class_ids: Optional list of class IDs to filter. If None, use all classes.

    Attributes:
        images: List of image info dicts from COCO annotations.
        annotations: Dict mapping image_id to list of annotations.
        categories: Dict mapping category_id to category info.
        class_names: List of class names.
        num_classes: Number of classes.
    """

    def __init__(
        self,
        images_dir: str | Path,
        annotations_file: str | Path,
        transform: Callable | None = None,
        min_bbox_area: float = 0.0,
        class_ids: list[int] | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)
        self.transform = transform
        self.min_bbox_area = min_bbox_area

        # Load COCO annotations
        with open(self.annotations_file) as f:
            coco_data = json.load(f)

        self.images = coco_data["images"]
        self.categories = {cat["id"]: cat for cat in coco_data["categories"]}

        # Filter categories if class_ids provided
        if class_ids is not None:
            self.categories = {
                k: v for k, v in self.categories.items() if k in class_ids
            }

        # Create contiguous class mapping (COCO IDs can be non-contiguous)
        self.cat_id_to_class_idx = {
            cat_id: idx for idx, cat_id in enumerate(sorted(self.categories.keys()))
        }
        self.class_idx_to_cat_id = {v: k for k, v in self.cat_id_to_class_idx.items()}
        self.class_names = [
            self.categories[self.class_idx_to_cat_id[i]]["name"]
            for i in range(len(self.categories))
        ]
        self.num_classes = len(self.class_names)

        # Group annotations by image_id
        self.img_id_to_anns: dict[int, list[dict]] = {}
        for ann in coco_data.get("annotations", []):
            # Filter by category
            if class_ids is not None and ann["category_id"] not in class_ids:
                continue
            # Filter by area
            if ann.get("area", 0) < min_bbox_area:
                continue
            # Skip crowd annotations
            if ann.get("iscrowd", 0):
                continue

            img_id = ann["image_id"]
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

        # Create image_id to image info mapping
        self.img_id_to_info = {img["id"]: img for img in self.images}

        # Filter images to only those with annotations
        self.image_ids = [
            img["id"] for img in self.images if img["id"] in self.img_id_to_anns
        ]

        if len(self.image_ids) == 0:
            raise ValueError(
                f"No images with valid annotations found. "
                f"Check that {annotations_file} contains annotations "
                f"and images_dir={images_dir} contains the images."
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get image and annotations.

        Returns:
            Dict with keys:
                - 'image': Tensor [C, H, W]
                - 'boxes': Tensor [N, 4] in (x1, y1, x2, y2) format
                - 'labels': Tensor [N] with class indices
                - 'image_id': int
                - 'orig_size': Tensor [2] with (H, W)
        """
        import cv2

        img_id = self.image_ids[index]
        img_info = self.img_id_to_info[img_id]
        annotations = self.img_id_to_anns.get(img_id, [])

        # Load image
        img_path = self.images_dir / img_info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]

        # Extract bboxes and labels
        bboxes = []
        labels = []
        for ann in annotations:
            bbox = ann["bbox"]  # [x, y, w, h] COCO format
            cat_id = ann["category_id"]

            # Skip if not in our category set
            if cat_id not in self.cat_id_to_class_idx:
                continue

            bboxes.append(bbox)
            labels.append(self.cat_id_to_class_idx[cat_id])

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["labels"]

        # Convert bboxes from COCO [x, y, w, h] to [x1, y1, x2, y2]
        boxes_xyxy = []
        for bbox in bboxes:
            x, y, w, h = bbox
            boxes_xyxy.append([x, y, x + w, y + h])

        # Create output tensors
        if len(boxes_xyxy) > 0:
            boxes = torch.tensor(boxes_xyxy, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
            "orig_size": torch.tensor([orig_h, orig_w]),
        }


class CSVDetectionDataset(Dataset):
    """Dataset for object detection from CSV.

    CSV format (one row per bounding box)::

        image_path,x_min,y_min,x_max,y_max,label
        img001.jpg,10,20,100,200,cat
        img001.jpg,50,60,150,250,dog
        img002.jpg,30,40,120,220,cat

    Multiple rows per image are grouped automatically.

    Parameters:
        csv_path: Path to CSV file.
        image_dir: Directory containing image files.
        image_column: Column name for image paths. Default ``"image_path"``.
        bbox_columns: Column names for bounding box coordinates.
            Default ``["x_min", "y_min", "x_max", "y_max"]``.
        label_column: Column name for class labels. Default ``"label"``.
        transform: Albumentations transform with bbox_params.
        min_bbox_area: Minimum bbox area to include. Default 0.

    Attributes:
        class_names: List of class names.
        num_classes: Number of classes.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path = ".",
        image_column: str = "image_path",
        bbox_columns: list[str] | None = None,
        label_column: str = "label",
        transform: Callable | None = None,
        min_bbox_area: float = 0.0,
    ):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.min_bbox_area = min_bbox_area

        if bbox_columns is None:
            bbox_columns = ["x_min", "y_min", "x_max", "y_max"]

        # Parse CSV and group by image
        image_anns: dict[str, list[dict]] = defaultdict(list)
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])

            # Validate columns exist
            for col in [image_column, label_column] + bbox_columns:
                if col not in fieldnames:
                    raise ValueError(
                        f"Column '{col}' not found in CSV. "
                        f"Available columns: {fieldnames}"
                    )

            for row in reader:
                img_path = row[image_column]
                x1 = float(row[bbox_columns[0]])
                y1 = float(row[bbox_columns[1]])
                x2 = float(row[bbox_columns[2]])
                y2 = float(row[bbox_columns[3]])

                # Filter by area
                area = (x2 - x1) * (y2 - y1)
                if area < min_bbox_area:
                    continue

                image_anns[img_path].append(
                    {"bbox": [x1, y1, x2, y2], "label": row[label_column]}
                )

        if not image_anns:
            raise ValueError(
                f"No images with valid annotations found in {self.csv_path}."
            )

        # Build class mapping
        all_labels = sorted(
            {ann["label"] for anns in image_anns.values() for ann in anns}
        )
        self.class_names: list[str] = all_labels
        self._class_to_idx: dict[str, int] = {
            name: idx for idx, name in enumerate(all_labels)
        }
        self.num_classes: int = len(all_labels)

        # Store as ordered list for indexing
        self._image_paths: list[str] = sorted(image_anns.keys())
        self._image_anns = image_anns

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import cv2

        img_rel = self._image_paths[index]
        img_path = self.image_dir / img_rel
        anns = self._image_anns[img_rel]

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]

        # CSV stores xyxy, but the transform pipelines declare COCO bbox
        # format ([x, y, w, h]); convert before the transform and back after,
        # mirroring COCODetectionDataset.
        bboxes = [
            [b[0], b[1], b[2] - b[0], b[3] - b[1]]
            for b in (ann["bbox"] for ann in anns)
        ]
        labels = [self._class_to_idx[ann["label"]] for ann in anns]

        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["labels"]

        # Convert bboxes from COCO [x, y, w, h] back to [x1, y1, x2, y2]
        boxes_xyxy = [[x, y, x + w, y + h] for x, y, w, h in bboxes]

        if len(boxes_xyxy) > 0:
            boxes = torch.tensor(boxes_xyxy, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "image_id": index,
            "orig_size": torch.tensor([orig_h, orig_w]),
        }


def detection_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function for detection batches.

    Handles variable numbers of objects per image by keeping boxes/labels
    as lists rather than stacking into a single tensor.

    Args:
        batch: List of sample dicts from COCODetectionDataset.

    Returns:
        Dict with:
            - 'image': Tensor [B, C, H, W]
            - 'boxes': List of B tensors, each [N_i, 4]
            - 'labels': List of B tensors, each [N_i]
            - 'image_ids': List of B ints
            - 'orig_sizes': Tensor [B, 2]
    """
    images = torch.stack([sample["image"] for sample in batch])
    boxes = [sample["boxes"] for sample in batch]
    labels = [sample["labels"] for sample in batch]
    image_ids = [sample["image_id"] for sample in batch]
    orig_sizes = torch.stack([sample["orig_size"] for sample in batch])

    return {
        "image": images,
        "boxes": boxes,
        "labels": labels,
        "image_ids": image_ids,
        "orig_sizes": orig_sizes,
    }
