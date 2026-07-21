"""Datasets for instance segmentation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from pycocotools import mask as mask_utils

    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    mask_utils = None


class COCOInstanceDataset(Dataset):
    """COCO instance segmentation dataset.

    Loads COCO-format annotations with instance segmentation masks.

    Args:
        data_dir: Root directory of COCO dataset
        split: Dataset split ('train', 'val', 'test')
        transforms: Albumentations transforms to apply
        min_keypoints: Minimum number of keypoints for an instance to be valid
        min_area: Minimum area for an instance to be valid
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transforms: Any = None,
        min_keypoints: int = 0,
        min_area: float = 0.0,
    ):
        if not HAS_PYCOCOTOOLS:
            raise ImportError(
                "pycocotools is required for COCO instance segmentation. "
                "Install it with: pip install autotimm[coco]  "
                "(or: pip install pycocotools)"
            )

        self.data_dir = Path(data_dir)
        self.split = split
        self.transforms = transforms
        self.min_keypoints = min_keypoints
        self.min_area = min_area

        # Load COCO annotations
        ann_file = self.data_dir / "annotations" / f"instances_{split}2017.json"
        if not ann_file.exists():
            # Try without year suffix
            ann_file = self.data_dir / "annotations" / f"instances_{split}.json"

        if not ann_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {ann_file}")

        with open(ann_file) as f:
            self.coco_data = json.load(f)

        # Build index
        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.categories = {cat["id"]: cat for cat in self.coco_data["categories"]}

        # Create category ID to contiguous ID mapping
        category_ids = sorted(self.categories.keys())
        self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(category_ids)}

        # Group annotations by image
        self.img_to_anns = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Filter images with valid annotations
        self.image_ids = []
        for img_id in self.images.keys():
            if img_id in self.img_to_anns:
                self.image_ids.append(img_id)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _decode_mask(
        self, segmentation: list | dict, height: int, width: int
    ) -> np.ndarray:
        """Decode COCO segmentation to binary mask.

        Args:
            segmentation: COCO segmentation (polygon or RLE)
            height: Image height
            width: Image width

        Returns:
            Binary mask [H, W]
        """
        if isinstance(segmentation, list):
            # Polygon format
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        elif isinstance(segmentation, dict):
            # RLE format
            if isinstance(segmentation["counts"], list):
                # Uncompressed RLE
                rle = mask_utils.frPyObjects(segmentation, height, width)
            else:
                # Compressed RLE
                rle = segmentation
        else:
            raise ValueError(f"Unknown segmentation format: {type(segmentation)}")

        mask = mask_utils.decode(rle)
        return mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample from the dataset.

        Returns:
            Dict with keys:
            - image: [C, H, W] tensor
            - boxes: [N, 4] tensor in xyxy format
            - labels: [N] tensor with class indices
            - masks: [N, H, W] binary masks
            - image_id: int
            - orig_size: [2] tensor (H, W)
        """
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.data_dir / f"{self.split}2017" / img_info["file_name"]
        if not img_path.exists():
            # Try without year suffix
            img_path = self.data_dir / self.split / img_info["file_name"]

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])

        # Filter valid annotations
        boxes = []
        labels = []
        masks = []

        for ann in anns:
            # Skip if no segmentation
            if "segmentation" not in ann:
                continue

            # Skip if too small
            if ann.get("area", 0) < self.min_area:
                continue

            # Skip if not enough keypoints
            if ann.get("num_keypoints", 0) < self.min_keypoints:
                continue

            # Skip if iscrowd
            if ann.get("iscrowd", 0):
                continue

            # Get bbox in xyxy format
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])

            # Get label (convert to contiguous)
            cat_id = ann["category_id"]
            label = self.cat_id_to_label[cat_id]
            labels.append(label)

            # Decode mask
            mask = self._decode_mask(ann["segmentation"], height, width)
            masks.append(mask)

        # Convert to arrays
        if len(boxes) == 0:
            # No valid annotations - create dummy data
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            masks = np.zeros((0, height, width), dtype=np.uint8)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            masks = np.stack(masks, axis=0).astype(np.uint8)

        # Store original size
        orig_size = torch.tensor([height, width], dtype=torch.long)

        # Apply transforms
        if self.transforms:
            # Albumentations expects masks as list for instance segmentation
            mask_list = [masks[i] for i in range(len(masks))]

            transformed = self.transforms(
                image=image,
                masks=mask_list,
                bboxes=boxes,
                labels=labels,
            )

            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]
            masks = transformed["masks"]

            # Convert back to array
            if len(masks) > 0:
                masks = np.stack(masks, axis=0)
            else:
                masks = np.zeros((0, image.shape[1], image.shape[2]), dtype=np.uint8)

            # Convert to tensors
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            masks = torch.from_numpy(masks).float()
        else:
            # Convert to tensors manually
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            boxes = torch.from_numpy(boxes).float()
            labels = torch.from_numpy(labels).long()
            masks = torch.from_numpy(masks).float()

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": img_id,
            "orig_size": orig_size,
        }


class CSVInstanceDataset(Dataset):
    """Dataset for instance segmentation from CSV.

    CSV format (one row per instance)::

        image_path,x_min,y_min,x_max,y_max,label,mask_path
        img001.jpg,10,20,100,200,cat,masks/img001_0.png
        img001.jpg,50,60,150,250,dog,masks/img001_1.png

    Each ``mask_path`` is a binary mask PNG for that instance.
    Does NOT require pycocotools.

    Args:
        csv_path: Path to CSV file.
        image_dir: Directory containing images and masks.
        image_column: Column name for image paths. Default ``"image_path"``.
        bbox_columns: Column names for bbox coordinates.
            Default ``["x_min", "y_min", "x_max", "y_max"]``.
        label_column: Column name for class labels. Default ``"label"``.
        mask_column: Column name for mask file paths. Default ``"mask_path"``.
        transform: Albumentations transforms to apply.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path = ".",
        image_column: str = "image_path",
        bbox_columns: list[str] | None = None,
        label_column: str = "label",
        mask_column: str = "mask_path",
        transform: Any = None,
    ):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        if bbox_columns is None:
            bbox_columns = ["x_min", "y_min", "x_max", "y_max"]

        # Parse CSV and group by image
        image_anns: dict[str, list[dict]] = defaultdict(list)
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])

            for col in [image_column, label_column, mask_column] + bbox_columns:
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

                image_anns[img_path].append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "label": row[label_column],
                        "mask_path": row[mask_column],
                    }
                )

        if not image_anns:
            raise ValueError(f"No images with annotations found in {self.csv_path}.")

        # Build class mapping
        all_labels = sorted(
            {ann["label"] for anns in image_anns.values() for ann in anns}
        )
        self.class_names: list[str] = all_labels
        self._class_to_idx: dict[str, int] = {
            name: idx for idx, name in enumerate(all_labels)
        }
        self.num_classes: int = len(all_labels)

        self._image_paths: list[str] = sorted(image_anns.keys())
        self._image_anns = image_anns

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_rel = self._image_paths[idx]
        img_path = self.image_dir / img_rel
        anns = self._image_anns[img_rel]

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        boxes = []
        labels = []
        masks = []

        for ann in anns:
            boxes.append(ann["bbox"])
            labels.append(self._class_to_idx[ann["label"]])
            mask = cv2.imread(
                str(self.image_dir / ann["mask_path"]), cv2.IMREAD_GRAYSCALE
            )
            if mask is None:
                raise RuntimeError(
                    f"Failed to load mask: {self.image_dir / ann['mask_path']}"
                )
            # Binarize
            mask = (mask > 0).astype(np.uint8)
            masks.append(mask)

        orig_size = torch.tensor([height, width], dtype=torch.long)

        if len(boxes) == 0:
            boxes_arr = np.zeros((0, 4), dtype=np.float32)
            labels_arr = np.zeros((0,), dtype=np.int64)
            masks_arr = np.zeros((0, height, width), dtype=np.uint8)
        else:
            boxes_arr = np.array(boxes, dtype=np.float32)
            labels_arr = np.array(labels, dtype=np.int64)
            masks_arr = np.stack(masks, axis=0)

        if self.transform:
            mask_list = [masks_arr[i] for i in range(len(masks_arr))]
            transformed = self.transform(
                image=image,
                masks=mask_list,
                bboxes=boxes_arr.tolist(),
                labels=labels_arr.tolist(),
            )
            image = transformed["image"]
            bboxes_out = transformed["bboxes"]
            labels_out = transformed["labels"]
            masks_out = transformed["masks"]

            if len(masks_out) > 0:
                masks_arr = np.stack(masks_out, axis=0)
            else:
                masks_arr = np.zeros(
                    (0, image.shape[1], image.shape[2]), dtype=np.uint8
                )

            boxes_t = torch.tensor(bboxes_out, dtype=torch.float32)
            labels_t = torch.tensor(labels_out, dtype=torch.long)
            masks_t = torch.from_numpy(masks_arr).float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            boxes_t = torch.from_numpy(boxes_arr).float()
            labels_t = torch.from_numpy(labels_arr).long()
            masks_t = torch.from_numpy(masks_arr).float()

        return {
            "image": image,
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": idx,
            "orig_size": orig_size,
        }


def collate_instance_segmentation(batch: list[dict]) -> dict[str, Any]:
    """Collate function for instance segmentation datasets.

    Handles variable number of instances per image.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched dict with lists for variable-length tensors
    """
    images = torch.stack([sample["image"] for sample in batch])
    boxes = [sample["boxes"] for sample in batch]
    labels = [sample["labels"] for sample in batch]
    masks = [sample["masks"] for sample in batch]
    image_ids = [sample["image_id"] for sample in batch]
    orig_sizes = torch.stack([sample["orig_size"] for sample in batch])

    return {
        "image": images,
        "boxes": boxes,
        "labels": labels,
        "masks": masks,
        "image_id": image_ids,
        "orig_size": orig_sizes,
    }
