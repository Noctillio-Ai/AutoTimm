"""Tests for CSV-based data loading across all task types."""

import csv
import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from autotimm.data.dataset import CSVImageDataset
from autotimm.data.datamodule import ImageDataModule
from autotimm.data.detection_dataset import CSVDetectionDataset
from autotimm.data.detection_datamodule import DetectionDataModule
from autotimm.data.segmentation_dataset import SemanticSegmentationDataset
from autotimm.data.segmentation_datamodule import SegmentationDataModule
from autotimm.data.instance_dataset import CSVInstanceDataset
from autotimm.data.instance_datamodule import InstanceSegmentationDataModule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMG_SIZE = 32


def _create_dummy_image(path, size=(IMG_SIZE, IMG_SIZE)):
    """Create a small dummy RGB image."""
    Image.new("RGB", size, color=(100, 150, 200)).save(path)


def _create_dummy_mask(path, size=(IMG_SIZE, IMG_SIZE), num_classes=3):
    """Create a dummy segmentation mask."""
    arr = np.random.randint(0, num_classes, size=size, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _create_dummy_binary_mask(path, size=(IMG_SIZE, IMG_SIZE)):
    """Create a dummy binary instance mask."""
    arr = np.zeros(size, dtype=np.uint8)
    # Put a blob in the center
    h, w = size
    arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255
    Image.fromarray(arr, mode="L").save(path)


def _create_classification_csv(tmpdir, num_images=10, classes=None):
    """Create classification CSV and dummy images."""
    if classes is None:
        classes = ["cat", "dog", "bird"]

    img_dir = os.path.join(tmpdir, "images")
    os.makedirs(img_dir, exist_ok=True)

    csv_path = os.path.join(tmpdir, "labels.csv")
    rows = []
    for i in range(num_images):
        fname = f"img_{i:03d}.jpg"
        _create_dummy_image(os.path.join(img_dir, fname))
        label = classes[i % len(classes)]
        rows.append([fname, label])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(rows)

    return csv_path, img_dir, classes


def _create_detection_csv(tmpdir, num_images=5, classes=None):
    """Create detection CSV and dummy images."""
    if classes is None:
        classes = ["cat", "dog"]

    img_dir = os.path.join(tmpdir, "images")
    os.makedirs(img_dir, exist_ok=True)

    csv_path = os.path.join(tmpdir, "annotations.csv")
    rows = []
    for i in range(num_images):
        fname = f"img_{i:03d}.jpg"
        _create_dummy_image(os.path.join(img_dir, fname))
        # Add 1-2 boxes per image
        for j in range(1 + i % 2):
            label = classes[(i + j) % len(classes)]
            x1, y1 = 2 + j * 5, 3 + j * 5
            x2, y2 = x1 + 10, y1 + 10
            rows.append([fname, x1, y1, x2, y2, label])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "x_min", "y_min", "x_max", "y_max", "label"])
        writer.writerows(rows)

    return csv_path, img_dir, classes


def _create_segmentation_csv(tmpdir, num_images=5):
    """Create segmentation CSV and dummy images + masks."""
    img_dir = os.path.join(tmpdir, "images")
    mask_dir = os.path.join(tmpdir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    csv_path = os.path.join(tmpdir, "segmentation.csv")
    rows = []
    for i in range(num_images):
        img_name = f"img_{i:03d}.jpg"
        mask_name = f"mask_{i:03d}.png"
        _create_dummy_image(os.path.join(img_dir, img_name))
        _create_dummy_mask(os.path.join(mask_dir, mask_name))
        rows.append([f"images/{img_name}", f"masks/{mask_name}"])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "mask_path"])
        writer.writerows(rows)

    return csv_path, tmpdir


def _create_instance_csv(tmpdir, num_images=5, classes=None):
    """Create instance segmentation CSV with images, boxes, and masks."""
    if classes is None:
        classes = ["cat", "dog"]

    img_dir = os.path.join(tmpdir, "images")
    mask_dir = os.path.join(tmpdir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    csv_path = os.path.join(tmpdir, "instances.csv")
    rows = []
    for i in range(num_images):
        img_name = f"img_{i:03d}.jpg"
        _create_dummy_image(os.path.join(img_dir, img_name))

        for j in range(1 + i % 2):
            label = classes[(i + j) % len(classes)]
            mask_name = f"mask_{i:03d}_{j}.png"
            _create_dummy_binary_mask(os.path.join(mask_dir, mask_name))
            x1, y1 = 2 + j * 5, 3 + j * 5
            x2, y2 = x1 + 10, y1 + 10
            rows.append(
                [f"images/{img_name}", x1, y1, x2, y2, label, f"masks/{mask_name}"]
            )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_path", "x_min", "y_min", "x_max", "y_max", "label", "mask_path"]
        )
        writer.writerows(rows)

    return csv_path, tmpdir, classes


# ---------------------------------------------------------------------------
# CSVImageDataset tests
# ---------------------------------------------------------------------------


class TestCSVImageDataset:
    def test_creation_and_class_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, classes = _create_classification_csv(tmpdir)
            ds = CSVImageDataset(csv_path=csv_path, image_dir=img_dir)

            assert ds.num_classes == len(classes)
            assert ds.classes == sorted(classes)
            assert len(ds) == 10

    def test_returns_correct_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, _ = _create_classification_csv(tmpdir, num_images=3)
            from torchvision import transforms

            tfm = transforms.Compose(
                [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
            )
            ds = CSVImageDataset(csv_path=csv_path, image_dir=img_dir, transform=tfm)

            image, target = ds[0]
            assert isinstance(image, torch.Tensor)
            assert image.shape == (3, IMG_SIZE, IMG_SIZE)
            assert isinstance(target, int)

    def test_samples_attribute(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, _ = _create_classification_csv(tmpdir, num_images=5)
            ds = CSVImageDataset(csv_path=csv_path, image_dir=img_dir)

            assert len(ds.samples) == 5
            # Each sample is (image_rel_path, class_idx)
            for path, idx in ds.samples:
                assert isinstance(path, str)
                assert isinstance(idx, int)
                assert 0 <= idx < ds.num_classes

    def test_class_to_idx_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, classes = _create_classification_csv(
                tmpdir, classes=["zebra", "apple"]
            )
            ds = CSVImageDataset(csv_path=csv_path, image_dir=img_dir)

            # Classes should be sorted
            assert ds.classes == ["apple", "zebra"]
            assert ds.class_to_idx["apple"] == 0
            assert ds.class_to_idx["zebra"] == 1

    def test_explicit_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, _ = _create_classification_csv(tmpdir)
            ds = CSVImageDataset(
                csv_path=csv_path,
                image_dir=img_dir,
                image_column="image_path",
                label_column="label",
            )
            assert len(ds) == 10

    def test_missing_column_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, _ = _create_classification_csv(tmpdir)
            with pytest.raises(ValueError, match="Image column"):
                CSVImageDataset(
                    csv_path=csv_path,
                    image_dir=img_dir,
                    image_column="nonexistent",
                )

    def test_missing_label_column_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, _ = _create_classification_csv(tmpdir)
            with pytest.raises(ValueError, match="Label column"):
                CSVImageDataset(
                    csv_path=csv_path,
                    image_dir=img_dir,
                    label_column="nonexistent",
                )

    def test_empty_csv_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "empty.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image_path", "label"])
                # No data rows

            with pytest.raises(ValueError, match="no data rows"):
                CSVImageDataset(csv_path=csv_path, image_dir=tmpdir)


# ---------------------------------------------------------------------------
# ImageDataModule CSV mode tests
# ---------------------------------------------------------------------------


class TestImageDataModuleCSV:
    def test_setup_with_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, classes = _create_classification_csv(
                tmpdir, num_images=20
            )
            dm = ImageDataModule(
                train_csv=csv_path,
                image_dir=img_dir,
                val_split=0.2,
                image_size=IMG_SIZE,
                batch_size=4,
                num_workers=0,
            )
            dm.setup("fit")

            assert dm.num_classes == len(classes)
            assert dm.class_names is not None

    def test_setup_with_separate_val_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_csv, img_dir, classes = _create_classification_csv(
                tmpdir, num_images=10
            )
            val_dir = os.path.join(tmpdir, "val_data")
            val_csv, _, _ = _create_classification_csv(
                val_dir, num_images=5, classes=classes
            )

            dm = ImageDataModule(
                train_csv=train_csv,
                val_csv=val_csv,
                image_dir=img_dir,
                image_size=IMG_SIZE,
                batch_size=4,
                num_workers=0,
            )
            dm.setup("fit")

            assert len(dm.train_dataset) == 10
            assert len(dm.val_dataset) == 5

    def test_dataloaders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, classes = _create_classification_csv(
                tmpdir, num_images=8
            )
            dm = ImageDataModule(
                train_csv=csv_path,
                image_dir=img_dir,
                val_split=0.25,
                image_size=IMG_SIZE,
                batch_size=2,
                num_workers=0,
            )
            dm.setup("fit")

            batch = next(iter(dm.train_dataloader()))
            images, labels = batch
            assert images.ndim == 4
            assert images.shape[1] == 3

    def test_balanced_sampling_with_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, _ = _create_classification_csv(tmpdir, num_images=12)

            dm = ImageDataModule(
                train_csv=csv_path,
                image_dir=img_dir,
                val_split=0.2,
                image_size=IMG_SIZE,
                batch_size=4,
                num_workers=0,
                balanced_sampling=True,
            )
            dm.setup("fit")

            # Should have _train_targets set for balanced sampling
            assert dm._train_targets is not None


# ---------------------------------------------------------------------------
# CSVDetectionDataset tests
# ---------------------------------------------------------------------------


class TestCSVDetectionDataset:
    def test_creation_and_class_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, classes = _create_detection_csv(tmpdir)
            ds = CSVDetectionDataset(csv_path=csv_path, image_dir=img_dir)

            assert ds.num_classes == len(classes)
            assert ds.class_names == sorted(classes)

    def test_multi_box_per_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, _ = _create_detection_csv(tmpdir, num_images=5)
            ds = CSVDetectionDataset(csv_path=csv_path, image_dir=img_dir)

            # Some images should have multiple boxes
            found_multi = False
            for i in range(len(ds)):
                sample = ds[i]
                if sample["boxes"].shape[0] > 1:
                    found_multi = True
                    break
            assert found_multi, "Expected at least one image with multiple boxes"

    def test_returns_correct_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, _ = _create_detection_csv(tmpdir, num_images=3)
            ds = CSVDetectionDataset(csv_path=csv_path, image_dir=img_dir)

            sample = ds[0]
            assert "image" in sample
            assert "boxes" in sample
            assert "labels" in sample
            assert "image_id" in sample
            assert "orig_size" in sample

            assert sample["boxes"].ndim == 2
            assert sample["boxes"].shape[1] == 4
            assert sample["labels"].ndim == 1
            assert sample["orig_size"].shape == (2,)

    def test_missing_column_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, _ = _create_detection_csv(tmpdir)
            with pytest.raises(ValueError, match="Column"):
                CSVDetectionDataset(
                    csv_path=csv_path,
                    image_dir=img_dir,
                    bbox_columns=["bad_x1", "bad_y1", "bad_x2", "bad_y2"],
                )


# ---------------------------------------------------------------------------
# DetectionDataModule CSV mode tests
# ---------------------------------------------------------------------------


class TestDetectionDataModuleCSV:
    def test_setup_with_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, classes = _create_detection_csv(tmpdir)
            dm = DetectionDataModule(
                train_csv=csv_path,
                val_csv=csv_path,
                image_dir=img_dir,
                image_size=IMG_SIZE,
                batch_size=2,
                num_workers=0,
            )
            dm.setup("fit")

            assert dm.num_classes == len(classes)
            assert dm.train_dataset is not None
            assert dm.val_dataset is not None

    def test_dataloaders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, _ = _create_detection_csv(tmpdir)
            dm = DetectionDataModule(
                train_csv=csv_path,
                val_csv=csv_path,
                image_dir=img_dir,
                image_size=IMG_SIZE,
                batch_size=2,
                num_workers=0,
            )
            dm.setup("fit")

            batch = next(iter(dm.train_dataloader()))
            assert "image" in batch
            assert "boxes" in batch
            assert "labels" in batch
            assert batch["image"].ndim == 4


# ---------------------------------------------------------------------------
# SemanticSegmentationDataset CSV format tests
# ---------------------------------------------------------------------------


class TestSegmentationDatasetCSV:
    def test_csv_format_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, data_dir = _create_segmentation_csv(tmpdir, num_images=5)
            ds = SemanticSegmentationDataset(
                data_dir=data_dir,
                format="csv",
                csv_path=csv_path,
            )

            assert len(ds) == 5

    def test_csv_format_returns_correct_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, data_dir = _create_segmentation_csv(tmpdir, num_images=3)
            ds = SemanticSegmentationDataset(
                data_dir=data_dir,
                format="csv",
                csv_path=csv_path,
            )

            sample = ds[0]
            assert "image" in sample
            assert "mask" in sample
            assert "image_id" in sample
            assert "orig_size" in sample

    def test_csv_format_without_csv_path_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="csv_path is required"):
                SemanticSegmentationDataset(
                    data_dir=tmpdir,
                    format="csv",
                )


# ---------------------------------------------------------------------------
# SegmentationDataModule CSV mode tests
# ---------------------------------------------------------------------------


class TestSegmentationDataModuleCSV:
    def test_setup_with_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, data_dir = _create_segmentation_csv(tmpdir, num_images=5)
            dm = SegmentationDataModule(
                data_dir=data_dir,
                format="csv",
                train_csv=csv_path,
                image_size=IMG_SIZE,
                batch_size=2,
                num_workers=0,
            )
            dm.setup("fit")

            assert dm.train_dataset is not None
            assert len(dm.train_dataset) == 5

    def test_dataloaders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, data_dir = _create_segmentation_csv(tmpdir, num_images=4)
            dm = SegmentationDataModule(
                data_dir=data_dir,
                format="csv",
                train_csv=csv_path,
                val_csv=csv_path,
                image_size=IMG_SIZE,
                batch_size=2,
                num_workers=0,
            )
            dm.setup("fit")

            batch = next(iter(dm.train_dataloader()))
            assert "image" in batch
            assert "mask" in batch
            assert batch["image"].ndim == 4


# ---------------------------------------------------------------------------
# CSVInstanceDataset tests
# ---------------------------------------------------------------------------


class TestCSVInstanceDataset:
    def test_creation_and_class_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, data_dir, classes = _create_instance_csv(tmpdir)
            ds = CSVInstanceDataset(csv_path=csv_path, image_dir=data_dir)

            assert ds.num_classes == len(classes)
            assert ds.class_names == sorted(classes)

    def test_multi_instance_per_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, data_dir, _ = _create_instance_csv(tmpdir, num_images=5)
            ds = CSVInstanceDataset(csv_path=csv_path, image_dir=data_dir)

            found_multi = False
            for i in range(len(ds)):
                sample = ds[i]
                if sample["boxes"].shape[0] > 1:
                    found_multi = True
                    break
            assert found_multi, "Expected at least one image with multiple instances"

    def test_returns_correct_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, data_dir, _ = _create_instance_csv(tmpdir, num_images=3)
            ds = CSVInstanceDataset(csv_path=csv_path, image_dir=data_dir)

            sample = ds[0]
            assert "image" in sample
            assert "boxes" in sample
            assert "labels" in sample
            assert "masks" in sample
            assert "image_id" in sample
            assert "orig_size" in sample

            assert sample["boxes"].ndim == 2
            assert sample["boxes"].shape[1] == 4
            assert sample["labels"].ndim == 1
            assert sample["masks"].ndim == 3  # [N, H, W]
            assert sample["masks"].shape[0] == sample["boxes"].shape[0]

    def test_mask_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, data_dir, _ = _create_instance_csv(tmpdir, num_images=2)
            ds = CSVInstanceDataset(csv_path=csv_path, image_dir=data_dir)

            sample = ds[0]
            # Masks should be binary (0 or 1 after binarization)
            mask_vals = sample["masks"].unique()
            for v in mask_vals:
                assert v.item() in (0.0, 1.0)

    def test_missing_column_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, data_dir, _ = _create_instance_csv(tmpdir)
            with pytest.raises(ValueError, match="Column"):
                CSVInstanceDataset(
                    csv_path=csv_path,
                    image_dir=data_dir,
                    mask_column="nonexistent_mask",
                )


# ---------------------------------------------------------------------------
# InstanceSegmentationDataModule CSV mode tests
# ---------------------------------------------------------------------------


class TestInstanceSegmentationDataModuleCSV:
    def test_setup_with_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, data_dir, classes = _create_instance_csv(tmpdir)
            dm = InstanceSegmentationDataModule(
                train_csv=csv_path,
                val_csv=csv_path,
                image_dir=data_dir,
                image_size=IMG_SIZE,
                batch_size=2,
                num_workers=0,
            )
            dm.setup("fit")

            assert dm.train_dataset is not None
            assert dm.val_dataset is not None

    def test_dataloaders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, data_dir, _ = _create_instance_csv(tmpdir, num_images=4)
            dm = InstanceSegmentationDataModule(
                train_csv=csv_path,
                val_csv=csv_path,
                image_dir=data_dir,
                image_size=IMG_SIZE,
                batch_size=2,
                num_workers=0,
            )
            dm.setup("fit")

            batch = next(iter(dm.train_dataloader()))
            assert "image" in batch
            assert "boxes" in batch
            assert "labels" in batch
            assert "masks" in batch
            assert batch["image"].ndim == 4
