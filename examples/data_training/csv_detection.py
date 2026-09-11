"""Train an object detector using CSV-based data loading.

This example demonstrates:
- Using DetectionDataModule with train_csv for object detection
- CSV format with one row per bounding box (multiple rows per image)
- Auto-detection of class names from CSV
- Custom bbox column names

Usage:
    python examples/data_training/csv_detection.py
"""

from __future__ import annotations

import csv
import os
import tempfile

from PIL import Image

from autotimm import (
    AutoTrainer,
    DetectionDataModule,
    MetricConfig,
    ObjectDetector,
)


def create_demo_dataset(root: str, num_images: int = 50) -> tuple[str, str, str]:
    """Create a tiny synthetic detection CSV dataset.

    Returns (train_csv, val_csv, image_dir).
    """
    img_dir = os.path.join(root, "images")
    os.makedirs(img_dir, exist_ok=True)

    classes = ["car", "person", "bicycle"]

    for split, n in [("train", num_images), ("val", num_images // 5)]:
        csv_path = os.path.join(root, f"{split}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "x_min", "y_min", "x_max", "y_max", "label"])
            for i in range(n):
                fname = f"{split}_{i:04d}.jpg"
                Image.new("RGB", (640, 480), color=(i * 17 % 256, 80, 120)).save(
                    os.path.join(img_dir, fname)
                )
                # Add 1-3 boxes per image
                for j in range(1 + i % 3):
                    x1 = 20 + j * 100
                    y1 = 30 + j * 80
                    x2 = x1 + 80
                    y2 = y1 + 100
                    label = classes[(i + j) % len(classes)]
                    writer.writerow([fname, x1, y1, x2, y2, label])

    return (
        os.path.join(root, "train.csv"),
        os.path.join(root, "val.csv"),
        img_dir,
    )


def main():
    # ------------------------------------------------------------------
    # 1. Prepare data (replace with your own CSV files and image dir)
    # ------------------------------------------------------------------
    # In practice your CSV looks like:
    #
    #   image_path,x_min,y_min,x_max,y_max,label
    #   img_001.jpg,10,20,100,200,car
    #   img_001.jpg,50,60,150,250,person
    #   img_002.jpg,30,40,120,220,car
    #   ...
    tmpdir = tempfile.mkdtemp()
    train_csv, val_csv, image_dir = create_demo_dataset(tmpdir)

    data = DetectionDataModule(
        train_csv=train_csv,
        val_csv=val_csv,
        image_dir=image_dir,
        image_size=640,
        batch_size=8,
        num_workers=0,
    )
    data.setup("fit")

    print(f"Classes: {data.class_names}")
    print(f"Num classes: {data.num_classes}")
    print(f"Train images: {len(data.train_dataset)}")
    print(f"Val images: {len(data.val_dataset)}")

    # ------------------------------------------------------------------
    # 2. Define metrics
    # ------------------------------------------------------------------
    metric_configs = [
        MetricConfig(
            name="mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "bbox"},
            stages=["val"],
            prog_bar=True,
        ),
    ]

    # ------------------------------------------------------------------
    # 3. Create model
    # ------------------------------------------------------------------
    model = ObjectDetector(
        backbone="resnet18",
        num_classes=data.num_classes,
        metrics=metric_configs,
        lr=1e-4,
    )

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    trainer = AutoTrainer(
        max_epochs=3,
        accelerator="auto",
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=data)

    print("\nTraining complete!")
    print(
        "In production, replace the synthetic dataset with your own CSV "
        "and image directory."
    )


if __name__ == "__main__":
    main()
