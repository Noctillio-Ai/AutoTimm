"""Train an instance segmentation model using CSV-based data loading.

This example demonstrates:
- Using InstanceSegmentationDataModule with train_csv
- CSV format with bounding boxes, labels, and per-instance binary mask PNGs
- No pycocotools required for CSV mode

Usage:
    python examples/data_training/csv_instance_segmentation.py
"""

from __future__ import annotations

import csv
import os
import tempfile

import numpy as np
from PIL import Image

from autotimm import (
    AutoTrainer,
    InstanceSegmentationDataModule,
    InstanceSegmentor,
)


def create_demo_dataset(root: str, num_images: int = 30) -> tuple[str, str, str]:
    """Create a tiny synthetic instance segmentation CSV dataset.

    Returns (train_csv, val_csv, data_dir).
    """
    img_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    classes = ["car", "person"]

    for split, n in [("train", num_images), ("val", num_images // 5)]:
        csv_path = os.path.join(root, f"{split}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "image_path",
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                    "label",
                    "mask_path",
                ]
            )
            for i in range(n):
                img_name = f"{split}_{i:04d}.jpg"
                Image.new("RGB", (128, 128), color=(i * 17 % 256, 80, 120)).save(
                    os.path.join(img_dir, img_name)
                )

                # Add 1-2 instances per image
                for j in range(1 + i % 2):
                    label = classes[(i + j) % len(classes)]
                    mask_name = f"{split}_{i:04d}_{j}.png"

                    # Create binary mask with a blob
                    mask = np.zeros((128, 128), dtype=np.uint8)
                    x1 = 10 + j * 40
                    y1 = 15 + j * 30
                    x2 = x1 + 50
                    y2 = y1 + 60
                    mask[y1:y2, x1:x2] = 255
                    Image.fromarray(mask, mode="L").save(
                        os.path.join(mask_dir, mask_name)
                    )

                    writer.writerow(
                        [
                            f"images/{img_name}",
                            x1,
                            y1,
                            x2,
                            y2,
                            label,
                            f"masks/{mask_name}",
                        ]
                    )

    return (
        os.path.join(root, "train.csv"),
        os.path.join(root, "val.csv"),
        root,
    )


def main():
    # ------------------------------------------------------------------
    # 1. Prepare data (replace with your own CSV files)
    # ------------------------------------------------------------------
    # In practice your CSV looks like:
    #
    #   image_path,x_min,y_min,x_max,y_max,label,mask_path
    #   images/img001.jpg,10,20,100,200,car,masks/img001_0.png
    #   images/img001.jpg,50,60,150,250,person,masks/img001_1.png
    #   ...
    tmpdir = tempfile.mkdtemp()
    train_csv, val_csv, data_dir = create_demo_dataset(tmpdir)

    data = InstanceSegmentationDataModule(
        train_csv=train_csv,
        val_csv=val_csv,
        image_dir=data_dir,
        image_size=128,
        batch_size=4,
        num_workers=0,
    )
    data.setup("fit")

    print(f"Train images: {len(data.train_dataset)}")
    print(f"Val images: {len(data.val_dataset)}")

    # ------------------------------------------------------------------
    # 2. Create model
    # ------------------------------------------------------------------
    model = InstanceSegmentor(
        backbone="resnet18",
        num_classes=2,
        lr=1e-4,
    )

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    trainer = AutoTrainer(
        max_epochs=3,
        accelerator="auto",
    )

    trainer.fit(model, datamodule=data)

    print("\nTraining complete!")
    print(
        "In production, replace the synthetic dataset with your own CSV "
        "and image/mask directories."
    )


if __name__ == "__main__":
    main()
