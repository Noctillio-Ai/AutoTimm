"""Train a semantic segmentation model using CSV-based data loading.

This example demonstrates:
- Using SegmentationDataModule with format="csv" and train_csv
- CSV format with image_path,mask_path columns
- Training with DeepLabV3+ and combined loss

Usage:
    python examples/data_training/csv_segmentation.py
"""

from __future__ import annotations

import csv
import os
import tempfile

import numpy as np
from PIL import Image

from autotimm import (
    AutoTrainer,
    MetricConfig,
    SegmentationDataModule,
    SemanticSegmentor,
)


def create_demo_dataset(
    root: str, num_images: int = 30, num_classes: int = 5
) -> tuple[str, str, str]:
    """Create a tiny synthetic segmentation CSV dataset.

    Returns (train_csv, val_csv, data_dir).
    """
    img_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for split, n in [("train", num_images), ("val", num_images // 5)]:
        csv_path = os.path.join(root, f"{split}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "mask_path"])
            for i in range(n):
                img_name = f"{split}_{i:04d}.jpg"
                mask_name = f"{split}_{i:04d}.png"

                # Create dummy image
                Image.new("RGB", (128, 128), color=(i * 17 % 256, 80, 120)).save(
                    os.path.join(img_dir, img_name)
                )

                # Create dummy mask with random class assignments
                mask = np.random.randint(0, num_classes, (128, 128), dtype=np.uint8)
                Image.fromarray(mask, mode="L").save(os.path.join(mask_dir, mask_name))

                writer.writerow([f"images/{img_name}", f"masks/{mask_name}"])

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
    #   image_path,mask_path
    #   images/img001.jpg,masks/mask001.png
    #   images/img002.jpg,masks/mask002.png
    #   ...
    tmpdir = tempfile.mkdtemp()
    num_classes = 5
    train_csv, val_csv, data_dir = create_demo_dataset(tmpdir, num_classes=num_classes)

    data = SegmentationDataModule(
        data_dir=data_dir,
        format="csv",
        train_csv=train_csv,
        val_csv=val_csv,
        image_size=128,
        batch_size=4,
        num_workers=0,
        augmentation_preset="default",
    )

    # ------------------------------------------------------------------
    # 2. Define metrics
    # ------------------------------------------------------------------
    metric_configs = [
        MetricConfig(
            name="mIoU",
            backend="torchmetrics",
            metric_class="JaccardIndex",
            params={
                "task": "multiclass",
                "num_classes": num_classes,
                "average": "macro",
            },
            stages=["val"],
            prog_bar=True,
        ),
    ]

    # ------------------------------------------------------------------
    # 3. Create model
    # ------------------------------------------------------------------
    model = SemanticSegmentor(
        backbone="resnet18",
        num_classes=num_classes,
        head_type="deeplabv3plus",
        loss_fn="combined_segmentation",
        metrics=metric_configs,
        lr=1e-4,
    )

    # ------------------------------------------------------------------
    # 4. Train
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
