"""Train a semantic segmentation model.

This example demonstrates:
- Semantic segmentation with timm backbones
- Custom dataset loading with multiple format support
- Training with various loss functions (CE, Dice, Focal, Combined)
- Evaluation with mIoU and pixel accuracy metrics

Requirements:
- Dataset in supported format (PNG, Cityscapes, VOC, COCO)
- albumentations (optional): pip install autotimm[albumentations]

Expected directory structure for PNG format:
    ./data/
      train/
        images/       # Training images
        masks/        # Training masks (single-channel PNG)
      val/
        images/       # Validation images
        masks/        # Validation masks
"""

from autotimm import (
    AutoTrainer,
    LoggerConfig,
    LoggingConfig,
    MetricConfig,
    SegmentationDataModule,
    SemanticSegmentor,
)


def main():
    # 1. Data - PNG format segmentation dataset
    data = SegmentationDataModule(
        data_dir="./data",
        format="png",  # Supported: 'png', 'cityscapes', 'voc', 'coco'
        image_size=512,
        batch_size=8,
        num_workers=4,
        augmentation_preset="default",  # Options: "default", "strong", "light"
        ignore_index=255,  # Index for ignored pixels (e.g., boundaries)
    )

    # 2. Configure metrics
    # JaccardIndex (IoU) is the standard metric for segmentation
    metric_configs = [
        MetricConfig(
            name="mIoU",
            backend="torchmetrics",
            metric_class="JaccardIndex",
            params={
                "task": "multiclass",
                "num_classes": 21,  # Adjust based on your dataset
                "average": "macro",
                "ignore_index": 255,
            },
            stages=["val", "test"],
            prog_bar=True,
        ),
        MetricConfig(
            name="pixel_acc",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={
                "task": "multiclass",
                "num_classes": 21,
                "ignore_index": 255,
            },
            stages=["val", "test"],
        ),
    ]

    # 3. Create semantic segmentation model
    # Uses DeepLabV3+ by default with:
    # - timm backbone (ResNet-50 by default)
    # - ASPP module for multi-scale context
    # - Decoder with skip connections
    model = SemanticSegmentor(
        backbone="resnet50",  # Any timm backbone works
        num_classes=21,  # Number of segmentation classes
        head_type="deeplabv3plus",  # Options: "deeplabv3plus", "fcn"
        loss_fn="combined_segmentation",  # Options: "cross_entropy", "dice", "focal_pixelwise", "combined_segmentation"
        ce_weight=1.0,  # Weight for cross-entropy loss (when using "combined_segmentation")
        dice_weight=1.0,  # Weight for Dice loss (when using "combined_segmentation")
        ignore_index=255,
        metrics=metric_configs,
        logging_config=LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=True,
        ),
        # Optimizer settings
        lr=1e-4,
        weight_decay=1e-4,
        optimizer="adamw",
        scheduler="cosine",
    )

    # 4. Train
    trainer = AutoTrainer(
        max_epochs=100,
        accelerator="auto",
        devices=1,
        precision="16-mixed",  # Use mixed precision for faster training
        logger=[
            LoggerConfig(
                backend="tensorboard", params={"save_dir": "logs/semantic_seg"}
            ),
        ],
        checkpoint_monitor="val/mIoU",
        checkpoint_mode="max",
    )

    # Train the model
    trainer.fit(model, datamodule=data)

    # Test the model
    results = trainer.test(model, datamodule=data)
    print(f"Test mIoU: {results[0]['test/mIoU']:.4f}")
    print(f"Test Pixel Accuracy: {results[0]['test/pixel_acc']:.4f}")


if __name__ == "__main__":
    main()
