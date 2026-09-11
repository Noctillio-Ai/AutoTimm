"""Train an instance segmentation model.

This example demonstrates:
- Instance segmentation with timm backbones (Mask R-CNN style)
- COCO dataset loading with instance masks
- Training with detection + mask losses
- Evaluation with mask mAP and bbox mAP metrics

Requirements:
- COCO dataset downloaded to ./coco directory
- pycocotools: pip install pycocotools

Expected directory structure:
    ./coco/
      train2017/           # Training images
      val2017/             # Validation images
      annotations/
        instances_train2017.json
        instances_val2017.json
"""

from autotimm import (
    AutoTrainer,
    InstanceSegmentationDataModule,
    InstanceSegmentor,
    LoggerConfig,
    LoggingConfig,
    MetricConfig,
)


def main():
    # 1. Data - COCO format instance segmentation dataset
    data = InstanceSegmentationDataModule(
        data_dir="./coco",
        image_size=640,
        batch_size=4,  # Instance segmentation needs more memory
        num_workers=4,
        augmentation_preset="default",  # Options: "default", "strong", "light"
        min_area=0.0,  # Minimum area for an instance to be valid
    )

    # 2. Configure metrics
    # MeanAveragePrecision with mask support is the standard metric
    metric_configs = [
        MetricConfig(
            name="mask_mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={
                "box_format": "xyxy",
                "iou_type": "segm",  # Use mask IoU instead of bbox IoU
            },
            stages=["val", "test"],
            prog_bar=True,
        ),
        MetricConfig(
            name="bbox_mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={
                "box_format": "xyxy",
                "iou_type": "bbox",  # Bounding box mAP for comparison
            },
            stages=["val", "test"],
        ),
    ]

    # 3. Create instance segmentation model
    # Combines FCOS-style detection with per-instance mask prediction:
    # - timm backbone (ResNet-50 by default)
    # - Feature Pyramid Network (FPN)
    # - Detection head for box prediction
    # - Mask head for instance masks
    model = InstanceSegmentor(
        backbone="resnet50",  # Any timm backbone works
        num_classes=80,  # COCO has 80 classes
        metrics=metric_configs,
        logging_config=LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=True,
        ),
        # Architecture options
        fpn_channels=256,
        head_num_convs=4,
        mask_size=28,  # ROI mask resolution
        roi_pool_size=14,  # ROI pooling output size
        # Loss weights
        focal_alpha=0.25,
        focal_gamma=2.0,
        cls_loss_weight=1.0,
        reg_loss_weight=1.0,
        centerness_loss_weight=1.0,
        mask_loss_weight=1.0,  # Weight for mask prediction loss
        # Inference settings
        score_thresh=0.05,
        nms_thresh=0.5,
        max_detections_per_image=100,
        mask_threshold=0.5,  # Threshold for binarizing masks
        # Optimizer settings
        lr=1e-4,
        weight_decay=1e-4,
        optimizer="adamw",
        scheduler="cosine",
    )

    # 4. Train
    trainer = AutoTrainer(
        max_epochs=12,  # Standard COCO training schedule
        accelerator="auto",
        devices=1,
        precision="16-mixed",  # Use mixed precision
        gradient_clip_val=1.0,  # Clip gradients to prevent instability
        logger=[
            LoggerConfig(
                backend="tensorboard", params={"save_dir": "logs/instance_seg"}
            ),
        ],
        checkpoint_monitor="val/mask_mAP",
        checkpoint_mode="max",
    )

    # Train the model
    trainer.fit(model, datamodule=data)

    # Test the model
    results = trainer.test(model, datamodule=data)
    print(f"Test mask mAP: {results[0]['test/mask_mAP']:.4f}")
    print(f"Test bbox mAP: {results[0]['test/bbox_mAP']:.4f}")


if __name__ == "__main__":
    main()
