"""Transformer-based object detection with vision transformer backbones.

This example demonstrates:
- Using Vision Transformers (ViT, Swin, DeiT) as detection backbones
- FCOS-style anchor-free detection with transformer features
- Configuration tips for transformer-based detectors
- Performance vs accuracy trade-offs across different transformers
"""

from autotimm import (
    AutoTrainer,
    DetectionDataModule,
    LoggerConfig,
    MetricConfig,
    ObjectDetector,
)


def main():
    # Data - COCO format detection dataset
    data = DetectionDataModule(
        data_dir="./coco",
        image_size=640,
        batch_size=8,  # Smaller batch size for transformers (higher memory usage)
        num_workers=4,
        augmentation_preset="default",
    )

    # Metrics - MeanAveragePrecision for object detection
    metric_configs = [
        MetricConfig(
            name="mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "bbox"},
            stages=["val", "test"],
            prog_bar=True,
        ),
    ]

    # ========================================================================
    # Option 1: Vision Transformer (ViT) - Best for high accuracy
    # ========================================================================
    print("=" * 60)
    print("Option 1: Vision Transformer (ViT) Backbone")
    print("=" * 60)

    model_vit = ObjectDetector(
        backbone="vit_base_patch16_224",  # ViT-B/16
        num_classes=80,  # COCO has 80 classes
        metrics=metric_configs,
        fpn_channels=256,
        head_num_convs=4,
        lr=1e-4,
        weight_decay=1e-4,
        scheduler="cosine",
        freeze_backbone=False,  # Fine-tune entire model
    )

    # ViT characteristics:
    # - Highest accuracy potential
    # - Larger memory footprint
    # - Slower inference than CNNs
    # - Best with pretrained weights

    # ========================================================================
    # Option 2: Swin Transformer - Balanced efficiency and accuracy
    # ========================================================================
    print("=" * 60)
    print("Option 2: Swin Transformer Backbone")
    print("=" * 60)

    model_swin = ObjectDetector(
        backbone="swin_tiny_patch4_window7_224",  # Swin-T
        num_classes=80,
        metrics=metric_configs,
        fpn_channels=256,
        head_num_convs=4,
        lr=1e-4,
        weight_decay=1e-4,
        scheduler="multistep",
        scheduler_kwargs={"milestones": [8, 11], "gamma": 0.1},
    )

    # Swin characteristics:
    # - Hierarchical feature maps (natural fit for FPN)
    # - More efficient than ViT
    # - Better for multi-scale detection
    # - Excellent accuracy/speed trade-off

    # ========================================================================
    # Option 3: DeiT - Data-efficient training
    # ========================================================================
    print("=" * 60)
    print("Option 3: DeiT (Data-Efficient Image Transformer)")
    print("=" * 60)

    model_deit = ObjectDetector(
        backbone="deit_base_patch16_224",  # DeiT-B/16
        num_classes=80,
        metrics=metric_configs,
        fpn_channels=256,
        head_num_convs=4,
        lr=1e-4,
        weight_decay=1e-4,
        scheduler="cosine",
    )

    # DeiT characteristics:
    # - Similar to ViT but trained with distillation
    # - Good for smaller datasets
    # - Strong pretrained weights
    # - Comparable accuracy to ViT

    # ========================================================================
    # Option 4: Two-phase training (recommended for transformers)
    # ========================================================================
    print("=" * 60)
    print("Option 4: Two-Phase Transformer Training")
    print("=" * 60)

    # Phase 1: Freeze backbone, train FPN and detection head
    model_twophase = ObjectDetector(
        backbone="swin_base_patch4_window7_224",  # Swin-B
        num_classes=80,
        metrics=metric_configs,
        fpn_channels=256,
        head_num_convs=4,
        freeze_backbone=True,  # Freeze during phase 1
        lr=1e-3,  # Higher LR for detection head
        weight_decay=1e-4,
        scheduler="cosine",
    )

    trainer_phase1 = AutoTrainer(
        max_epochs=5,
        logger=[
            LoggerConfig(backend="tensorboard", params={"save_dir": "logs/phase1"}),
        ],
        checkpoint_monitor="val/map",
        checkpoint_mode="max",
        gradient_clip_val=1.0,
    )

    print("\nPhase 1: Training FPN and detection head with frozen backbone...")
    trainer_phase1.fit(model_twophase, datamodule=data)

    # Phase 2: Unfreeze backbone and fine-tune entire model
    for param in model_twophase.backbone.parameters():
        param.requires_grad = True

    model_twophase._lr = 1e-5  # Much lower LR for fine-tuning
    model_twophase.freeze_backbone = False

    trainer_phase2 = AutoTrainer(
        max_epochs=15,
        logger=[
            LoggerConfig(backend="tensorboard", params={"save_dir": "logs/phase2"}),
        ],
        checkpoint_monitor="val/map",
        checkpoint_mode="max",
        gradient_clip_val=1.0,
    )

    print("\nPhase 2: Fine-tuning entire model with lower learning rate...")
    trainer_phase2.fit(model_twophase, datamodule=data)

    # Test final model
    trainer_phase2.test(model_twophase, datamodule=data)

    # ========================================================================
    # Backbone Performance Comparison
    # ========================================================================
    print("\n" + "=" * 60)
    print("Transformer Backbone Comparison")
    print("=" * 60)
    print("""
┌─────────────────────────────┬─────────┬──────────┬─────────────┐
│ Backbone                    │ Speed   │ Accuracy │ Memory      │
├─────────────────────────────┼─────────┼──────────┼─────────────┤
│ vit_tiny_patch16_224        │ Medium  │ Good     │ Low         │
│ vit_base_patch16_224        │ Slow    │ Best     │ High        │
│ swin_tiny_patch4_window7    │ Fast    │ Good     │ Medium      │
│ swin_base_patch4_window7    │ Medium  │ Better   │ Medium-High │
│ deit_small_patch16_224      │ Medium  │ Good     │ Low-Medium  │
│ deit_base_patch16_224       │ Slow    │ Best     │ High        │
└─────────────────────────────┴─────────┴──────────┴─────────────┘

Recommendations:
- Quick experiments: swin_tiny_patch4_window7_224
- Best accuracy: vit_base_patch16_224 or swin_base_patch4_window7_224
- Balanced: swin_tiny or swin_small
- Limited data: deit variants (trained with distillation)

Tips:
- Use smaller batch sizes (8-16) due to higher memory usage
- Two-phase training works very well with transformers
- Gradient clipping (1.0) is important for stable training
- Lower learning rates (1e-4 to 1e-5) work better than CNNs
- Longer warmup can help (first 1-2 epochs)
""")


if __name__ == "__main__":
    main()
