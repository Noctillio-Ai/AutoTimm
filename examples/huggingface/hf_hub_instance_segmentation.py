"""Example: Instance Segmentation with Hugging Face Hub Models.

This example demonstrates using HF Hub models as backbones for instance
segmentation with Mask R-CNN style architecture.

Usage:
    python examples/hf_hub_instance_segmentation.py
"""

from __future__ import annotations

import autotimm as at  # recommended alias
from autotimm import (
    AutoTrainer,
    InstanceSegmentationDataModule,
    InstanceSegmentor,
    MetricConfig,
    list_hf_hub_backbones,
)


def discover_instance_segmentation_backbones():
    """Discover suitable backbones for instance segmentation on HF Hub."""
    print("=" * 80)
    print("Discovering Instance Segmentation Backbones on HF Hub")
    print("=" * 80)

    # ResNet models (proven for Mask R-CNN)
    print("\n1. ResNet models (Mask R-CNN standard):")
    resnets = list_hf_hub_backbones(model_name="resnet", limit=5)
    for model in resnets:
        print(f"   {model}")

    # ResNeXt models (stronger features)
    print("\n2. ResNeXt models (improved features):")
    resnexts = list_hf_hub_backbones(model_name="resnext", limit=5)
    for model in resnexts:
        print(f"   {model}")

    # ConvNeXt models (modern)
    print("\n3. ConvNeXt models (modern architecture):")
    convnexts = list_hf_hub_backbones(model_name="convnext", limit=5)
    for model in convnexts:
        print(f"   {model}")


def analyze_mask_head_compatibility():
    """Analyze backbone compatibility with mask head."""
    print("\n" + "=" * 80)
    print("Analyzing Backbone Features for Mask Head")
    print("=" * 80)

    backbones = [
        "hf-hub:timm/resnet50.a1_in1k",
        "hf-hub:timm/resnet101.a1_in1k",
        "hf-hub:timm/resnext50_32x4d.a1_in1k",
        "hf-hub:timm/convnext_tiny.fb_in22k",
    ]

    print(
        f"\n{'Backbone':<40} {'Params':>12} {'FPN Channels':<30} {'Suitable for':>20}"
    )
    print("-" * 105)

    for backbone_name in backbones:
        try:
            feature_backbone = at.create_feature_backbone(backbone_name)
            params = at.count_parameters(feature_backbone, trainable_only=False)
            channels = at.get_feature_channels(feature_backbone)

            short_name = backbone_name.replace("hf-hub:timm/", "")
            channels_str = str(channels[-4:])  # Last 4 levels

            # Determine suitability
            if params < 30_000_000:
                suitability = "Fast inference"
            elif params < 50_000_000:
                suitability = "Balanced"
            else:
                suitability = "High accuracy"

            print(
                f"{short_name:<40} {params:>12,} {channels_str:<30} {suitability:>20}"
            )
        except Exception as e:
            print(f"{backbone_name:<40} Error: {e}")

    print("\nNote: Instance segmentation requires multi-scale features from FPN")
    print("      Higher capacity backbones generally produce better masks")


def train_instance_segmentor_resnet():
    """Train instance segmentation model with ResNet from HF Hub."""
    print("\n" + "=" * 80)
    print("Training Instance Segmentor with ResNet50 from HF Hub")
    print("=" * 80)

    # Data setup
    InstanceSegmentationDataModule(
        data_dir="./coco",  # Your COCO dataset path
        image_size=640,
        batch_size=2,
        num_workers=4,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="mask_mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "segm"},
            stages=["val", "test"],
            prog_bar=True,
        ),
        MetricConfig(
            name="bbox_mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "bbox"},
            stages=["val", "test"],
            prog_bar=False,
        ),
    ]

    # Model using HF Hub ResNet50 backbone
    print("\nCreating instance segmentor with HF Hub ResNet50...")
    model = InstanceSegmentor(
        backbone="hf-hub:timm/resnet50.a1_in1k",
        num_classes=80,
        fpn_channels=256,
        mask_loss_weight=1.0,
        metrics=metrics,
        lr=1e-3,
        optimizer="adamw",
        weight_decay=1e-4,
    )

    print(f"Total parameters: {at.count_parameters(model):,}")

    # Trainer
    AutoTrainer(
        max_epochs=100,
        accelerator="auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=8,  # Simulate larger batch
        logger_configs=[],
    )

    print("\nNote: This is a demonstration. Update data_dir to run training.")
    # Uncomment to run actual training:
    # trainer.fit(model, datamodule=data)
    # trainer.test(model, datamodule=data)


def train_instance_segmentor_resnext():
    """Train instance segmentation with stronger ResNeXt backbone."""
    print("\n" + "=" * 80)
    print("Training Instance Segmentor with ResNeXt from HF Hub")
    print("=" * 80)

    # Data setup
    InstanceSegmentationDataModule(
        data_dir="./coco",
        image_size=800,  # Higher resolution
        batch_size=1,  # Smaller batch for memory
        num_workers=4,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="mask_mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "segm"},
            stages=["val", "test"],
            prog_bar=True,
        ),
    ]

    # Model using HF Hub ResNeXt50 backbone
    print("\nCreating instance segmentor with HF Hub ResNeXt50...")
    model = InstanceSegmentor(
        backbone="hf-hub:timm/resnext50_32x4d.a1_in1k",
        num_classes=80,
        fpn_channels=256,
        mask_loss_weight=1.0,
        metrics=metrics,
        lr=1e-3,
        optimizer="adamw",
        weight_decay=1e-4,
    )

    print(f"Total parameters: {at.count_parameters(model):,}")
    print("ResNeXt provides stronger features for better mask quality")

    # Trainer
    AutoTrainer(
        max_epochs=100,
        accelerator="auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=16,
        logger_configs=[],
    )

    print("\nNote: This is a demonstration. Update data_dir to run training.")
    # Uncomment to run actual training:
    # trainer.fit(model, datamodule=data)
    # trainer.test(model, datamodule=data)


def train_modern_instance_segmentor():
    """Train instance segmentation with modern ConvNeXt backbone."""
    print("\n" + "=" * 80)
    print("Training Modern Instance Segmentor with ConvNeXt from HF Hub")
    print("=" * 80)

    # Data setup
    InstanceSegmentationDataModule(
        data_dir="./coco",
        image_size=640,
        batch_size=2,
        num_workers=4,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="mask_mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "segm"},
            stages=["val", "test"],
            prog_bar=True,
        ),
    ]

    # Model using HF Hub ConvNeXt backbone
    print("\nCreating instance segmentor with HF Hub ConvNeXt...")
    model = InstanceSegmentor(
        backbone="hf-hub:timm/convnext_tiny.fb_in22k",
        num_classes=80,
        fpn_channels=256,
        mask_loss_weight=1.0,
        metrics=metrics,
        lr=5e-4,  # Lower LR for modern architecture
        optimizer="adamw",
        weight_decay=0.05,
    )

    print(f"Total parameters: {at.count_parameters(model):,}")
    print("ConvNeXt: State-of-the-art CNN with modern design principles")

    # Trainer
    AutoTrainer(
        max_epochs=100,
        accelerator="auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        logger_configs=[],
    )

    print("\nNote: This is a demonstration. Update data_dir to run training.")
    # Uncomment to run actual training:
    # trainer.fit(model, datamodule=data)
    # trainer.test(model, datamodule=data)


def demonstrate_configuration_options():
    """Show different configuration options for instance segmentation."""
    print("\n" + "=" * 80)
    print("Instance Segmentation Configuration Options")
    print("=" * 80)

    configs = [
        {
            "name": "High Accuracy",
            "backbone": "hf-hub:timm/resnext101_32x8d.fb_wsl_ig1b_ft_in1k",
            "image_size": 1024,
            "fpn_channels": 256,
            "mask_channels": 256,
            "batch_size": 1,
            "description": "Best mask quality, research setting",
        },
        {
            "name": "Balanced",
            "backbone": "hf-hub:timm/resnet50.a1_in1k",
            "image_size": 640,
            "fpn_channels": 256,
            "mask_channels": 256,
            "batch_size": 2,
            "description": "Production-ready, good trade-off",
        },
        {
            "name": "Fast Inference",
            "backbone": "hf-hub:timm/resnet18.a1_in1k",
            "image_size": 512,
            "fpn_channels": 128,
            "mask_channels": 128,
            "batch_size": 4,
            "description": "Faster but lower mask quality",
        },
        {
            "name": "Modern",
            "backbone": "hf-hub:timm/convnext_small.fb_in22k_ft_in1k",
            "image_size": 640,
            "fpn_channels": 256,
            "mask_channels": 256,
            "batch_size": 2,
            "description": "State-of-the-art architecture",
        },
    ]

    print("\nRecommended configurations:\n")
    for config in configs:
        print(f"{config['name']}:")
        print(f"  Backbone: {config['backbone']}")
        print(f"  Image Size: {config['image_size']}x{config['image_size']}")
        print(f"  FPN Channels: {config['fpn_channels']}")
        print(f"  Mask Channels: {config['mask_channels']}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Use Case: {config['description']}")
        print()


def compare_computational_requirements():
    """Compare computational requirements of different backbones."""
    print("\n" + "=" * 80)
    print("Computational Requirements Comparison")
    print("=" * 80)

    print("\nCreating instance segmentors with different backbones...\n")

    backbones = [
        ("ResNet18", "hf-hub:timm/resnet18.a1_in1k"),
        ("ResNet50", "hf-hub:timm/resnet50.a1_in1k"),
        ("ResNet101", "hf-hub:timm/resnet101.a1_in1k"),
        ("ResNeXt50", "hf-hub:timm/resnext50_32x4d.a1_in1k"),
        ("ConvNeXt-Tiny", "hf-hub:timm/convnext_tiny.fb_in22k"),
    ]

    print(f"{'Backbone':<20} {'Parameters':>15} {'Relative Size':>15} {'Speed':>15}")
    print("-" * 70)

    base_params = None
    for name, backbone_name in backbones:
        try:
            model = InstanceSegmentor(
                backbone=backbone_name,
                num_classes=80,
                fpn_channels=256,
            )
            params = at.count_parameters(model, trainable_only=False)

            if base_params is None:
                base_params = params

            relative = f"{params / base_params:.2f}x"

            # Estimate speed category
            if params < 40_000_000:
                speed = "Fast"
            elif params < 60_000_000:
                speed = "Medium"
            else:
                speed = "Slow"

            print(f"{name:<20} {params:>15,} {relative:>15} {speed:>15}")
        except Exception as e:
            print(f"{name:<20} Error: {e}")

    print("\nNote: Instance segmentation is compute-intensive")
    print("      Balance accuracy needs with available hardware")


def provide_training_tips():
    """Provide tips for training instance segmentation models."""
    print("\n" + "=" * 80)
    print("Training Tips for Instance Segmentation")
    print("=" * 80)

    tips = [
        {
            "category": "Data",
            "tips": [
                "Use at least 1000 annotated instances per class",
                "Augment with horizontal flips, scales, and color jitter",
                "Ensure mask annotations are high quality",
                "Balance classes if highly imbalanced",
            ],
        },
        {
            "category": "Hyperparameters",
            "tips": [
                "Start with lr=1e-3 for ResNet, lr=5e-4 for ConvNeXt",
                "Use weight_decay=1e-4 for ResNet, 0.05 for modern architectures",
                "Set mask_loss_weight=1.0 for equal bbox and mask importance",
                "Gradient clipping at 1.0 helps stability",
            ],
        },
        {
            "category": "Training Strategy",
            "tips": [
                "Train for 100+ epochs on COCO-sized datasets",
                "Use gradient accumulation if GPU memory limited",
                "Monitor both bbox mAP and mask mAP",
                "Consider two-stage: detection first, then add masks",
            ],
        },
        {
            "category": "Model Selection",
            "tips": [
                "ResNet50: reliable baseline, widely used",
                "ResNeXt50: better features, slightly slower",
                "ConvNeXt: modern choice, excellent performance",
                "ResNet101: higher capacity for complex scenes",
            ],
        },
    ]

    for tip_group in tips:
        print(f"\n{tip_group['category']}:")
        for tip in tip_group["tips"]:
            print(f"  • {tip}")


def main():
    """Run all instance segmentation examples."""
    print("\n" + "=" * 80)
    print("HF Hub Instance Segmentation Examples")
    print("=" * 80)

    # Example 1: Discover models
    discover_instance_segmentation_backbones()

    # Example 2: Analyze compatibility
    analyze_mask_head_compatibility()

    # Example 3: Configuration options
    demonstrate_configuration_options()

    # Example 4: Computational comparison
    compare_computational_requirements()

    # Example 5: Training tips
    provide_training_tips()

    # Example 6: Training examples (optional - uncomment to run)
    # Note: These require actual COCO instance segmentation dataset
    # train_instance_segmentor_resnet()
    # train_instance_segmentor_resnext()
    # train_modern_instance_segmentor()

    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Instance segmentation requires strong backbones")
    print("  2. ResNet50 is a solid baseline for most use cases")
    print("  3. ResNeXt and ConvNeXt offer better features")
    print("  4. Balance accuracy needs with compute constraints")
    print("  5. All HF Hub backbones work seamlessly with mask heads")
    print("  6. Monitor both detection and segmentation metrics")


if __name__ == "__main__":
    main()
