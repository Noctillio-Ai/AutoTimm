"""Example: Object Detection with Hugging Face Hub Models.

This example demonstrates using HF Hub models as backbones for object detection
with FCOS-style anchor-free detection heads.

Usage:
    python examples/hf_hub_object_detection.py
"""

from __future__ import annotations

import autotimm as at  # recommended alias
from autotimm import (
    AutoTrainer,
    DetectionDataModule,
    MetricConfig,
    ObjectDetector,
    list_hf_hub_backbones,
)


def discover_detection_backbones():
    """Discover suitable backbones for object detection on HF Hub."""
    print("=" * 80)
    print("Discovering Object Detection Backbones on HF Hub")
    print("=" * 80)

    # ResNet models (proven for detection)
    print("\n1. ResNet models (proven choice):")
    resnets = list_hf_hub_backbones(model_name="resnet", limit=5)
    for model in resnets:
        print(f"   {model}")

    # ResNeXt models (stronger features)
    print("\n2. ResNeXt models (stronger features):")
    resnexts = list_hf_hub_backbones(model_name="resnext", limit=5)
    for model in resnexts:
        print(f"   {model}")

    # EfficientNet models (efficient)
    print("\n3. EfficientNet models (efficient):")
    efficientnets = list_hf_hub_backbones(model_name="efficientnet", limit=5)
    for model in efficientnets:
        print(f"   {model}")

    # ConvNeXt models (modern)
    print("\n4. ConvNeXt models (modern):")
    convnexts = list_hf_hub_backbones(model_name="convnext", limit=5)
    for model in convnexts:
        print(f"   {model}")


def analyze_feature_pyramid_capabilities():
    """Analyze multi-scale feature extraction for detection."""
    print("\n" + "=" * 80)
    print("Analyzing Feature Pyramid Network Capabilities")
    print("=" * 80)

    models_to_analyze = [
        "hf-hub:timm/resnet50.a1_in1k",
        "hf-hub:timm/resnet101.a1_in1k",
        "hf-hub:timm/resnext50_32x4d.a1_in1k",
        "hf-hub:timm/convnext_tiny.fb_in22k",
        "hf-hub:timm/efficientnet_b3.ra2_in1k",
    ]

    print(f"\n{'Model':<45} {'Params':>12} {'FPN Levels':>12} {'Channels':<25}")
    print("-" * 110)

    for model_name in models_to_analyze:
        try:
            backbone = at.create_feature_backbone(model_name)
            params = at.count_parameters(backbone, trainable_only=False)
            channels = at.get_feature_channels(backbone)
            at.get_feature_strides(backbone)

            short_name = model_name.replace("hf-hub:timm/", "")
            channels_str = str(channels[-4:])  # Last 4 levels for FPN
            print(
                f"{short_name:<45} {params:>12,} {len(channels):>12} {channels_str:<25}"
            )
        except Exception as e:
            print(f"{model_name:<45} Error: {e}")

    print("\nNote: FPN typically uses the last 4 feature levels (C2-C5)")
    print("      Higher channel counts enable better feature representation")


def train_detector_with_resnet():
    """Train FCOS detector with ResNet from HF Hub."""
    print("\n" + "=" * 80)
    print("Training FCOS Detector with ResNet50 from HF Hub")
    print("=" * 80)

    # Data setup
    DetectionDataModule(
        data_dir="./coco",  # Your COCO dataset path
        image_size=640,
        batch_size=4,
        num_workers=4,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "bbox"},
            stages=["val", "test"],
            prog_bar=True,
        ),
    ]

    # Model using HF Hub ResNet50 backbone
    print("\nCreating FCOS detector with HF Hub ResNet50...")
    model = ObjectDetector(
        backbone="hf-hub:timm/resnet50.a1_in1k",
        num_classes=80,
        fpn_channels=256,
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
        accumulate_grad_batches=4,  # Simulate larger batch
        logger_configs=[],
    )

    print("\nNote: This is a demonstration. Update data_dir to run training.")
    # Uncomment to run actual training:
    # trainer.fit(model, datamodule=data)
    # trainer.test(model, datamodule=data)


def train_detector_with_resnext():
    """Train detector with stronger ResNeXt backbone."""
    print("\n" + "=" * 80)
    print("Training Detector with ResNeXt from HF Hub")
    print("=" * 80)

    # Data setup
    DetectionDataModule(
        data_dir="./coco",
        image_size=800,  # Higher resolution for better accuracy
        batch_size=2,
        num_workers=4,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "bbox"},
            stages=["val", "test"],
            prog_bar=True,
        ),
    ]

    # Model using HF Hub ResNeXt50 backbone
    print("\nCreating detector with HF Hub ResNeXt50...")
    model = ObjectDetector(
        backbone="hf-hub:timm/resnext50_32x4d.a1_in1k",
        num_classes=80,
        fpn_channels=256,
        metrics=metrics,
        lr=1e-3,
        optimizer="adamw",
        weight_decay=1e-4,
    )

    print(f"Total parameters: {at.count_parameters(model):,}")
    print("ResNeXt provides stronger features than ResNet")

    # Trainer
    AutoTrainer(
        max_epochs=100,
        accelerator="auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=8,
        logger_configs=[],
    )

    print("\nNote: This is a demonstration. Update data_dir to run training.")
    # Uncomment to run actual training:
    # trainer.fit(model, datamodule=data)
    # trainer.test(model, datamodule=data)


def train_efficient_detector():
    """Train efficient detector for fast inference."""
    print("\n" + "=" * 80)
    print("Training Efficient Detector with EfficientNet from HF Hub")
    print("=" * 80)

    # Data setup
    DetectionDataModule(
        data_dir="./coco",
        image_size=512,  # Smaller for efficiency
        batch_size=8,
        num_workers=4,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "bbox"},
            stages=["val", "test"],
            prog_bar=True,
        ),
    ]

    # Model using HF Hub EfficientNet backbone
    print("\nCreating detector with HF Hub EfficientNet-B2...")
    model = ObjectDetector(
        backbone="hf-hub:timm/efficientnet_b2.ra_in1k",
        num_classes=80,
        fpn_channels=128,  # Smaller FPN for efficiency
        metrics=metrics,
        lr=1e-3,
        optimizer="adamw",
        weight_decay=1e-4,
    )

    print(f"Total parameters: {at.count_parameters(model):,}")
    print("Optimized for fast inference and lower memory usage")

    # Trainer
    AutoTrainer(
        max_epochs=120,
        accelerator="auto",
        precision="16-mixed",
        logger_configs=[],
    )

    print("\nNote: This is a demonstration. Update data_dir to run training.")
    # Uncomment to run actual training:
    # trainer.fit(model, datamodule=data)
    # trainer.test(model, datamodule=data)


def train_modern_detector():
    """Train detector with modern ConvNeXt backbone."""
    print("\n" + "=" * 80)
    print("Training Modern Detector with ConvNeXt from HF Hub")
    print("=" * 80)

    # Data setup
    DetectionDataModule(
        data_dir="./coco",
        image_size=640,
        batch_size=4,
        num_workers=4,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "bbox"},
            stages=["val", "test"],
            prog_bar=True,
        ),
    ]

    # Model using HF Hub ConvNeXt backbone
    print("\nCreating detector with HF Hub ConvNeXt...")
    model = ObjectDetector(
        backbone="hf-hub:timm/convnext_tiny.fb_in22k",
        num_classes=80,
        fpn_channels=256,
        metrics=metrics,
        lr=5e-4,  # Lower LR for modern architecture
        optimizer="adamw",
        weight_decay=0.05,  # Higher weight decay
    )

    print(f"Total parameters: {at.count_parameters(model):,}")
    print("ConvNeXt: Modern CNN with transformer-like design")

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


def demonstrate_backbone_selection():
    """Guide for selecting the right backbone for object detection."""
    print("\n" + "=" * 80)
    print("Backbone Selection Guide for Object Detection")
    print("=" * 80)

    configurations = [
        {
            "name": "High Accuracy",
            "backbone": "hf-hub:timm/resnext101_32x8d.fb_wsl_ig1b_ft_in1k",
            "image_size": 1024,
            "batch_size": 1,
            "fpn_channels": 256,
            "use_case": "Research, offline processing, high mAP required",
        },
        {
            "name": "Balanced",
            "backbone": "hf-hub:timm/resnet50.a1_in1k",
            "image_size": 640,
            "batch_size": 4,
            "fpn_channels": 256,
            "use_case": "Production, good balance of speed and accuracy",
        },
        {
            "name": "Fast Inference",
            "backbone": "hf-hub:timm/efficientnet_b0.ra_in1k",
            "image_size": 512,
            "batch_size": 8,
            "fpn_channels": 128,
            "use_case": "Real-time applications, edge devices",
        },
        {
            "name": "Modern Architecture",
            "backbone": "hf-hub:timm/convnext_small.fb_in22k_ft_in1k",
            "image_size": 640,
            "batch_size": 4,
            "fpn_channels": 256,
            "use_case": "State-of-the-art performance with modern CNN",
        },
        {
            "name": "Resource Constrained",
            "backbone": "hf-hub:timm/mobilenetv3_large_100.ra_in1k",
            "image_size": 416,
            "batch_size": 16,
            "fpn_channels": 128,
            "use_case": "Mobile, embedded systems, low memory",
        },
    ]

    print("\nRecommended configurations:\n")
    for config in configurations:
        print(f"{config['name']}:")
        print(f"  Backbone: {config['backbone']}")
        print(f"  Image Size: {config['image_size']}x{config['image_size']}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  FPN Channels: {config['fpn_channels']}")
        print(f"  Use Case: {config['use_case']}")
        print()


def compare_detection_backbones():
    """Compare computational requirements of different backbones."""
    print("\n" + "=" * 80)
    print("Computational Comparison for Detection")
    print("=" * 80)

    print("\nCreating detectors with different backbones...\n")

    backbones = [
        ("MobileNetV3", "hf-hub:timm/mobilenetv3_large_100.ra_in1k"),
        ("ResNet50", "hf-hub:timm/resnet50.a1_in1k"),
        ("ResNeXt50", "hf-hub:timm/resnext50_32x4d.a1_in1k"),
        ("ConvNeXt-Tiny", "hf-hub:timm/convnext_tiny.fb_in22k"),
    ]

    print(f"{'Backbone':<20} {'Parameters':>15} {'Relative Size':>15}")
    print("-" * 50)

    base_params = None
    for name, backbone_name in backbones:
        try:
            model = ObjectDetector(
                backbone=backbone_name,
                num_classes=80,
                fpn_channels=256,
            )
            params = at.count_parameters(model, trainable_only=False)

            if base_params is None:
                base_params = params

            relative = f"{params / base_params:.2f}x"
            print(f"{name:<20} {params:>15,} {relative:>15}")
        except Exception as e:
            print(f"{name:<20} Error: {e}")

    print("\nNote: Smaller models are faster but may have lower accuracy")
    print("      Choose based on your speed/accuracy requirements")


def main():
    """Run all object detection examples."""
    print("\n" + "=" * 80)
    print("HF Hub Object Detection Examples")
    print("=" * 80)

    # Example 1: Discover models
    discover_detection_backbones()

    # Example 2: Analyze FPN capabilities
    analyze_feature_pyramid_capabilities()

    # Example 3: Compare backbones
    compare_detection_backbones()

    # Example 4: Selection guide
    demonstrate_backbone_selection()

    # Example 5: Training examples (optional - uncomment to run)
    # Note: These require actual COCO dataset
    # train_detector_with_resnet()
    # train_detector_with_resnext()
    # train_efficient_detector()
    # train_modern_detector()

    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. ResNet50 is a solid baseline for object detection")
    print("  2. ResNeXt provides better features at the cost of speed")
    print("  3. EfficientNet offers good efficiency for real-time use")
    print("  4. ConvNeXt brings modern architecture benefits")
    print("  5. MobileNet is ideal for edge and mobile deployment")
    print("  6. All HF Hub backbones work seamlessly with FCOS detector")


if __name__ == "__main__":
    main()
