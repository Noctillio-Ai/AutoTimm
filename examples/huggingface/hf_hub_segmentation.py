"""Example: Semantic Segmentation with Hugging Face Hub Models.

This example demonstrates using HF Hub models as backbones for semantic
segmentation tasks with DeepLabV3+ and FCN heads.

Usage:
    python examples/hf_hub_segmentation.py
"""

from __future__ import annotations

import autotimm as at  # recommended alias
from autotimm import (
    AutoTrainer,
    MetricConfig,
    SegmentationDataModule,
    SemanticSegmentor,
    list_hf_hub_backbones,
)


def discover_segmentation_backbones():
    """Discover suitable backbones for segmentation on HF Hub."""
    print("=" * 80)
    print("Discovering Segmentation Backbones on HF Hub")
    print("=" * 80)

    # ResNet models (classic for segmentation)
    print("\n1. ResNet models (classic choice):")
    resnets = list_hf_hub_backbones(model_name="resnet", limit=5)
    for model in resnets:
        print(f"   {model}")

    # ConvNeXt models (modern alternative)
    print("\n2. ConvNeXt models (modern architecture):")
    convnexts = list_hf_hub_backbones(model_name="convnext", limit=5)
    for model in convnexts:
        print(f"   {model}")

    # EfficientNet models (efficient)
    print("\n3. EfficientNet models (efficient):")
    efficientnets = list_hf_hub_backbones(model_name="efficientnet", limit=5)
    for model in efficientnets:
        print(f"   {model}")

    # MobileNet models (lightweight)
    print("\n4. MobileNet models (lightweight):")
    mobilenets = list_hf_hub_backbones(model_name="mobilenet", limit=5)
    for model in mobilenets:
        print(f"   {model}")


def compare_backbone_features():
    """Compare feature extraction capabilities of different backbones."""
    print("\n" + "=" * 80)
    print("Comparing Feature Extraction for Segmentation")
    print("=" * 80)

    models_to_compare = [
        "hf-hub:timm/resnet50.a1_in1k",
        "hf-hub:timm/convnext_tiny.fb_in22k",
        "hf-hub:timm/efficientnet_b2.ra_in1k",
        "hf-hub:timm/mobilenetv3_large_100.ra_in1k",
    ]

    print(f"\n{'Model':<50} {'Stages':>10} {'Channels':<30}")
    print("-" * 90)

    for model_name in models_to_compare:
        try:
            backbone = at.create_feature_backbone(model_name)
            channels = at.get_feature_channels(backbone)
            at.get_feature_strides(backbone)

            short_name = model_name.replace("hf-hub:timm/", "")
            channels_str = str(channels)
            print(f"{short_name:<50} {len(channels):>10} {channels_str:<30}")
        except Exception as e:
            print(f"{model_name:<50} Error: {e}")

    print("\nNote: More channels generally means better feature representation,")
    print("      but also higher memory usage and slower inference.")


def train_deeplabv3_with_resnet():
    """Train DeepLabV3+ segmentation model with ResNet from HF Hub."""
    print("\n" + "=" * 80)
    print("Training DeepLabV3+ with ResNet50 from HF Hub")
    print("=" * 80)

    # Data setup (using PNG mask format)
    # Replace with your actual dataset path
    SegmentationDataModule(
        data_dir="./cityscapes",  # Your dataset path
        format="cityscapes",
        image_size=512,
        batch_size=4,
        num_workers=4,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="iou",
            backend="torchmetrics",
            metric_class="JaccardIndex",
            params={
                "task": "multiclass",
                "num_classes": 19,
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
                "num_classes": 19,
                "average": "micro",
                "ignore_index": 255,
            },
            stages=["val", "test"],
            prog_bar=False,
        ),
    ]

    # Model using HF Hub ResNet50 backbone
    print("\nCreating DeepLabV3+ with HF Hub ResNet50...")
    model = SemanticSegmentor(
        backbone="hf-hub:timm/resnet50.a1_in1k",
        num_classes=19,
        head_type="deeplabv3plus",
        loss_fn="combined_segmentation",  # CrossEntropy + Dice
        dice_weight=1.0,
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
        logger_configs=[],
    )

    print("\nNote: This is a demonstration. Update data_dir to run training.")
    # Uncomment to run actual training:
    # trainer.fit(model, datamodule=data)
    # trainer.test(model, datamodule=data)


def train_fcn_with_convnext():
    """Train FCN segmentation model with ConvNeXt from HF Hub."""
    print("\n" + "=" * 80)
    print("Training FCN with ConvNeXt from HF Hub")
    print("=" * 80)

    # Data setup
    SegmentationDataModule(
        data_dir="./cityscapes",
        format="cityscapes",
        image_size=512,
        batch_size=4,
        num_workers=4,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="iou",
            backend="torchmetrics",
            metric_class="JaccardIndex",
            params={
                "task": "multiclass",
                "num_classes": 19,
                "average": "macro",
                "ignore_index": 255,
            },
            stages=["val", "test"],
            prog_bar=True,
        ),
    ]

    # Model using HF Hub ConvNeXt backbone
    print("\nCreating FCN with HF Hub ConvNeXt...")
    model = SemanticSegmentor(
        backbone="hf-hub:timm/convnext_tiny.fb_in22k",
        num_classes=19,
        head_type="fcn",
        loss_fn="dice",
        metrics=metrics,
        lr=1e-4,  # Lower LR for pretrained ConvNeXt
        optimizer="adamw",
        weight_decay=0.05,
    )

    print(f"Total parameters: {at.count_parameters(model):,}")

    # Trainer
    AutoTrainer(
        max_epochs=100,
        accelerator="auto",
        precision="16-mixed",
        logger_configs=[],
    )

    print("\nNote: This is a demonstration. Update data_dir to run training.")
    # Uncomment to run actual training:
    # trainer.fit(model, datamodule=data)
    # trainer.test(model, datamodule=data)


def train_lightweight_segmentation():
    """Train lightweight segmentation model for edge deployment."""
    print("\n" + "=" * 80)
    print("Training Lightweight Segmentation for Edge Deployment")
    print("=" * 80)

    # Data setup
    SegmentationDataModule(
        data_dir="./cityscapes",
        format="cityscapes",
        image_size=256,  # Smaller for edge
        batch_size=8,
        num_workers=4,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="iou",
            backend="torchmetrics",
            metric_class="JaccardIndex",
            params={
                "task": "multiclass",
                "num_classes": 19,
                "average": "macro",
                "ignore_index": 255,
            },
            stages=["val", "test"],
            prog_bar=True,
        ),
    ]

    # Model using HF Hub MobileNetV3 backbone
    print("\nCreating FCN with HF Hub MobileNetV3...")
    model = SemanticSegmentor(
        backbone="hf-hub:timm/mobilenetv3_large_100.ra_in1k",
        num_classes=19,
        head_type="fcn",
        loss_type="combined",
        metrics=metrics,
        lr=1e-3,
        optimizer="adamw",
    )

    print(f"Total parameters: {at.count_parameters(model):,}")
    print("(Optimized for edge devices and mobile deployment!)")

    # Trainer
    AutoTrainer(
        max_epochs=150,
        accelerator="auto",
        precision="16-mixed",
        logger_configs=[],
    )

    print("\nNote: This is a demonstration. Update data_dir to run training.")
    # Uncomment to run actual training:
    # trainer.fit(model, datamodule=data)
    # trainer.test(model, datamodule=data)


def demonstrate_model_flexibility():
    """Demonstrate using different HF Hub models for different scenarios."""
    print("\n" + "=" * 80)
    print("Model Selection Guide for Semantic Segmentation")
    print("=" * 80)

    scenarios = [
        {
            "name": "High Accuracy (Research)",
            "backbone": "hf-hub:timm/resnet50.fb_swsl_ig1b_ft_in1k",
            "head": "deeplabv3plus",
            "batch_size": 2,
            "image_size": 1024,
            "description": "Best accuracy, slower inference",
        },
        {
            "name": "Balanced (Production)",
            "backbone": "hf-hub:timm/resnet50.a1_in1k",
            "head": "deeplabv3plus",
            "batch_size": 4,
            "image_size": 512,
            "description": "Good balance of speed and accuracy",
        },
        {
            "name": "Fast Inference (Edge)",
            "backbone": "hf-hub:timm/mobilenetv3_large_100.ra_in1k",
            "head": "fcn",
            "batch_size": 8,
            "image_size": 256,
            "description": "Fast, lightweight, mobile-friendly",
        },
        {
            "name": "Modern Architecture",
            "backbone": "hf-hub:timm/convnext_tiny.fb_in22k",
            "head": "deeplabv3plus",
            "batch_size": 4,
            "image_size": 512,
            "description": "State-of-the-art CNN architecture",
        },
    ]

    print("\nRecommended configurations:\n")
    for scenario in scenarios:
        print(f"{scenario['name']}:")
        print(f"  Backbone: {scenario['backbone']}")
        print(f"  Head: {scenario['head']}")
        print(f"  Image Size: {scenario['image_size']}x{scenario['image_size']}")
        print(f"  Batch Size: {scenario['batch_size']}")
        print(f"  Use Case: {scenario['description']}")
        print()


def main():
    """Run all segmentation examples."""
    print("\n" + "=" * 80)
    print("HF Hub Semantic Segmentation Examples")
    print("=" * 80)

    # Example 1: Discover models
    discover_segmentation_backbones()

    # Example 2: Compare features
    compare_backbone_features()

    # Example 3: Model selection guide
    demonstrate_model_flexibility()

    # Example 4: Training examples (optional - uncomment to run)
    # Note: These require actual dataset paths
    # train_deeplabv3_with_resnet()
    # train_fcn_with_convnext()
    # train_lightweight_segmentation()

    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. HF Hub models work seamlessly with segmentation heads")
    print("  2. Choose backbone based on your accuracy/speed requirements")
    print("  3. DeepLabV3+ generally gives better accuracy than FCN")
    print("  4. Use smaller backbones (MobileNet) for edge deployment")
    print("  5. Modern architectures (ConvNeXt) offer great performance")


if __name__ == "__main__":
    main()
