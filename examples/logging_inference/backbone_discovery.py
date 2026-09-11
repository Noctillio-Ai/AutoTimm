"""Explore available timm backbones and transform presets.

This example demonstrates:
- Listing available backbones by architecture family
- Filtering by pretrained availability
- Inspecting model properties (parameters, features)
- Comparing different model configurations
- Discovering available transform presets

Usage:
    python examples/backbone_discovery.py
"""

import autotimm as at  # recommended alias
from autotimm import BackboneConfig, list_transform_presets


def main():
    print("=" * 70)
    print("AutoTimm Backbone Discovery")
    print("=" * 70)

    # ========================================================================
    # List all available backbones
    # ========================================================================
    all_models = at.list_backbones()
    print(f"\nTotal available backbones: {len(all_models)}")

    # ========================================================================
    # Filter by architecture family
    # ========================================================================
    print("\n" + "-" * 70)
    print("Architecture Families")
    print("-" * 70)

    families = {
        "ResNet": "resnet*",
        "EfficientNet": "efficientnet*",
        "Vision Transformer": "vit_*",
        "ConvNeXt": "convnext*",
        "MobileNet": "*mobilenet*",
        "RegNet": "regnet*",
        "DenseNet": "densenet*",
        "Swin Transformer": "swin_*",
        "MaxViT": "maxvit*",
        "CoAtNet": "coatnet*",
    }

    for name, pattern in families.items():
        models = at.list_backbones(pattern)
        pretrained = at.list_backbones(pattern, pretrained_only=True)
        print(f"{name:.<30} {len(models):>5} total, {len(pretrained):>5} pretrained")

    # ========================================================================
    # Show sample models for each family
    # ========================================================================
    print("\n" + "-" * 70)
    print("Sample Models by Family")
    print("-" * 70)

    for name, pattern in list(families.items())[:4]:  # First 4 families
        models = at.list_backbones(pattern, pretrained_only=True)
        print(f"\n{name}:")
        for m in models[:5]:
            print(f"  - {m}")
        if len(models) > 5:
            print(f"  ... and {len(models) - 5} more")

    # ========================================================================
    # Inspect specific backbones
    # ========================================================================
    print("\n" + "-" * 70)
    print("Model Inspection")
    print("-" * 70)

    models_to_inspect = [
        "resnet18",
        "resnet50",
        "efficientnet_b0",
        "efficientnet_b3",
        "vit_base_patch16_224",
        "convnext_tiny",
        "swin_tiny_patch4_window7_224",
    ]

    print(f"\n{'Model':<35} {'Params (M)':>12} {'Features':>10}")
    print("-" * 60)

    for model_name in models_to_inspect:
        try:
            backbone = at.create_backbone(model_name)
            params = at.count_parameters(backbone, trainable_only=False)
            print(f"{model_name:<35} {params / 1e6:>12.2f} {backbone.num_features:>10}")
        except Exception as e:
            print(f"{model_name:<35} Error: {e}")

    # ========================================================================
    # Advanced backbone configuration
    # ========================================================================
    print("\n" + "-" * 70)
    print("Advanced Backbone Configuration")
    print("-" * 70)

    # Standard backbone
    print("\n1. Standard backbone (pretrained):")
    backbone = at.create_backbone("resnet50")
    print(f"   Output features: {backbone.num_features}")
    print(f"   Total params: {at.count_parameters(backbone):,}")

    # With BackboneConfig for more control
    print("\n2. BackboneConfig with drop path:")
    config = BackboneConfig(
        model_name="vit_base_patch16_224",
        pretrained=True,
        drop_path_rate=0.1,  # Stochastic depth
    )
    backbone = at.create_backbone(config)
    print(f"   Model: {config.model_name}")
    print(f"   Drop path rate: {config.drop_path_rate}")
    print(f"   Output features: {backbone.num_features}")
    print(f"   Total params: {at.count_parameters(backbone):,}")

    # With dropout
    print("\n3. BackboneConfig with dropout:")
    config = BackboneConfig(
        model_name="efficientnet_b0",
        pretrained=True,
        drop_rate=0.2,  # Classifier dropout
    )
    backbone = at.create_backbone(config)
    print(f"   Model: {config.model_name}")
    print(f"   Drop rate: {config.drop_rate}")
    print(f"   Output features: {backbone.num_features}")

    # Without pretrained weights
    print("\n4. Training from scratch (no pretrained):")
    config = BackboneConfig(
        model_name="resnet18",
        pretrained=False,
    )
    backbone = at.create_backbone(config)
    print(f"   Model: {config.model_name}")
    print(f"   Pretrained: {config.pretrained}")
    print(f"   Output features: {backbone.num_features}")

    # ========================================================================
    # Search for specific model variants
    # ========================================================================
    print("\n" + "-" * 70)
    print("Searching for Model Variants")
    print("-" * 70)

    # Find all ImageNet-21k pretrained models
    print("\nImageNet-21k pretrained ViTs:")
    vit_21k = at.list_backbones("vit_*in21k*", pretrained_only=True)
    for m in vit_21k[:5]:
        print(f"  - {m}")

    # Find CLIP models
    print("\nCLIP models:")
    clip_models = at.list_backbones("*clip*", pretrained_only=True)
    for m in clip_models[:5]:
        print(f"  - {m}")

    # Find models by size
    print("\nSmall models (good for edge deployment):")
    small_models = [
        "mobilenetv3_small_100",
        "efficientnet_lite0",
        "resnet18",
        "convnext_tiny",
    ]
    for model_name in small_models:
        try:
            backbone = at.create_backbone(model_name)
            params = at.count_parameters(backbone)
            print(f"  - {model_name}: {params / 1e6:.1f}M params")
        except Exception:
            pass

    # ========================================================================
    # Discover Transform Presets
    # ========================================================================
    print("\n" + "-" * 70)
    print("Transform Preset Discovery")
    print("-" * 70)

    # List torchvision presets
    print("\nTorchvision presets:")
    for name, desc in list_transform_presets(backend="torchvision", verbose=True):
        print(f"  {name:.<20} {desc}")

    # List albumentations presets
    print("\nAlbumentations presets:")
    for name, desc in list_transform_presets(backend="albumentations", verbose=True):
        print(f"  {name:.<20} {desc}")

    print("\n" + "=" * 70)
    print("Done! Use at.list_backbones() and list_transform_presets() to explore.")
    print("=" * 70)


if __name__ == "__main__":
    main()
