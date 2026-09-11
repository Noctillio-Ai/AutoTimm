# Backbone Utilities Examples

This page demonstrates how to explore and use timm backbones in AutoTimm.

## Backbone Discovery

Explore available timm backbones and their properties.

```python
import autotimm as at  # recommended alias


def main():
    # List all backbones
    all_models = at.list_backbones()
    print(f"Total models: {len(all_models)}")

    # Search by pattern
    resnet = at.list_backbones("*resnet*")
    print(f"ResNet variants: {len(resnet)}")

    efficientnet = at.list_backbones("*efficientnet*", pretrained_only=True)
    print(f"EfficientNet variants (pretrained): {len(efficientnet)}")

    vit = at.list_backbones("*vit*")
    print(f"Vision Transformer variants: {len(vit)}")

    # Print some examples
    print("\nSample ResNet models:")
    for model in resnet[:5]:
        print(f"  - {model}")

    # Inspect a backbone
    backbone = at.create_backbone("resnet50")
    print(f"\nResNet50 details:")
    print(f"  Output features: {backbone.num_features}")
    print(f"  Parameters: {at.count_parameters(backbone):,}")


if __name__ == "__main__":
    main()
```

---

## Popular Backbone Families

Explore common backbone architectures:

```python
import autotimm as at  # recommended alias

# ResNet family
resnets = at.list_backbones("resnet*")
popular_resnets = ["resnet18", "resnet34", "resnet50", "resnet101", "resnext50_32x4d"]

# EfficientNet family
efficientnets = at.list_backbones("efficientnet*")
popular_efficientnets = ["efficientnet_b0", "efficientnet_b3", "efficientnetv2_rw_s"]

# Vision Transformers
vits = at.list_backbones("vit*")
popular_vits = ["vit_tiny_patch16_224", "vit_base_patch16_224", "vit_large_patch16_224"]

# Swin Transformers
swins = at.list_backbones("swin*")
popular_swins = ["swin_tiny_patch4_window7_224", "swin_base_patch4_window7_224"]

# Compare parameters
for model in popular_resnets[:3]:
    backbone = at.create_backbone(model)
    print(f"{model}: {at.count_parameters(backbone):,} params")
```

---

## Backbone Comparison

Compare multiple backbones for your use case.

```python
import autotimm as at  # recommended alias


def compare_backbones(model_names):
    """Compare multiple backbones."""
    print(f"{'Model':<40} {'Params':>12} {'Features':>10}")
    print("-" * 65)

    for name in model_names:
        backbone = at.create_backbone(name)
        params = at.count_parameters(backbone)
        features = backbone.num_features
        print(f"{name:<40} {params:>12,} {features:>10}")


def main():
    # Compare different ResNet variants
    print("ResNet Comparison:")
    compare_backbones([
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
    ])
    
    print("\nEfficientNet Comparison:")
    compare_backbones([
        "efficientnet_b0",
        "efficientnet_b2",
        "efficientnet_b4",
    ])
    
    print("\nTransformer Comparison:")
    compare_backbones([
        "vit_tiny_patch16_224",
        "vit_base_patch16_224",
        "swin_tiny_patch4_window7_224",
        "swin_base_patch4_window7_224",
    ])


if __name__ == "__main__":
    main()
```

---

## Backbone Inspection

Inspect backbone architecture and output features:

```python
import autotimm as at  # recommended alias
import torch


def inspect_backbone(model_name):
    backbone = at.create_backbone(at.BackboneConfig(model_name=model_name, pretrained=True))
    print(f"Model: {model_name}")
    print(f"  Parameters: {at.count_parameters(backbone):,}")
    print(f"  Output features: {backbone.num_features}")

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        output = backbone(dummy_input)
    print(f"  Output shape: {output.shape}\n")


# Inspect various backbones
for model in ["resnet50", "efficientnet_b3", "vit_base_patch16_224"]:
    inspect_backbone(model)

# Search for specific models
mobile_models = at.list_backbones("*mobile*")
convnext_models = at.list_backbones("*convnext*")
pretrained_only = at.list_backbones("*efficientnet*", pretrained_only=True)
```

---

## Backbone Selection Guide

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| Quick prototyping | `resnet18`, `resnet34` | Fast, small, reliable |
| Balanced performance | `resnet50`, `efficientnet_b3` | Good accuracy/speed |
| Maximum accuracy | `vit_large_patch16_224`, `swin_large` | State-of-the-art |
| Mobile/Edge | `efficientnet_b0`, `mobilenetv3_small_100` | Efficient inference |
| Object Detection | `resnet50`, `swin_tiny` | Hierarchical features |

---

## Optimizers and Schedulers

Discover available optimizers and schedulers:

```python
import autotimm as at  # recommended alias

# List all optimizers (PyTorch + timm)
optimizers = at.list_optimizers()

# List all schedulers
schedulers = at.list_schedulers()

# Use in model
model = at.ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    optimizer="adamw",
    scheduler="cosineannealinglr",
    scheduler_kwargs={"T_max": 50},
)
```

---

## Running Examples

```bash
python examples/logging_inference/backbone_discovery.py
```
