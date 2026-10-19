# Hugging Face Hub Examples

This guide demonstrates how to use Hugging Face Hub models with AutoTimm for various computer vision tasks.

## Overview

AutoTimm supports loading timm-compatible models directly from Hugging Face Hub using the `hf-hub:` prefix. This gives you access to thousands of pretrained models with version control, model cards, and community contributions.

## Quick Start

```python
import autotimm as at  # recommended alias
from autotimm import ImageClassifier

# Discover models on HF Hub
models = at.list_hf_hub_backbones(model_name="resnet", limit=5)

# Use HF Hub model as backbone
model = ImageClassifier(
    backbone="hf-hub:timm/resnet50.a1_in1k",
    num_classes=10,
)
```

## Example Files

### 1. Introduction to HF Hub Models

**File:** `huggingface_hub_models.py`

**What it covers:**

- Discovering models on HF Hub
- Comparing timm vs HF Hub models
- Basic training workflow
- Model search and filtering

**Run it:**

```bash
python examples/huggingface/huggingface_hub_models.py
```

**Key Features:**

- `list_hf_hub_backbones()` function usage
- Creating backbones with `hf-hub:` prefix
- Integration with all AutoTimm tasks

### 2. Image Classification

**File:** `hf_hub_classification.py`

**What it covers:**

- Classification-specific backbone discovery
- Comparing model sizes and complexities
- Training with different architectures (ResNet, ViT, MobileNet)
- Model selection for different scenarios

**Architectures covered:**

- **ResNet**: Classic, reliable, fast
- **Vision Transformer (ViT)**: State-of-the-art accuracy
- **EfficientNet**: Excellent efficiency
- **ConvNeXt**: Modern CNN design
- **MobileNet**: Edge/mobile deployment

**Run it:**

```bash
python examples/huggingface/hf_hub_classification.py
```

**Use cases:**

- Standard image classification
- Fine-grained recognition
- Edge deployment
- Transfer learning

### 3. Semantic Segmentation

**File:** `hf_hub_segmentation.py`

**What it covers:**

- Segmentation-specific backbone selection
- Feature extraction analysis
- DeepLabV3+ vs FCN comparison
- Lightweight models for edge devices

**Architectures:**

- ResNet50 + DeepLabV3+
- ConvNeXt + FCN
- MobileNetV3 + FCN (lightweight)

**Run it:**

```bash
python examples/huggingface/hf_hub_segmentation.py
```

**Scenarios:**

- High-accuracy segmentation (research)
- Balanced production models
- Fast inference (edge devices)
- Modern architectures

### 4. Object Detection

**File:** `hf_hub_object_detection.py`

**What it covers:**

- Detection-specific backbone analysis
- Feature Pyramid Network (FPN) compatibility
- FCOS detector configurations
- Computational requirements comparison

**Architectures:**

- ResNet50 (baseline)
- ResNeXt50 (stronger features)
- EfficientNet (efficiency)
- ConvNeXt (modern design)

**Run it:**

```bash
python examples/huggingface/hf_hub_object_detection.py
```

**Configurations:**

- High accuracy (ResNeXt101)
- Balanced (ResNet50)
- Fast inference (EfficientNet)
- Modern (ConvNeXt)

### 5. Instance Segmentation

**File:** `hf_hub_instance_segmentation.py`

**What it covers:**

- Mask R-CNN style architecture
- Backbone compatibility analysis
- Training tips and best practices
- Computational trade-offs

**Architectures:**

- ResNet50 (standard)
- ResNeXt50 (improved features)
- ConvNeXt (state-of-the-art)
- ResNet101 (high capacity)

**Run it:**

```bash
python examples/huggingface/hf_hub_instance_segmentation.py
```

**Tips:**

- Data requirements (1000+ instances per class)
- Hyperparameter settings
- Two-stage training strategies
- Metric monitoring (bbox + mask mAP)

### 6. Advanced Usage

**File:** `hf_hub_advanced.py`

**What it covers:** Pretraining datasets (IN1k/21k/22k), architecture analysis, transfer learning, model versioning, deployment scenarios.

**Run it:**

```bash
python examples/huggingface/hf_hub_advanced.py
```

## Model Naming Convention

HF Hub models follow a structured naming convention:

```
hf-hub:timm/<architecture>_<variant>.<recipe>_<dataset>
```

**Examples:**

- `hf-hub:timm/resnet50.a1_in1k`
    - Architecture: ResNet-50
    - Recipe: a1 (training configuration)
    - Dataset: ImageNet-1k

- `hf-hub:timm/convnext_base.fb_in22k_ft_in1k`
    - Architecture: ConvNeXt Base
    - Recipe: fb (Facebook)
    - Pretraining: ImageNet-22k
    - Fine-tuned on: ImageNet-1k

- `hf-hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k`
    - Architecture: Vision Transformer Small
    - Patch size: 16x16
    - Input: 224x224
    - Recipe: augreg (augmentation + regularization)
    - Pretraining: ImageNet-21k
    - Fine-tuned on: ImageNet-1k

## Supported Prefixes

You can use any of these formats:

- `hf-hub:timm/model_name`
- `hf_hub:timm/model_name`
- `timm/model_name`

## Model Discovery

```python
import autotimm as at  # recommended alias

# Search by architecture
resnets = at.list_hf_hub_backbones(model_name="resnet", limit=10)
vits = at.list_hf_hub_backbones(model_name="vit", limit=10)

# Search by author
timm_models = at.list_hf_hub_backbones(author="timm", limit=20)

# Filter for efficient models
efficient = at.list_hf_hub_backbones(model_name="mobilenet", limit=5)
```

## Quick Reference: Task-Specific Recommendations

### Image Classification

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Standard | `hf-hub:timm/resnet50.a1_in1k` | Proven, reliable baseline |
| High Accuracy | `hf-hub:timm/convnext_base.fb_in22k_ft_in1k` | Modern, strong features |
| Edge/Mobile | `hf-hub:timm/mobilenetv3_small_100.lamb_in1k` | Lightweight, fast |
| Fine-grained | `hf-hub:timm/vit_base_patch16_224.augreg_in21k_ft_in1k` | Rich features from 21k pretraining |

### Semantic Segmentation

| Use Case | Recommended Model | Head Type |
|----------|------------------|-----------|
| High Quality | `hf-hub:timm/resnet50.a1_in1k` | DeepLabV3+ |
| Modern | `hf-hub:timm/convnext_tiny.fb_in22k` | DeepLabV3+ |
| Fast | `hf-hub:timm/mobilenetv3_large_100.ra_in1k` | FCN |

### Object Detection

| Use Case | Recommended Model | Configuration |
|----------|------------------|---------------|
| Baseline | `hf-hub:timm/resnet50.a1_in1k` | 640px, FPN-256 |
| Best mAP | `hf-hub:timm/resnext50_32x4d.a1_in1k` | 800px, FPN-256 |
| Real-time | `hf-hub:timm/efficientnet_b0.ra_in1k` | 512px, FPN-128 |

### Instance Segmentation

| Use Case | Recommended Model | Configuration |
|----------|------------------|---------------|
| Standard | `hf-hub:timm/resnet50.a1_in1k` | 640px, Mask-256 |
| Research | `hf-hub:timm/resnext101_32x8d.fb_wsl_ig1b_ft_in1k` | 1024px, Mask-256 |
| Production | `hf-hub:timm/convnext_small.fb_in22k_ft_in1k` | 640px, Mask-256 |

## Transfer Learning Guidelines

### Dataset Size

- **Small (<1k images)**: Use feature extraction, freeze backbone
- **Medium (1k-10k)**: Fine-tune top layers only
- **Large (>10k)**: Fine-tune all layers

### Learning Rate

- **ResNet family**: Start with `lr=1e-3`
- **EfficientNet**: Start with `lr=1e-3`
- **Vision Transformers**: Start with `lr=1e-4`
- **ConvNeXt**: Start with `lr=5e-4`

### Weight Decay

- **ResNet/EfficientNet**: `weight_decay=1e-4`
- **ViT/ConvNeXt**: `weight_decay=0.05`

## Common Patterns

### Pattern 1: Quick Experimentation

```python
# Try different backbones quickly
backbones = [
    "hf-hub:timm/resnet18.a1_in1k",
    "hf-hub:timm/resnet50.a1_in1k",
    "hf-hub:timm/efficientnet_b0.ra_in1k",
]

for backbone_name in backbones:
    model = ImageClassifier(backbone=backbone_name, num_classes=10)
    # Train and compare
```

### Pattern 2: Progressive Scaling

```python
# Start small, scale up
stages = [
    ("prototype", "hf-hub:timm/resnet18.a1_in1k"),
    ("baseline", "hf-hub:timm/resnet50.a1_in1k"),
    ("production", "hf-hub:timm/convnext_small.fb_in22k_ft_in1k"),
]

for stage_name, backbone in stages:
    model = ImageClassifier(backbone=backbone, num_classes=10)
    # Train with appropriate epochs/budget
```

### Pattern 3: Custom Configuration

```python
from autotimm.core.backbone import BackboneConfig

# Fine control over model creation
config = BackboneConfig(
    model_name="hf-hub:timm/resnet50.a1_in1k",
    pretrained=True,
    drop_rate=0.3,      # Custom dropout
    drop_path_rate=0.1,  # Stochastic depth
)

backbone = autotimm.create_backbone(config)
```

## Troubleshooting

For HuggingFace Hub issues, see the [Troubleshooting - HuggingFace](../../troubleshooting/integration/huggingface.md) which covers:

- Model not found on Hub
- Model download is slow
- Checkpoint loading fails
- Out of memory issues
- And more HF Hub-related issues

## Benefits

- **Version Control**: Exact model versions with reproducible results
- **Model Cards**: Detailed training info and metrics
- **Community**: Access community-trained models
- **Centralized**: Single source for all model weights

## Resources

- [Hugging Face Hub - timm](https://huggingface.co/timm)
- [timm Documentation](https://huggingface.co/docs/timm)
- [AutoTimm Documentation](https://noctillio-ai.github.io/AutoTimm/)
