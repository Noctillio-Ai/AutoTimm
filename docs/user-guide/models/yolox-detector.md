---
title: YOLOX Object Detection
description: Train YOLOX object detectors with official weights or custom timm backbones. Supports YOLOX-S/M/L/X variants, mosaic augmentation, and COCO-format datasets.
---

# YOLOX Object Detection Guide

Complete guide to using YOLOX models in AutoTimm, including official YOLOX implementation and YOLOX-style detection with timm backbones.

## Table of Contents

1. [Overview](#overview)
2. [Two Approaches](#two-approaches)
3. [Official YOLOX Models](#official-yolox-models)
4. [YOLOX-Style with timm Backbones](#yolox-style-with-timm-backbones)
5. [Model Selection](#model-selection)
6. [Training Settings](#training-settings)
7. [Advanced Usage](#advanced-usage)
8. [Performance Comparison](#performance-comparison)

## Overview

YOLOX is a high-performance anchor-free object detector that improves upon YOLO series with:
- **Decoupled Head**: Separate branches for classification and regression
- **Anchor-Free**: Grid-based predictions without anchor boxes
- **Strong Augmentations**: Mosaic, MixUp for better generalization
- **SimOTA**: Advanced label assignment strategy

AutoTimm provides **two ways** to use YOLOX:

1. **Official YOLOX** (`YOLOXDetector`): Complete official implementation with CSPDarknet backbone
2. **YOLOX-Style** (`ObjectDetector`): YOLOX head with any timm backbone

## Two Approaches

### Comparison

| Feature | YOLOXDetector | ObjectDetector (yolox) |
|---------|---------------|------------------------|
| **Backbone** | CSPDarknet (official) | Any timm model (1000+) |
| **Neck** | YOLOXPAFPN (official) | FPN (standard) |
| **Head** | YOLOXHead | YOLOXHead |
| **Optimizer** | SGD (official settings) | Configurable |
| **Scheduler** | YOLOX (warmup + cosine) | Configurable |
| **Use Case** | Reproduce official results | Experimentation |
| **Performance** | Matches YOLOX paper | Flexible trade-offs |

### When to Use Each

**Use YOLOXDetector when:**

- You want to reproduce official YOLOX results
- You need production-ready performance
- You want YOLOX paper benchmarks
- You prefer the optimized YOLOX architecture

**Use ObjectDetector (yolox) when:**

- You want to experiment with different backbones
- You need transfer learning from pretrained models
- You want to compare different architectures
- You prefer flexibility over official settings

## Official YOLOX Models

### Quick Start

```python
from autotimm import YOLOXDetector, DetectionDataModule, AutoTrainer

# Create official YOLOX model
model = YOLOXDetector(
    model_name="yolox-s",  # nano, tiny, s, m, l, x
    num_classes=80,
)

# Data
data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=64,
)

# Train
trainer = AutoTrainer(max_epochs=300, precision="16-mixed")
trainer.fit(model, datamodule=data)
```

### Available Models

Use `list_yolox_models()` to see all available models:

```python
from autotimm import list_yolox_models

# Simple list
models = list_yolox_models()
# ['yolox-nano', 'yolox-tiny', 'yolox-s', 'yolox-m', 'yolox-l', 'yolox-x']

# Detailed information
list_yolox_models(verbose=True)
```

| Model | Params | FLOPs | mAP (COCO) | Use Case |
|-------|--------|-------|------------|----------|
| yolox-nano | 0.9M | 1.1G | 25.8 | Edge devices, mobile |
| yolox-tiny | 5.1M | 6.5G | 32.8 | Resource-constrained |
| yolox-s | 9.0M | 26.8G | 40.5 | Balanced speed/accuracy |
| yolox-m | 25.3M | 73.8G | 47.2 | Medium performance |
| yolox-l | 54.2M | 155.6G | 50.1 | High accuracy |
| yolox-x | 99.1M | 281.9G | 51.5 | Maximum accuracy |

### Official Training Settings

YOLOXDetector uses official training configuration by default:

```python
model = YOLOXDetector(
    model_name="yolox-s",
    num_classes=80,
    lr=0.01,              # Base LR for batch size 64
    weight_decay=5e-4,    # Official weight decay
    optimizer="sgd",      # SGD with momentum=0.9, nesterov=True
    scheduler="yolox",    # Warmup + cosine decay
    total_epochs=300,
    warmup_epochs=5,      # Linear warmup
    no_aug_epochs=15,     # No augmentation at end
    reg_loss_weight=5.0,  # YOLOX uses higher reg weight
)
```

### Learning Rate Scheduler

The official YOLOX scheduler has three phases:

1. **Warmup (5 epochs)**: Linear warmup from 0 to base_lr
2. **Main Training (280 epochs)**: Cosine annealing or linear decay
3. **No Augmentation (15 epochs)**: Fixed minimum LR for stability

```python
# Customize scheduler
model = YOLOXDetector(
    model_name="yolox-s",
    scheduler="yolox",
    scheduler_kwargs={
        "total_epochs": 300,
        "warmup_epochs": 10,        # Longer warmup
        "no_aug_epochs": 20,         # More no-aug epochs
        "min_lr_ratio": 0.01,        # Lower minimum LR
        "scheduler_type": "linear",  # Linear instead of cosine
    },
)
```

### Model Architecture

Official YOLOX components:

```python
from autotimm import list_yolox_backbones, list_yolox_necks, list_yolox_heads

# List components
backbones = list_yolox_backbones()  # CSPDarknet variants
necks = list_yolox_necks()          # YOLOXPAFPN variants
heads = list_yolox_heads()          # YOLOXHead

# Get detailed architecture
from autotimm import get_yolox_model_info

info = get_yolox_model_info("yolox-s")
print(f"Backbone: {info['backbone']}")        # csp_darknet_s
print(f"Neck: {info['neck']}")                # yolox_pafpn_s
print(f"Head: {info['head']}")                # yolox_head
print(f"Channels: {info['backbone_channels']}")  # (128, 256, 512)
```

## YOLOX-Style with timm Backbones

### Quick Start

```python
from autotimm import ObjectDetector, DetectionDataModule, AutoTrainer

# YOLOX-style head with any timm backbone
model = ObjectDetector(
    backbone="resnet50",  # Any timm model
    num_classes=80,
    detection_arch="yolox",  # Use YOLOX head
    fpn_channels=256,
    head_num_convs=2,
    cls_loss_weight=1.0,
    reg_loss_weight=5.0,
)

data = DetectionDataModule(data_dir="./coco", image_size=640, batch_size=16)
trainer = AutoTrainer(max_epochs=300)
trainer.fit(model, datamodule=data)
```

### Flexible Backbone Selection

Use any of the 1000+ timm models:

```python
# ResNet family
model = ObjectDetector(backbone="resnet50", detection_arch="yolox", ...)

# EfficientNet family
model = ObjectDetector(backbone="efficientnet_b0", detection_arch="yolox", ...)

# ConvNeXt family
model = ObjectDetector(backbone="convnext_tiny", detection_arch="yolox", ...)

# Vision Transformers
model = ObjectDetector(backbone="vit_base_patch16_224", detection_arch="yolox", ...)

# Search available backbones
import autotimm as at  # recommended alias
at.list_backbones("*efficientnet*", pretrained_only=True)
```

### Custom Training Settings

Full control over optimizer and scheduler:

```python
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    detection_arch="yolox",
    lr=1e-3,
    weight_decay=1e-4,
    optimizer="adamw",       # AdamW instead of SGD
    scheduler="cosine",      # Simple cosine decay
    fpn_channels=256,
    head_num_convs=2,
    focal_alpha=0.25,
    focal_gamma=2.0,
    cls_loss_weight=1.0,
    reg_loss_weight=5.0,
)
```

## Model Selection

### By Use Case

**Edge Devices / Mobile**:
```python
# YOLOX-Nano: 0.9M params, 1.1G FLOPs
model = YOLOXDetector(model_name="yolox-nano", num_classes=80)
```

**Balanced Performance**:
```python
# YOLOX-S: 9.0M params, 26.8G FLOPs, 40.5 mAP
model = YOLOXDetector(model_name="yolox-s", num_classes=80)
```

**High Accuracy**:
```python
# YOLOX-L: 54.2M params, 155.6G FLOPs, 50.1 mAP
model = YOLOXDetector(model_name="yolox-l", num_classes=80)
```

**Maximum Accuracy**:
```python
# YOLOX-X: 99.1M params, 281.9G FLOPs, 51.5 mAP
model = YOLOXDetector(model_name="yolox-x", num_classes=80)
```

### By Performance Requirements

```python
from autotimm import get_yolox_model_info

# Find models with specific requirements
for model_name in ['yolox-nano', 'yolox-tiny', 'yolox-s', 'yolox-m', 'yolox-l', 'yolox-x']:
    info = get_yolox_model_info(model_name)
    if info['mAP'] > 40 and float(info['params'][:-1]) < 30:  # mAP > 40, < 30M params
        print(f"Recommended: {model_name} (mAP: {info['mAP']}, Params: {info['params']})")
```

## Training Settings

### Official YOLOX Settings

For reproducing official results:

```python
model = YOLOXDetector(
    model_name="yolox-s",
    num_classes=80,
    lr=0.01,              # 0.01 for batch size 64 (scale linearly)
    weight_decay=5e-4,
    optimizer="sgd",
    scheduler="yolox",
    total_epochs=300,
    warmup_epochs=5,
    no_aug_epochs=15,
    reg_loss_weight=5.0,
)

data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=64,  # Official uses 64 (8 GPUs × 8 per GPU)
)

trainer = AutoTrainer(
    max_epochs=300,
    precision="16-mixed",
    accumulate_grad_batches=1,
)
```

### Learning Rate Scaling

Scale learning rate based on batch size:

```python
base_lr = 0.01  # For batch size 64
batch_size = 32
lr = base_lr * (batch_size / 64)  # 0.005 for batch size 32

model = YOLOXDetector(model_name="yolox-s", lr=lr, ...)
```

### Multi-GPU Training

```python
trainer = AutoTrainer(
    max_epochs=300,
    devices=8,           # Use 8 GPUs
    strategy="ddp",      # Distributed Data Parallel
    precision="16-mixed",
)

# Adjust batch size per GPU
data = DetectionDataModule(
    data_dir="./coco",
    batch_size=8,  # Per GPU: 8 GPUs × 8 = 64 total
)
```

## Advanced Usage

### Custom Metrics

```python
from autotimm import YOLOXDetector, MetricConfig

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

model = YOLOXDetector(model_name="yolox-s", metrics=metrics, ...)
```

### torch.compile (PyTorch 2.0+)

**Enabled by default** for faster training and inference:

```python
# Default: torch.compile enabled
model = YOLOXDetector(
    model_name="yolox-s",
    num_classes=80,
)

# Disable if needed
model = YOLOXDetector(
    model_name="yolox-s",
    num_classes=80,
    compile_model=False,
)

# Custom compile options
model = YOLOXDetector(
    model_name="yolox-s",
    num_classes=80,
    compile_kwargs={"mode": "reduce-overhead"},
)
```

**What gets compiled:** CSPDarknet Backbone + YOLOXPAFPN Neck + YOLOX Head

See [ImageClassifier](image-classifier.md#performance-optimization) for compile mode details.

### Inference

```python
import torch
from autotimm import YOLOXDetector

# Load trained model
model = YOLOXDetector.load_from_checkpoint("path/to/checkpoint.ckpt", compile_model=False)
model.eval()

# Prepare image
images = torch.randn(1, 3, 640, 640)

# Run inference
with torch.inference_mode():
    predictions = model.predict(images)

# Process results
for pred in predictions:
    boxes = pred["boxes"]      # [N, 4] in xyxy format
    scores = pred["scores"]    # [N]
    labels = pred["labels"]    # [N]
```

### Transfer Learning

Fine-tune on custom dataset:

```python
# Start from official YOLOX-S
model = YOLOXDetector(
    model_name="yolox-s",
    num_classes=10,  # Your custom classes
    lr=0.001,        # Lower LR for fine-tuning
    total_epochs=50, # Fewer epochs
)

# Load pretrained weights (when available)
# model = YOLOXDetector.load_from_checkpoint("yolox_s_coco.ckpt")
```

## Performance Comparison

### Official YOLOX vs YOLOX-Style

Tested on COCO val2017:

| Model | Architecture | mAP | FPS (V100) | Notes |
|-------|-------------|-----|------------|-------|
| YOLOX-S (official) | CSPDarknet + PAFPN | 40.5 | 102 | Official settings |
| YOLOX-S (timm R50) | ResNet50 + FPN | ~38.0 | 95 | YOLOX head only |
| YOLOX-M (official) | CSPDarknet + PAFPN | 47.2 | 81 | Official settings |
| YOLOX-L (official) | CSPDarknet + PAFPN | 50.1 | 69 | Official settings |

### Speed vs Accuracy Trade-offs

```
YOLOX-Nano:  25.8 mAP,   ~2ms inference (fastest)
YOLOX-Tiny:  32.8 mAP,   ~4ms inference
YOLOX-S:     40.5 mAP,  ~10ms inference (balanced)
YOLOX-M:     47.2 mAP,  ~12ms inference
YOLOX-L:     50.1 mAP,  ~15ms inference
YOLOX-X:     51.5 mAP,  ~20ms inference (most accurate)
```

## Examples

See the `examples/` directory for complete working examples:

- **`yolox_official.py`**: Official YOLOX training with all settings
- **`object_detection_yolox.py`**: YOLOX-style head with timm backbones
- **`explore_yolox_models.py`**: Interactive model explorer

## References

- **[YOLOX Quick Reference](../guides/yolox-quick-reference.md)** - Fast reference card
- **[Object Detector Guide](object-detector.md)** - YOLOX-style with timm backbones
- **Official YOLOX**: https://github.com/Megvii-BaseDetection/YOLOX
- **Paper**: https://arxiv.org/abs/2107.08430
- **AutoTimm Docs**: https://noctillio-ai.github.io/AutoTimm/

## Troubleshooting

For YOLOX detector issues, see the [Troubleshooting - YOLOX](../../troubleshooting/task-specific/yolox.md) including:

- CUDA out of memory
- Slow training
- Poor performance
- Proper settings configuration
