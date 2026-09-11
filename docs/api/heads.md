# Heads

Neural network heads for classification and object detection tasks.

## ClassificationHead

Simple classification head with optional dropout.

### API Reference

::: autotimm.ClassificationHead
    options:
      show_source: true
      members:
        - __init__
        - forward

### Usage Example

```python
from autotimm import ClassificationHead
import torch

head = ClassificationHead(
    in_features=2048,
    num_classes=10,
    dropout=0.5,
)

features = torch.randn(32, 2048)
logits = head(features)  # (32, 10)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | `int` | Required | Input feature dimension |
| `num_classes` | `int` | Required | Number of output classes |
| `dropout` | `float` | `0.0` | Dropout rate before classifier |

### Architecture

```
ClassificationHead
├── dropout (if dropout > 0)
└── Linear(in_features, num_classes)
```

---

## DetectionHead

FCOS-style detection head with classification, bbox regression, and centerness branches.

### API Reference

::: autotimm.DetectionHead
    options:
      show_source: true
      members:
        - __init__
        - forward
        - init_weights

### Usage Example

```python
from autotimm import DetectionHead
import torch

head = DetectionHead(
    in_channels=256,
    num_classes=80,
    num_convs=4,
)

features = [
    torch.randn(2, 256, 80, 80),   # P3
    torch.randn(2, 256, 40, 40),   # P4
    torch.randn(2, 256, 20, 20),   # P5
]

cls_scores, bbox_preds, centernesses = head(features)
# cls_scores: List of (B, num_classes, H, W) tensors
# bbox_preds: List of (B, 4, H, W) tensors (l, t, r, b)
# centernesses: List of (B, 1, H, W) tensors
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_channels` | `int` | Required | Number of input channels from FPN |
| `num_classes` | `int` | Required | Number of object classes |
| `num_convs` | `int` | `4` | Number of conv layers per branch |
| `prior_prob` | `float` | `0.01` | Prior probability for focal loss initialization |

### Architecture

```
DetectionHead (shared across all FPN levels)
├── cls_subnet
│   ├── Conv3x3 + GroupNorm + ReLU (×num_convs)
│   └── Conv3x3 → (num_classes, H, W)
├── bbox_subnet
│   ├── Conv3x3 + GroupNorm + ReLU (×num_convs)
│   └── Conv3x3 → (4, H, W)  # l, t, r, b offsets
└── centerness_subnet
    ├── Conv3x3 + GroupNorm + ReLU (×num_convs)
    └── Conv3x3 → (1, H, W)
```

### Output Format

**Classification Scores:**

- Shape: `(B, num_classes, H, W)` for each FPN level
- Values: Raw logits (apply sigmoid for probabilities)

**Bounding Box Predictions:**

- Shape: `(B, 4, H, W)` for each FPN level
- Format: `(left, top, right, bottom)` offsets from each location
- Values: Raw predictions (apply exp() to get distances)

**Centerness Scores:**

- Shape: `(B, 1, H, W)` for each FPN level
- Values: Raw logits (apply sigmoid for probabilities)
- Purpose: Suppresses low-quality detections far from object centers

---

## FPN

Feature Pyramid Network for multi-scale feature extraction.

### API Reference

::: autotimm.FPN
    options:
      show_source: true
      members:
        - __init__
        - forward

### Usage Example

```python
from autotimm import FPN, create_feature_backbone
import torch

# Create backbone
backbone = create_feature_backbone("resnet50")
in_channels = [512, 1024, 2048]  # C3, C4, C5

# Create FPN
fpn = FPN(
    in_channels_list=in_channels,
    out_channels=256,
)

# Extract features
images = torch.randn(2, 3, 640, 640)
features = backbone(images)  # [C3, C4, C5]
pyramid = fpn(features)      # [P3, P4, P5, P6, P7]

# Pyramid features all have 256 channels
for i, feat in enumerate(pyramid):
    print(f"P{i+3}: {feat.shape}")
# P3: torch.Size([2, 256, 80, 80])
# P4: torch.Size([2, 256, 40, 40])
# P5: torch.Size([2, 256, 20, 20])
# P6: torch.Size([2, 256, 10, 10])
# P7: torch.Size([2, 256, 5, 5])
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_channels` | `list[int]` | Required | Input channels for each backbone level (C3, C4, C5) |
| `out_channels` | `int` | `256` | Output channels for all pyramid levels |

### Architecture

```
FPN
├── Lateral connections (1x1 conv)
│   ├── C5 → lateral_conv5
│   ├── C4 → lateral_conv4
│   └── C3 → lateral_conv3
├── Top-down pathway (upsample + add)
│   ├── P5 = lateral5
│   ├── P4 = lateral4 + upsample(P5)
│   └── P3 = lateral3 + upsample(P4)
├── Output convolutions (3x3 conv)
│   ├── P3 → fpn_conv3
│   ├── P4 → fpn_conv4
│   └── P5 → fpn_conv5
└── Additional levels (downsampling)
    ├── P6 = MaxPool(P5)
    └── P7 = MaxPool(P6)
```

### Feature Pyramid Levels

| Level | Stride | Resolution (640px) | Typical Object Size |
|-------|--------|-------------------|---------------------|
| P3 | 8 | 80×80 | Very small (8-64px) |
| P4 | 16 | 40×40 | Small (64-128px) |
| P5 | 32 | 20×20 | Medium (128-256px) |
| P6 | 64 | 10×10 | Large (256-512px) |
| P7 | 128 | 5×5 | Very large (>512px) |

### How FPN Works

1. **Bottom-up pathway**: Backbone extracts features at multiple scales (C3, C4, C5)
2. **Lateral connections**: 1×1 convolutions reduce channels to `out_channels`
3. **Top-down pathway**: Higher-level features are upsampled and added to lateral connections
4. **Output smoothing**: 3×3 convolutions smooth the merged features
5. **Additional levels**: P6 and P7 are created via max pooling for detecting larger objects

### Design Choices

**Why out_channels=256?**

- Standard choice in FPN literature (from original paper)
- Good balance between capacity and efficiency
- Can use 128 for faster inference or 512 for higher capacity

**Why 5 pyramid levels?**

- Covers object scales from very small to very large
- P3-P7 handles objects from ~8px to >512px
- More levels = better multi-scale detection but slower inference

**Why top-down + lateral?**

- Top-down: High-level semantic information flows to lower levels
- Lateral: Preserves precise localization from lower levels
- Combination: Semantically strong features at all scales

### Usage in Detection

```python
from autotimm import ObjectDetector

# FPN is automatically created inside ObjectDetector
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    fpn_channels=256,  # Controls FPN out_channels
)

# FPN construction:
# 1. Backbone extracts [C3, C4, C5] with channels [512, 1024, 2048]
# 2. FPN converts to [P3, P4, P5, P6, P7] with uniform 256 channels
# 3. DetectionHead processes all pyramid levels
```
