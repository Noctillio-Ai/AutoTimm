# Loss Functions

Loss functions for object detection tasks.

## FocalLoss

Focal Loss for addressing class imbalance in one-stage object detectors.

### Overview

Focal Loss reshapes the standard cross-entropy loss to down-weight easy examples and focus training on hard negatives. This is crucial for one-stage detectors where there's extreme imbalance between foreground and background classes.

**Formula:**
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

where:
  p_t = p if y = 1, else 1 - p
  α_t = α if y = 1, else 1 - α
```

### API Reference

::: autotimm.FocalLoss
    options:
      show_source: true
      members:
        - __init__
        - forward

### Usage Example

```python
from autotimm import FocalLoss
import torch

loss_fn = FocalLoss(
    alpha=0.25,
    gamma=2.0,
)

# Predictions and targets
pred = torch.randn(32, 80, 100, 100)  # (B, C, H, W)
target = torch.randint(0, 80, (32, 100, 100))  # (B, H, W)

loss = loss_fn(pred, target)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `0.25` | Weighting factor for class 1 (foreground) |
| `gamma` | `float` | `2.0` | Focusing parameter for hard examples |
| `reduction` | `str` | `"mean"` | Loss reduction: "mean", "sum", or "none" |

### How It Works

**Alpha (α):**

- Controls the relative importance of positive vs negative examples
- `α = 0.25` means positive examples get 25% weight, negatives get 75%
- Higher α → focus more on positives (foreground)
- Standard value: 0.25

**Gamma (γ):**

- Controls how much to down-weight easy examples
- `γ = 0`: Equivalent to standard cross-entropy
- `γ = 2`: Standard focal loss (down-weights easy examples significantly)
- Higher γ → focuses more on hard examples
- Standard value: 2.0

**Effect:**

- Easy examples (high confidence, correct predictions): Very low loss
- Hard examples (low confidence): High loss
- Result: Model focuses on learning hard negatives

### When to Use

**Use Focal Loss when:**

- Extreme class imbalance (e.g., 1:1000 positive:negative ratio in detection)
- One-stage detectors (FCOS, RetinaNet, YOLO)
- Many easy negatives dominate the loss

**Alternatives:**

- Balanced Cross Entropy: Simpler, use when imbalance is moderate
- Weighted BCE: Manual class weights

### Tuning Guidelines

| Scenario | Alpha | Gamma | Reason |
|----------|-------|-------|--------|
| Standard detection | 0.25 | 2.0 | Default, works for most cases |
| More positives | 0.5 | 2.0 | Less imbalance |
| Fewer positives | 0.1 | 2.0 | Extreme imbalance |
| Too many false positives | 0.25 | 3.0 | Focus even more on hard negatives |
| Missing detections | 0.5 | 1.0 | Easier on hard examples |

---

## GIoULoss

Generalized Intersection over Union (GIoU) Loss for bounding box regression.

### Overview

GIoU Loss extends IoU to handle non-overlapping boxes and provides better gradient flow. Unlike L1 or L2 loss, GIoU is scale-invariant and directly optimizes the detection metric.

**Formula:**
```
IoU = |A ∩ B| / |A ∪ B|
GIoU = IoU - |C \ (A ∪ B)| / |C|

where:
  A, B are predicted and target boxes
  C is the smallest enclosing box

GIoU Loss = 1 - GIoU
```

### API Reference

::: autotimm.GIoULoss
    options:
      show_source: true
      members:
        - __init__
        - forward

### Usage Example

```python
from autotimm import GIoULoss
import torch

loss_fn = GIoULoss()

# Boxes in (x1, y1, x2, y2) format
pred_boxes = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32)
target_boxes = torch.tensor([[12, 12, 48, 48], [25, 25, 65, 65]], dtype=torch.float32)

loss = loss_fn(pred_boxes, target_boxes)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reduction` | `str` | `"mean"` | Loss reduction: "mean", "sum", or "none" |
| `eps` | `float` | `1e-7` | Small value to avoid division by zero |

### How It Works

**Standard IoU:**

- Range: [0, 1]
- 1 = perfect overlap
- 0 = no overlap
- Problem: Zero gradient when boxes don't overlap

**GIoU Improvement:**

- Range: [-1, 1]
- Considers the area of the smallest enclosing box
- Provides gradient signal even for non-overlapping boxes
- Penalizes both misalignment and poor aspect ratios

**Advantages over L1/L2:**
1. **Scale-invariant**: Works equally well for small and large boxes
2. **Direct optimization**: Optimizes IoU directly, not coordinates
3. **Better gradients**: Non-zero gradients for non-overlapping boxes
4. **Aspect ratio aware**: Penalizes incorrect aspect ratios

### GIoU Values

| GIoU | IoU | Interpretation |
|------|-----|----------------|
| 1.0 | 1.0 | Perfect match |
| 0.5 | 0.5 | 50% overlap, no wasted space |
| 0.0 | 0.0 | No overlap, but touching |
| -0.5 | 0.0 | No overlap, some distance |
| -1.0 | 0.0 | No overlap, maximum distance |

### When to Use

**Use GIoU Loss when:**

- Training object detectors
- Need scale-invariant regression
- Boxes can be non-overlapping during training
- Want to optimize IoU metric directly

**Alternatives:**

- **IoU Loss**: Simpler, but undefined for non-overlapping boxes
- **DIoU Loss**: Faster convergence, considers center distance
- **CIoU Loss**: Best for accurate localization, considers aspect ratio explicitly
- **L1 Loss**: Simple, but not scale-invariant

### Input Format

Boxes must be in `(x1, y1, x2, y2)` format where:
- `x1, y1`: Top-left corner coordinates
- `x2, y2`: Bottom-right corner coordinates
- `x2 > x1` and `y2 > y1`

```python
# Correct format
boxes = torch.tensor([[10, 20, 50, 60]])  # (x1=10, y1=20, x2=50, y2=60)

# Convert from (x, y, w, h) if needed
def xywh_to_xyxy(boxes):
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)
```

---

## CenternessLoss

Binary cross-entropy loss for centerness prediction in FCOS.

### Overview

Centerness predicts how "centered" a location is within its assigned object. It's used to down-weight low-quality bounding boxes that are far from object centers, improving detection quality without NMS.

**Formula:**
```
centerness = sqrt((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))

where l, t, r, b are distances from a location to the left, top, right, bottom
of its target bounding box.
```

### API Reference

::: autotimm.CenternessLoss
    options:
      show_source: true
      members:
        - __init__
        - forward

### Usage Example

```python
from autotimm import CenternessLoss
import torch

loss_fn = CenternessLoss()

# Centerness predictions (logits)
pred = torch.randn(32, 1, 100, 100)  # (B, 1, H, W)
target = torch.rand(32, 1, 100, 100)  # (B, 1, H, W), values in [0, 1]

loss = loss_fn(pred, target)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reduction` | `str` | `"mean"` | Loss reduction: "mean", "sum", or "none" |

### How It Works

**Centerness Computation:**
1. For each location, compute distances to box edges: `l, t, r, b`
2. Compute geometric mean of horizontal and vertical ratios
3. Value is 1.0 at the center, decreases towards edges
4. Multiply with classification score during inference

**Purpose:**

- Suppress low-quality detections far from object centers
- Alternative to NMS (though FCOS still uses NMS)
- Improve localization quality by favoring central predictions

**Example:**
```
Object bounding box: [10, 10, 50, 50]
Location at (30, 30): Center → centerness ≈ 1.0
Location at (10, 10): Corner → centerness ≈ 0.0
Location at (30, 10): Edge → centerness ≈ 0.5
```

### When to Use

**Use Centerness Loss when:**

- Implementing FCOS or similar anchor-free detectors
- Want to suppress low-quality detections
- Training point-based detectors

**Not needed for:**

- Anchor-based detectors (Faster R-CNN, etc.)
- Methods with different quality measures (e.g., IoU prediction)

---

## FCOSLoss

Combined loss function for FCOS object detection.

### Overview

FCOSLoss combines Focal Loss (classification), GIoU Loss (bbox regression), and Centerness Loss into a single differentiable loss function for end-to-end training.

**Formula:**
```
Total Loss = λ_cls * Focal Loss
           + λ_reg * GIoU Loss
           + λ_centerness * Centerness Loss
```

### API Reference

::: autotimm.FCOSLoss
    options:
      show_source: true
      members:
        - __init__
        - forward

### Usage Example

```python
from autotimm import FCOSLoss
import torch

loss_fn = FCOSLoss(
    num_classes=80,
    focal_alpha=0.25,
    focal_gamma=2.0,
    cls_weight=1.0,
    reg_weight=1.0,
    centerness_weight=1.0,
)

# Predictions from detection head
cls_scores = [torch.randn(2, 80, 100, 100)]  # Classification logits
bbox_preds = [torch.randn(2, 4, 100, 100)]   # Bbox predictions (l,t,r,b)
centernesses = [torch.randn(2, 1, 100, 100)] # Centerness logits

# Ground truth
targets = [
    {
        "boxes": torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]]),
        "labels": torch.tensor([1, 5]),
    },
    {
        "boxes": torch.tensor([[20, 20, 40, 40]]),
        "labels": torch.tensor([3]),
    },
]

# Compute loss
loss_dict = loss_fn(cls_scores, bbox_preds, centernesses, targets)
total_loss = loss_dict["loss"]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | `int` | Required | Number of object classes |
| `focal_alpha` | `float` | `0.25` | Focal loss alpha |
| `focal_gamma` | `float` | `2.0` | Focal loss gamma |
| `cls_weight` | `float` | `1.0` | Classification loss weight (λ_cls) |
| `reg_weight` | `float` | `1.0` | Regression loss weight (λ_reg) |
| `centerness_weight` | `float` | `1.0` | Centerness loss weight (λ_centerness) |

### Returns

Dictionary with:
- `"loss"`: Total weighted loss
- `"cls_loss"`: Classification loss (before weighting)
- `"reg_loss"`: Regression loss (before weighting)
- `"centerness_loss"`: Centerness loss (before weighting)

### How It Works

1. **Target Assignment**:
   - Assigns ground truth boxes to FPN levels based on size
   - Computes target classification labels
   - Computes target bbox offsets (l, t, r, b)
   - Computes target centerness values

2. **Loss Computation**:
   - **Classification**: Focal loss on all locations (positive + negative)
   - **Regression**: GIoU loss on positive locations only
   - **Centerness**: BCE loss on positive locations only

3. **Normalization**:
   - Losses are normalized by the number of positive samples
   - Prevents loss explosion when there are few objects

### Loss Weight Tuning

| Scenario | cls_weight | reg_weight | centerness_weight |
|----------|------------|------------|-------------------|
| Standard (recommended) | 1.0 | 1.0 | 1.0 |
| Poor localization | 1.0 | 2.0 | 1.0 |
| Missing detections | 2.0 | 1.0 | 1.0 |
| False positives | 1.0 | 1.0 | 2.0 |
| Small objects | 1.0 | 2.0 | 0.5 |

### Usage in Training

```python
from autotimm import ObjectDetector

model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    # Loss configuration
    focal_alpha=0.25,
    focal_gamma=2.0,
    cls_loss_weight=1.0,
    reg_loss_weight=1.0,
    centerness_loss_weight=1.0,
)

# FCOSLoss is automatically created and used during training
```

### Target Assignment Strategy

**Regression Range Assignment:**

| FPN Level | Stride | Default Range | Object Size |
|-----------|--------|---------------|-------------|
| P3 | 8 | (-1, 64) | 0-64px |
| P4 | 16 | (64, 128) | 64-128px |
| P5 | 32 | (128, 256) | 128-256px |
| P6 | 64 | (256, 512) | 256-512px |
| P7 | 128 | (512, ∞) | >512px |

Objects are assigned to the FPN level whose regression range best matches the object's max dimension.

### Custom Regression Ranges

For datasets with specific object size distributions:

```python
loss_fn = FCOSLoss(
    num_classes=80,
    strides=(8, 16, 32, 64, 128),
    regress_ranges=(
        (-1, 32),          # P3: very small objects
        (32, 64),          # P4
        (64, 128),         # P5
        (128, 256),        # P6
        (256, float("inf")),  # P7
    ),
)
```
