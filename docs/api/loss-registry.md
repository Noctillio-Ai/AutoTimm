# Loss Registry API

The Loss Registry provides a centralized system for discovering, accessing, and managing loss functions across AutoTimm.

## Overview

The Loss Registry enables you to:

- **Discover** all available loss functions by task
- **Access** built-in losses by name instead of importing
- **Register** custom losses for reuse across your project
- **Configure** models with flexible loss function options

## Core Functions

### list_available_losses

List all available loss functions or filter by task.

```python
from autotimm.losses import list_available_losses

# List all losses
all_losses = list_available_losses()
# ['bce', 'bce_with_logits', 'centerness', 'combined_segmentation', 
#  'cross_entropy', 'dice', 'fcos', 'focal', ...]

# List by task
seg_losses = list_available_losses(task="segmentation")
# ['combined_segmentation', 'dice', 'focal_pixelwise', 'mask', 'tversky']

det_losses = list_available_losses(task="detection")
# ['centerness', 'fcos', 'focal', 'giou']

cls_losses = list_available_losses(task="classification")
# ['bce', 'bce_with_logits', 'cross_entropy', 'mse', 'nll']
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | str \| None | None | Filter by task: `'classification'`, `'detection'`, or `'segmentation'`. If `None`, returns all losses. |

**Returns:** `list[str]` - List of loss function names

---

### get_loss_registry

Get the global loss registry instance for advanced operations.

```python
from autotimm.losses import get_loss_registry

registry = get_loss_registry()

# Check if a loss exists
if registry.has_loss("dice"):
    print("Dice loss is available")

# Get loss info organized by task
info = registry.get_loss_info()
# {
#     'classification': ['bce', 'cross_entropy', ...],
#     'detection': ['focal', 'giou', ...],
#     'segmentation': ['dice', 'tversky', ...]
# }

# Create a loss instance
dice_loss = registry.get_loss("dice", num_classes=10, smooth=1.0)
```

**Returns:** `LossRegistry` - The global registry instance

---

### register_custom_loss

Register a custom loss function globally for use across your project.

```python
from autotimm.losses import register_custom_loss
import torch.nn as nn

class FocalTverskyLoss(nn.Module):
    """Custom Focal-Tversky loss for medical imaging."""
    
    def __init__(self, num_classes: int, alpha: float = 0.7, 
                 beta: float = 0.3, gamma: float = 0.75):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, logits, targets):
        # Implementation here
        return loss_value

# Register the loss
register_custom_loss(
    name="focal_tversky",
    loss_class=FocalTverskyLoss,
    alias="ft"  # Optional short alias
)

# Now use it anywhere by name
from autotimm import SemanticSegmentor

model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=2,
    loss_fn="focal_tversky",  # or "ft"
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Name for the loss function |
| `loss_class` | type[nn.Module] | The loss class (must be a subclass of `nn.Module`) |
| `alias` | str \| None | Optional short alias for the loss |

---

## LossRegistry Class

### Methods

#### `get_loss(name, **kwargs)`

Create a loss function instance from the registry.

```python
registry = get_loss_registry()

# Basic usage
dice = registry.get_loss("dice", num_classes=10)

# With custom parameters
combined = registry.get_loss(
    "combined_segmentation",
    num_classes=19,
    ce_weight=0.5,
    dice_weight=1.5,
    ignore_index=255
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Name or alias of the loss function |
| `**kwargs` | Any | Arguments to pass to the loss constructor |

**Returns:** `nn.Module` - Instantiated loss function

**Raises:** `ValueError` if the loss name is not found

---

#### `has_loss(name)`

Check if a loss function is registered.

```python
registry = get_loss_registry()

if registry.has_loss("dice"):
    print("Dice loss available!")

if registry.has_loss("my_custom_loss"):
    print("Custom loss registered!")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Name or alias of the loss function |

**Returns:** `bool` - `True` if the loss exists, `False` otherwise

---

#### `list_losses(task=None)`

List available loss functions, optionally filtered by task.

```python
registry = get_loss_registry()

# All losses
all_losses = registry.list_losses()

# By task
seg_losses = registry.list_losses(task="segmentation")
```

Same as `list_available_losses()` function.

---

#### `get_loss_info()`

Get information about all registered losses organized by task.

```python
registry = get_loss_registry()
info = registry.get_loss_info()

for task, losses in info.items():
    print(f"{task}: {', '.join(losses)}")

# Output:
# classification: bce, bce_with_logits, cross_entropy, mse, nll
# detection: centerness, fcos, focal, giou
# segmentation: combined_segmentation, dice, focal_pixelwise, mask, tversky
```

**Returns:** `dict[str, list[str]]` - Dictionary mapping task names to loss lists

---

#### `register_loss(name, loss_class, alias=None)`

Register a custom loss function (same as `register_custom_loss()`).

```python
registry = get_loss_registry()

registry.register_loss(
    name="my_loss",
    loss_class=MyLossClass,
    alias="ml"
)
```

---

## Using Losses with Models

All AutoTimm models now support the `loss_fn` parameter for flexible loss configuration.

### ImageClassifier

```python
from autotimm import ImageClassifier

# By name
model = ImageClassifier(
    backbone="resnet18",
    num_classes=10,
    loss_fn="cross_entropy"  # Use from registry
)

# Custom instance
import torch.nn as nn
custom_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

model = ImageClassifier(
    backbone="resnet18",
    num_classes=10,
    loss_fn=custom_loss
)
```

---

### SemanticSegmentor

```python
from autotimm import SemanticSegmentor

# By name
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    loss_fn="dice"  # From registry
)

# From registry with parameters
from autotimm.losses import get_loss_registry

registry = get_loss_registry()
combined_loss = registry.get_loss(
    "combined_segmentation",
    num_classes=19,
    ce_weight=1.0,
    dice_weight=1.5
)

model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    loss_fn=combined_loss
)

# Backward compatibility: loss_type still works
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    loss_type="combined"  # Old way
)
```

---

### ObjectDetector

```python
from autotimm import ObjectDetector

# Separate losses for classification and regression
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    cls_loss_fn="focal",      # Classification loss
    reg_loss_fn="giou",       # Regression loss
)

# Custom loss instances
from autotimm.losses import FocalLoss, GIoULoss

focal = FocalLoss(alpha=0.3, gamma=2.5)
giou = GIoULoss()

model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    cls_loss_fn=focal,
    reg_loss_fn=giou,
)
```

---

### InstanceSegmentor

```python
from autotimm import InstanceSegmentor

# Configure all three losses
model = InstanceSegmentor(
    backbone="resnet50",
    num_classes=80,
    cls_loss_fn="focal",      # Classification
    reg_loss_fn="giou",       # Regression
    mask_loss_fn="mask",      # Mask segmentation
)
```

---

## Complete Examples

### Example 1: Comparing Multiple Losses

```python
from autotimm import SemanticSegmentor, AutoTrainer, SegmentationDataModule
from autotimm.losses import list_available_losses

# List segmentation losses
seg_losses = list_available_losses(task="segmentation")
print(f"Available: {seg_losses}")

# Create models with different losses
models = {}
for loss_name in ["dice", "focal_pixelwise", "combined_segmentation"]:
    models[loss_name] = SemanticSegmentor(
        backbone="resnet50",
        num_classes=19,
        loss_fn=loss_name,
    )

# Train and compare
data = SegmentationDataModule(data_dir="./data", image_size=512)

for loss_name, model in models.items():
    print(f"Training with {loss_name}...")
    trainer = AutoTrainer(max_epochs=10)
    trainer.fit(model, datamodule=data)
```

### Example 2: Custom Weighted Loss

```python
import torch
import torch.nn as nn
from autotimm import ImageClassifier
from autotimm.losses import register_custom_loss

class ClassBalancedCrossEntropy(nn.Module):
    """Cross-entropy with class balancing for imbalanced datasets."""
    
    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, input, target):
        return self.ce(input, target)

# Register globally
register_custom_loss("balanced_ce", ClassBalancedCrossEntropy)

# Use in model
weights = torch.tensor([1.0, 2.0, 1.5, 3.0, 2.5])
model = ImageClassifier(
    backbone="resnet18",
    num_classes=5,
    loss_fn="balanced_ce",  # Use by name
)
```

### Example 3: Medical Imaging Custom Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from autotimm import SemanticSegmentor
from autotimm.losses import register_custom_loss

class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss for highly imbalanced medical image segmentation."""
    
    def __init__(self, num_classes: int, alpha: float = 0.7, 
                 beta: float = 0.3, gamma: float = 0.75, 
                 ignore_index: int = 255):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha    # Weight for FN
        self.beta = beta      # Weight for FP
        self.gamma = gamma    # Focal parameter
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        valid_mask = targets != self.ignore_index
        
        # One-hot encode
        targets_oh = F.one_hot(
            targets.clamp(0, self.num_classes - 1), 
            num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()
        
        # Apply mask
        valid_mask = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask
        targets_oh = targets_oh * valid_mask
        
        # Tversky index
        tp = (probs * targets_oh).sum(dim=(2, 3))
        fp = (probs * (1 - targets_oh)).sum(dim=(2, 3))
        fn = ((1 - probs) * targets_oh).sum(dim=(2, 3))
        
        tversky = tp / (tp + self.alpha * fn + self.beta * fp + 1e-7)
        
        # Focal modulation
        focal_tversky = torch.pow(1 - tversky, self.gamma)
        
        return focal_tversky.mean()

# Register
register_custom_loss("focal_tversky", FocalTverskyLoss, alias="ft")

# Use for binary tumor segmentation
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=2,
    loss_fn="focal_tversky",  # Emphasizes rare tumor pixels
)
```

---

## Migration Guide

### From loss_type to loss_fn

The old `loss_type` parameter still works but `loss_fn` is recommended:

**Old way:**
```python
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    loss_type="combined",  # String identifier
    ce_weight=1.0,
    dice_weight=1.5,
)
```

**New way (recommended):**
```python
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    loss_fn="combined_segmentation",  # From registry
    ce_weight=1.0,
    dice_weight=1.5,
)
```

**Benefits:**
- :material-check-circle:{ .success } Access to all 14+ built-in losses
- :material-check-circle:{ .success } Use custom losses easily
- :material-check-circle:{ .success } Better type hints and IDE support
- :material-check-circle:{ .success } Consistent API across all models

---

## See Also

- [Loss Function Comparison](../user-guide/training/loss-comparison.md) - Detailed comparison of available losses
- [Custom Loss Functions Example](../../examples/utilities/custom_loss_functions.py) - Complete working examples
- [API Reference: Losses](../api/losses.md) - Individual loss function documentation
