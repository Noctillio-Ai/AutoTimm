# Data Loading Issues

Common data loading problems and solutions.

## Corrupted or Missing Images

```python
import autotimm as at  # recommended alias
# Enable validation to skip corrupted images
from autotimm import ImageDataModule

data = ImageDataModule(
    data_dir="./data",
)
```

## Dataset Not Found

```python
import os

# Verify data directory structure
print("Data directory contents:")
print(os.listdir("./data"))

# For COCO format detection
# Expected structure:
# data/
#   ├── images/
#   │   ├── train/
#   │   └── val/
#   └── annotations/
#       ├── instances_train.json
#       └── instances_val.json
```

## Annotation Format Issues

**Symptoms:** KeyError, ValueError when loading annotations

**Solutions:**

```python
# For COCO format, verify structure
import json

with open("data/annotations/instances_train.json") as f:
    coco_data = json.load(f)

# Check required keys
required_keys = ["images", "annotations", "categories"]
for key in required_keys:
    assert key in coco_data, f"Missing key: {key}"

# Verify image IDs match
image_ids = {img["id"] for img in coco_data["images"]}
ann_image_ids = {ann["image_id"] for ann in coco_data["annotations"]}
print(f"Images: {len(image_ids)}, Annotated: {len(ann_image_ids)}")
```

## Class Imbalance Warnings

```python
from collections import Counter

# Check class distribution
def check_class_distribution(datamodule):
    train_labels = []
    for batch in datamodule.train_dataloader():
        labels = batch["labels"] if isinstance(batch, dict) else batch[1]
        train_labels.extend(labels.tolist())

    counts = Counter(train_labels)
    print("Class distribution:", counts)

    # If imbalanced, use weighted loss
    class_weights = [1.0 / count for count in counts.values()]
    return class_weights

# Apply weighted loss
from autotimm import ImageClassifier
class_weights = check_class_distribution(data)
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights)),
)
```

## Segmentation Masks Not Loading

**Symptoms:** Empty masks, incorrect pixel values, shape mismatches

**Common Causes:**

```python
# Check mask format
from PIL import Image
import numpy as np

mask = Image.open("path/to/mask.png")
print(f"Mode: {mask.mode}")  # Should be 'L' (grayscale)
print(f"Unique values: {np.unique(np.array(mask))}")  # Should be [0, 1, ..., num_classes-1, 255]
```

**Solutions:**

1. **Masks are RGB instead of single-channel:**
```python
# Convert RGB masks to single-channel
mask_rgb = Image.open("mask.png")
mask_gray = mask_rgb.convert('L')
mask_gray.save("mask_fixed.png")
```

2. **Pixel values out of range:**
```python
# Verify pixel values are [0, num_classes-1] and 255
mask_array = np.array(Image.open("mask.png"))
assert mask_array.max() <= num_classes or mask_array.max() == 255
assert mask_array.min() >= 0
```

3. **Filename mismatch:**
```python
# Verify images and masks have matching names
import os
from pathlib import Path

image_dir = Path("data/train/images")
mask_dir = Path("data/train/masks")

image_stems = {p.stem for p in image_dir.glob("*")}
mask_stems = {p.stem for p in mask_dir.glob("*")}

missing_masks = image_stems - mask_stems
print(f"Images without masks: {missing_masks}")
```

## Slow Segmentation Data Loading

**Symptoms:** Long wait times between epochs, low GPU utilization

**Solutions:**

1. **Increase num_workers:**
```python
import os
data = SegmentationDataModule(
    data_dir="./data",
    num_workers=min(os.cpu_count(), 8),  # Up to CPU cores
)
```

2. **Reduce image size:**
```python
data = SegmentationDataModule(
    data_dir="./data",
    image_size=512,  # Try 384 or 256 if still slow
)
```

3. **Use SSD storage:**
- Move dataset to SSD instead of HDD
- Use faster storage for better I/O performance

4. **Enable persistent workers:**
```python
data = SegmentationDataModule(
    data_dir="./data",
    num_workers=4,
    persistent_workers=True,  # Keep workers alive between epochs
)
```

## Segmentation Transform Errors

**Symptoms:** Shape mismatches, type errors, mask corruption

**Solutions:**

1. **For albumentations:**
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Correct order: transforms → normalize → ToTensorV2
train_transforms = A.Compose([
    A.RandomCrop(512, 512),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),  # Must be last
])
```

2. **Don't mix backends:**
```python
# WRONG: mixing torchvision and albumentations
import torchvision.transforms as T
import albumentations as A

# Do NOT do this:
transforms = T.Compose([...])  # torchvision
transforms = A.Compose([...])  # albumentations

# RIGHT: pick one backend
transforms = A.Compose([...])  # Use albumentations for segmentation
```

3. **Verify mask transforms:**
```python
# Masks should be transformed with nearest-neighbor interpolation
# Albumentations handles this automatically, just verify:
transforms = A.Compose([
    A.Resize(512, 512),  # Automatically uses cv2.INTER_NEAREST for masks
    ToTensorV2(),
])
```

## Related Issues

- [Augmentation](augmentation.md) - Transform and augmentation errors
- [Slow Training](../performance/slow-training.md) - Data loading bottlenecks
- [Profiling](../performance/profiling.md) - Performance analysis
