# Dataset Preparation

This guide covers how to prepare datasets for all tasks supported by AutoTimm: image classification, object detection, semantic segmentation, and instance segmentation.

## Image Classification

### ImageFolder Structure

AutoTimm uses PyTorch's `ImageFolder` format for classification datasets. Organize your images with class names as subdirectories:

```
dataset/
├── train/
│   ├── class_a/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   ├── class_b/
│   │   ├── image001.jpg
│   │   └── ...
│   └── class_c/
│       └── ...
├── val/
│   ├── class_a/
│   │   └── ...
│   ├── class_b/
│   │   └── ...
│   └── class_c/
│       └── ...
└── test/  # Optional
    ├── class_a/
    │   └── ...
    └── ...
```

### Using ImageDataModule

```python
from autotimm import ImageDataModule

# Local ImageFolder dataset
data = ImageDataModule(
    data_dir="./dataset",
    image_size=224,
    batch_size=32,
    num_workers=4,
)

# Built-in torchvision datasets
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",  # Auto-downloads if not present
    image_size=224,
    batch_size=32,
)
```

### Supported Image Formats

- **JPEG** (`.jpg`, `.jpeg`) - Recommended for photos
- **PNG** (`.png`) - Good for images with transparency or sharp edges
- **WebP** (`.webp`) - Smaller file size, good quality
- **BMP** (`.bmp`) - Uncompressed
- **GIF** (`.gif`) - Single frame only

### Label Mapping

Classes are automatically assigned numeric labels based on alphabetical folder order:

```
train/
├── airplane/  → 0
├── bird/      → 1
└── car/       → 2
```

To get the class mapping:

```python
data = ImageDataModule(data_dir="./dataset", image_size=224, batch_size=32)
data.setup("fit")

# Get class names
class_names = data.train_dataset.classes
print(class_names)  # ['airplane', 'bird', 'car']

# Get class to index mapping
class_to_idx = data.train_dataset.class_to_idx
print(class_to_idx)  # {'airplane': 0, 'bird': 1, 'car': 2}
```

---

## Object Detection

### COCO Format

AutoTimm expects detection datasets in COCO format, the standard for object detection.

#### Directory Structure

```
coco_dataset/
├── train2017/              # Training images
│   ├── 000000000001.jpg
│   ├── 000000000002.jpg
│   └── ...
├── val2017/                # Validation images
│   ├── 000000000001.jpg
│   └── ...
├── test2017/               # Test images (optional)
│   └── ...
└── annotations/
    ├── instances_train2017.json
    ├── instances_val2017.json
    └── instances_test2017.json  # Optional
```

#### COCO JSON Schema

```json
{
  "info": {
    "description": "My Dataset",
    "version": "1.0",
    "year": 2024
  },
  "licenses": [],
  "images": [
    {
      "id": 1,
      "file_name": "000000000001.jpg",
      "width": 640,
      "height": 480
    },
    {
      "id": 2,
      "file_name": "000000000002.jpg",
      "width": 800,
      "height": 600
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 300],
      "area": 60000,
      "iscrowd": 0
    },
    {
      "id": 2,
      "image_id": 1,
      "category_id": 2,
      "bbox": [300, 100, 150, 200],
      "area": 30000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "person", "supercategory": "human"},
    {"id": 2, "name": "car", "supercategory": "vehicle"}
  ]
}
```

**Important Notes:**

- `bbox` format is `[x, y, width, height]` (top-left corner + dimensions)
- `category_id` is 1-indexed in COCO format
- `area` should be `width * height` of the bounding box
- `iscrowd`: 0 for normal annotations, 1 for crowd regions

### Converting from YOLO Format

YOLO uses a different annotation format with one `.txt` file per image:

```
# YOLO format: class_id x_center y_center width height (normalized 0-1)
0 0.5 0.5 0.2 0.3
1 0.3 0.4 0.1 0.15
```

**Conversion script:**

```python
import json
import os
from PIL import Image


def yolo_to_coco(images_dir, labels_dir, class_names, output_path):
    """Convert YOLO format to COCO format."""
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i + 1, "name": name} for i, name in enumerate(class_names)
        ],
    }

    annotation_id = 1
    image_files = sorted(
        f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))
    )

    for image_id, image_file in enumerate(image_files, start=1):
        # Get image dimensions
        image_path = os.path.join(images_dir, image_file)
        with Image.open(image_path) as img:
            width, height = img.size

        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_file,
                "width": width,
                "height": height,
            }
        )

        # Read YOLO labels
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        box_width = float(parts[3]) * width
                        box_height = float(parts[4]) * height

                        # Convert to COCO format (x, y, width, height)
                        x = x_center - box_width / 2
                        y = y_center - box_height / 2

                        coco["annotations"].append(
                            {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": class_id + 1,  # 1-indexed
                                "bbox": [x, y, box_width, box_height],
                                "area": box_width * box_height,
                                "iscrowd": 0,
                            }
                        )
                        annotation_id += 1

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Converted {len(coco['images'])} images, {len(coco['annotations'])} annotations")


# Usage
yolo_to_coco(
    images_dir="./yolo_dataset/images/train",
    labels_dir="./yolo_dataset/labels/train",
    class_names=["person", "car", "dog"],
    output_path="./coco_dataset/annotations/instances_train2017.json",
)
```

### Converting from Pascal VOC Format

Pascal VOC uses XML annotation files:

```xml
<annotation>
  <filename>000001.jpg</filename>
  <size>
    <width>640</width>
    <height>480</height>
  </size>
  <object>
    <name>person</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>150</ymin>
      <xmax>300</xmax>
      <ymax>450</ymax>
    </bndbox>
  </object>
</annotation>
```

**Conversion script:**

```python
import json
import os
import xml.etree.ElementTree as ET


def voc_to_coco(images_dir, annotations_dir, class_names, output_path):
    """Convert Pascal VOC format to COCO format."""
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i + 1, "name": name} for i, name in enumerate(class_names)
        ],
    }

    class_to_id = {name: i + 1 for i, name in enumerate(class_names)}
    annotation_id = 1

    xml_files = sorted(f for f in os.listdir(annotations_dir) if f.endswith(".xml"))

    for image_id, xml_file in enumerate(xml_files, start=1):
        tree = ET.parse(os.path.join(annotations_dir, xml_file))
        root = tree.getroot()

        filename = root.find("filename").text
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        coco["images"].append(
            {
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
            }
        )

        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in class_to_id:
                continue

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            box_width = xmax - xmin
            box_height = ymax - ymin

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_to_id[class_name],
                    "bbox": [xmin, ymin, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)


# Usage
voc_to_coco(
    images_dir="./voc_dataset/JPEGImages",
    annotations_dir="./voc_dataset/Annotations",
    class_names=["person", "car", "dog"],
    output_path="./coco_dataset/annotations/instances_train2017.json",
)
```

---

## Semantic Segmentation

### PNG Mask Format

Semantic segmentation uses pixel-wise class labels stored as grayscale PNG images.

#### Directory Structure

```
segmentation_dataset/
├── images/
│   ├── train/
│   │   ├── image001.jpg
│   │   └── ...
│   └── val/
│       └── ...
└── masks/
    ├── train/
    │   ├── image001.png  # Same name, .png extension
    │   └── ...
    └── val/
        └── ...
```

#### Mask Format

- **Grayscale PNG**: Pixel values represent class indices (0, 1, 2, ...)
- **Ignore index**: Use 255 for pixels to ignore (e.g., boundaries)
- **Same dimensions**: Mask must match image dimensions

```python
import numpy as np
from PIL import Image

# Example: Create a segmentation mask
mask = np.zeros((480, 640), dtype=np.uint8)
mask[100:200, 100:300] = 1  # Class 1 region
mask[200:400, 200:500] = 2  # Class 2 region
mask[0:10, :] = 255  # Ignore boundary

# Save as PNG
Image.fromarray(mask).save("mask.png")
```

### Cityscapes Format

Cityscapes uses a specific naming convention:

```
cityscapes/
├── leftImg8bit/
│   ├── train/
│   │   └── aachen/
│   │       ├── aachen_000000_000019_leftImg8bit.png
│   │       └── ...
│   └── val/
│       └── ...
└── gtFine/
    ├── train/
    │   └── aachen/
    │       ├── aachen_000000_000019_gtFine_labelIds.png
    │       └── ...
    └── val/
        └── ...
```

**Cityscapes class mapping (19 classes):**

| ID | Class | Color |
|----|-------|-------|
| 0 | road | (128, 64, 128) |
| 1 | sidewalk | (244, 35, 232) |
| 2 | building | (70, 70, 70) |
| 3 | wall | (102, 102, 156) |
| ... | ... | ... |
| 255 | ignore | - |

### Pascal VOC Segmentation Format

```
voc_segmentation/
├── JPEGImages/
│   ├── 2007_000027.jpg
│   └── ...
└── SegmentationClass/
    ├── 2007_000027.png  # Colored masks
    └── ...
```

Pascal VOC uses colored masks that need to be converted to class indices:

```python
import numpy as np
from PIL import Image

# VOC color palette (21 classes)
VOC_COLORS = [
    [0, 0, 0],       # 0: background
    [128, 0, 0],     # 1: aeroplane
    [0, 128, 0],     # 2: bicycle
    [128, 128, 0],   # 3: bird
    # ... more classes
    [224, 224, 192], # 20: tvmonitor
]


def voc_color_to_class(colored_mask):
    """Convert VOC colored mask to class indices."""
    mask = np.array(colored_mask)
    class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

    for class_id, color in enumerate(VOC_COLORS):
        matches = np.all(mask == color, axis=-1)
        class_mask[matches] = class_id

    return class_mask


# Usage
colored = Image.open("colored_mask.png")
class_mask = voc_color_to_class(colored)
Image.fromarray(class_mask).save("class_mask.png")
```

### Using SegmentationDataModule

```python
from autotimm import SegmentationDataModule

data = SegmentationDataModule(
    data_dir="./segmentation_dataset",
    image_size=512,
    batch_size=8,
    num_workers=4,
)
```

---

## Instance Segmentation

### COCO Instance Format

Instance segmentation extends detection with per-instance masks. Masks can be stored as polygons or RLE (Run-Length Encoding).

#### Polygon Format

```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 300],
      "segmentation": [[100, 150, 300, 150, 300, 450, 100, 450]],
      "area": 60000,
      "iscrowd": 0
    }
  ]
}
```

- `segmentation`: List of polygons, each polygon is `[x1, y1, x2, y2, ...]`
- Multiple polygons for complex shapes (with holes)

#### RLE Format (for complex masks)

```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 300],
      "segmentation": {
        "counts": "encoded_string...",
        "size": [480, 640]
      },
      "area": 60000,
      "iscrowd": 0
    }
  ]
}
```

### Converting Masks to COCO Format

```python
import json
import numpy as np
from pycocotools import mask as mask_utils
from PIL import Image


def mask_to_polygon(binary_mask):
    """Convert binary mask to polygon points."""
    from skimage import measure

    contours = measure.find_contours(binary_mask, 0.5)
    polygons = []

    for contour in contours:
        if len(contour) < 3:
            continue
        # Flip x and y
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        polygons.append(segmentation)

    return polygons


def mask_to_rle(binary_mask):
    """Convert binary mask to RLE format."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def create_instance_annotation(
    annotation_id, image_id, category_id, binary_mask, use_rle=False
):
    """Create a COCO-format instance annotation."""
    # Get bounding box
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    bbox = [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]
    area = float(binary_mask.sum())

    if use_rle:
        segmentation = mask_to_rle(binary_mask)
    else:
        segmentation = mask_to_polygon(binary_mask)

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "segmentation": segmentation,
        "area": area,
        "iscrowd": 0,
    }
```

### Using InstanceSegmentationDataModule

```python
from autotimm import InstanceSegmentationDataModule

data = InstanceSegmentationDataModule(
    data_dir="./coco_instance",
    image_size=640,
    batch_size=4,
    num_workers=4,
)
```

---

## Data Validation

### Validation Script

```python
import json
import os
from PIL import Image


def validate_classification_dataset(data_dir):
    """Validate ImageFolder classification dataset."""
    errors = []
    stats = {"classes": 0, "images": 0}

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            if split != "test":
                errors.append(f"Missing {split} directory")
            continue

        classes = [
            d
            for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ]
        stats["classes"] = max(stats["classes"], len(classes))

        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            images = [
                f
                for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            stats["images"] += len(images)

            # Validate each image
            for img_file in images[:10]:  # Sample first 10
                try:
                    img_path = os.path.join(cls_dir, img_file)
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as e:
                    errors.append(f"Corrupt image: {img_path}: {e}")

    return errors, stats


def validate_coco_dataset(data_dir):
    """Validate COCO-format detection dataset."""
    errors = []
    stats = {}

    for split in ["train", "val"]:
        ann_file = os.path.join(
            data_dir, "annotations", f"instances_{split}2017.json"
        )
        img_dir = os.path.join(data_dir, f"{split}2017")

        if not os.path.exists(ann_file):
            errors.append(f"Missing annotation file: {ann_file}")
            continue

        with open(ann_file) as f:
            data = json.load(f)

        stats[split] = {
            "images": len(data["images"]),
            "annotations": len(data["annotations"]),
            "categories": len(data["categories"]),
        }

        # Validate images exist
        for img_info in data["images"][:100]:  # Sample first 100
            img_path = os.path.join(img_dir, img_info["file_name"])
            if not os.path.exists(img_path):
                errors.append(f"Missing image: {img_path}")

        # Validate annotations
        image_ids = {img["id"] for img in data["images"]}
        category_ids = {cat["id"] for cat in data["categories"]}

        for ann in data["annotations"]:
            if ann["image_id"] not in image_ids:
                errors.append(f"Annotation {ann['id']} references invalid image_id")
            if ann["category_id"] not in category_ids:
                errors.append(f"Annotation {ann['id']} references invalid category_id")
            if len(ann["bbox"]) != 4:
                errors.append(f"Annotation {ann['id']} has invalid bbox")

    return errors, stats


def validate_segmentation_dataset(images_dir, masks_dir):
    """Validate segmentation dataset."""
    errors = []
    stats = {"images": 0, "classes": set()}

    for split in ["train", "val"]:
        img_split = os.path.join(images_dir, split)
        mask_split = os.path.join(masks_dir, split)

        if not os.path.exists(img_split):
            errors.append(f"Missing images/{split} directory")
            continue

        for img_file in os.listdir(img_split):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            stats["images"] += 1

            # Check corresponding mask exists
            mask_file = os.path.splitext(img_file)[0] + ".png"
            mask_path = os.path.join(mask_split, mask_file)

            if not os.path.exists(mask_path):
                errors.append(f"Missing mask for {img_file}")
                continue

            # Validate dimensions match
            img = Image.open(os.path.join(img_split, img_file))
            mask = Image.open(mask_path)

            if img.size != mask.size:
                errors.append(
                    f"Size mismatch: {img_file} {img.size} vs mask {mask.size}"
                )

            # Collect unique classes
            mask_array = np.array(mask)
            stats["classes"].update(np.unique(mask_array).tolist())

    stats["classes"] = sorted(stats["classes"])
    return errors, stats


# Usage examples
if __name__ == "__main__":
    # Validate classification dataset
    errors, stats = validate_classification_dataset("./my_dataset")
    print(f"Classification: {stats}")
    if errors:
        print(f"Errors: {errors[:5]}")

    # Validate COCO dataset
    errors, stats = validate_coco_dataset("./coco")
    print(f"Detection: {stats}")
```

---

## Best Practices

### Train/Val/Test Splits

**Recommended split ratios:**

| Dataset Size | Train | Val | Test |
|--------------|-------|-----|------|
| < 1,000 images | 70% | 15% | 15% |
| 1,000 - 10,000 | 80% | 10% | 10% |
| > 10,000 | 90% | 5% | 5% |

**Splitting script:**

```python
import os
import random
import shutil


def split_dataset(source_dir, dest_dir, train_ratio=0.8, val_ratio=0.1):
    """Split images into train/val/test sets."""
    random.seed(42)

    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = [
            f
            for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(images)

        n_train = int(len(images) * train_ratio)
        n_val = int(len(images) * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        for split, split_images in splits.items():
            split_dir = os.path.join(dest_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)

            for img in split_images:
                src = os.path.join(class_dir, img)
                dst = os.path.join(split_dir, img)
                shutil.copy2(src, dst)


# Usage
split_dataset("./raw_images", "./dataset")
```

### Handling Imbalanced Data

```python
from autotimm import ImageDataModule

# Option 1: Use weighted sampling
data = ImageDataModule(
    data_dir="./imbalanced_dataset",
    image_size=224,
    batch_size=32,
    balanced_sampling=True,  # Oversample minority classes
)

# Option 2: Use class weights in loss
import torch

# Calculate class weights (inverse frequency)
class_counts = [1000, 500, 100]  # Example: 3 classes
total = sum(class_counts)
class_weights = torch.tensor([total / c for c in class_counts])
class_weights = class_weights / class_weights.sum()  # Normalize
```

### Data Augmentation Tips

| Task | Recommended Augmentations |
|------|--------------------------|
| Classification | RandomResizedCrop, HorizontalFlip, ColorJitter, RandAugment |
| Detection | RandomResizedCrop (with bbox transform), HorizontalFlip |
| Segmentation | RandomResizedCrop (with mask transform), HorizontalFlip, ColorJitter |

---

## See Also

- [Data Loading Overview](../data-loading/index.md) - Data module documentation
- [Image Classification Data](../data-loading/image-classification-data.md) - Classification details
- [Object Detection Data](../data-loading/object-detection-data.md) - Detection details
- [Segmentation Data](../data-loading/segmentation-data.md) - Segmentation details
