# Semantic Segmentation Inference

This guide covers how to perform inference with trained semantic segmentation models, including visualization, export, and batch processing.

## Overview

The [`segmentation_inference.py`](https://github.com/Noctillio-Ai/AutoTimm/blob/main/examples/logging_inference/segmentation_inference.py) script provides a comprehensive toolkit for semantic segmentation inference with the following features:

**Core Features:**

- Load trained segmentation models from checkpoints
- Single image and batch prediction
- Automatic preprocessing using model's data config
- Support for multiple datasets (Cityscapes, Pascal VOC, custom)

**Visualization:**

- Overlay segmentation masks on original images
- Adjustable transparency for overlays
- Pre-configured color palettes (Cityscapes, Pascal VOC)
- Create class legends with color boxes and labels

**Export Options:**

- Save colored or grayscale masks as PNG
- Export per-class pixel statistics to JSON
- Batch processing with comprehensive statistics

---

## Quick Start

### Basic Inference

```python
from examples.segmentation_inference import (
    load_model,
    predict_single_image,
    visualize_segmentation,
    CITYSCAPES_CLASSES,
    CITYSCAPES_COLORS,
)
import torch

# Load trained model
model = load_model(
    checkpoint_path="best-segmentor.ckpt",
    backbone="resnet50",
    num_classes=19,
    head_type="deeplabv3plus",
    image_size=512,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Predict on single image
result = predict_single_image(model, "street_scene.jpg")

# Result contains:
# - result["mask"]: (H, W) numpy array of class indices
# - result["probabilities"]: (C, H, W) numpy array of class probabilities
# - result["original_size"]: (width, height) tuple

print(f"Mask shape: {result['mask'].shape}")
print(f"Classes found: {sorted(set(result['mask'].flatten().tolist()))}")
```

### Visualizing Results

```python
from examples.segmentation_inference import visualize_segmentation

# Visualize with 50% transparency overlay
visualize_segmentation(
    image_path="street_scene.jpg",
    mask=result["mask"],
    output_path="output.jpg",
    color_palette=CITYSCAPES_COLORS,
    alpha=0.5,  # 0=transparent, 1=opaque
)

# Try different alpha values for different effects
for alpha in [0.3, 0.5, 0.7]:
    visualize_segmentation(
        "street_scene.jpg",
        result["mask"],
        f"output_alpha_{alpha}.jpg",
        color_palette=CITYSCAPES_COLORS,
        alpha=alpha,
    )
```

---

## Batch Processing

Process multiple images efficiently:

```python
from examples.segmentation_inference import predict_batch, export_to_json

# Process multiple images
image_paths = [
    "street1.jpg",
    "street2.jpg",
    "street3.jpg",
]

batch_results = predict_batch(
    model=model,
    image_paths=image_paths,
    batch_size=4,
)

# Visualize all results
for i, (path, result) in enumerate(zip(image_paths, batch_results)):
    visualize_segmentation(
        image_path=path,
        mask=result["mask"],
        output_path=f"batch_output_{i}.jpg",
        color_palette=CITYSCAPES_COLORS,
        alpha=0.5,
    )

# Export statistics for all images
masks = [r["mask"] for r in batch_results]
export_to_json(
    masks,
    "batch_statistics.json",
    image_paths=image_paths,
    class_names=CITYSCAPES_CLASSES,
)
```

---

## Export Options

### Export Masks as PNG

```python
from examples.segmentation_inference import export_mask_to_png

# Export colored mask (for visualization)
export_mask_to_png(
    result["mask"],
    "mask_colored.png",
    color_palette=CITYSCAPES_COLORS,
)

# Export grayscale mask (class indices, for further processing)
export_mask_to_png(
    result["mask"],
    "mask_grayscale.png",
    color_palette=None,  # No colors = grayscale
)
```

### Export Statistics to JSON

Get detailed per-class pixel counts and percentages:

```python
from examples.segmentation_inference import export_to_json

export_to_json(
    result["mask"],
    "statistics.json",
    class_names=CITYSCAPES_CLASSES,
)
```

**Output format:**

```json
{
  "statistics": {
    "road": {
      "class_idx": 0,
      "pixel_count": 125830,
      "percentage": 40.26
    },
    "building": {
      "class_idx": 2,
      "pixel_count": 89456,
      "percentage": 28.63
    },
    "vegetation": {
      "class_idx": 8,
      "pixel_count": 54320,
      "percentage": 17.38
    }
  }
}
```

---

## Color Palettes

### Cityscapes (19 classes)

```python
from examples.segmentation_inference import CITYSCAPES_CLASSES, CITYSCAPES_COLORS

# Classes:
# road, sidewalk, building, wall, fence, pole, traffic light,
# traffic sign, vegetation, terrain, sky, person, rider, car,
# truck, bus, train, motorcycle, bicycle

visualize_segmentation(
    "image.jpg",
    mask,
    "output.jpg",
    color_palette=CITYSCAPES_COLORS,
)
```

### Pascal VOC (21 classes)

```python
from examples.segmentation_inference import VOC_CLASSES, VOC_COLORS

# Classes:
# background, aeroplane, bicycle, bird, boat, bottle, bus, car,
# cat, chair, cow, dining table, dog, horse, motorbike, person,
# potted plant, sheep, sofa, train, tv/monitor

visualize_segmentation(
    "image.jpg",
    mask,
    "output.jpg",
    color_palette=VOC_COLORS,
)
```

### Custom Palettes

Define your own colors for custom datasets:

```python
# Define custom classes and colors
CUSTOM_CLASSES = ["background", "building", "road", "vegetation", "vehicle"]
CUSTOM_COLORS = [
    (0, 0, 0),        # black - background
    (128, 0, 0),      # maroon - building
    (128, 128, 128),  # gray - road
    (0, 128, 0),      # green - vegetation
    (0, 0, 255),      # blue - vehicle
]

# Use with inference
visualize_segmentation(
    "image.jpg",
    result["mask"],
    "output.jpg",
    color_palette=CUSTOM_COLORS,
    alpha=0.6,
)

# Create legend for your custom palette
from examples.segmentation_inference import create_legend

create_legend(
    CUSTOM_CLASSES,
    CUSTOM_COLORS,
    "legend.png",
)
```

---

## Creating Class Legends

Generate a legend image showing class names and colors:

```python
from examples.segmentation_inference import create_legend

# For Cityscapes
create_legend(
    CITYSCAPES_CLASSES,
    CITYSCAPES_COLORS,
    "cityscapes_legend.png",
)

# For Pascal VOC
create_legend(
    VOC_CLASSES,
    VOC_COLORS,
    "voc_legend.png",
)
```

---

## Complete Example

Here's a full workflow from loading a model to exporting results:

```python
import torch
from examples.segmentation_inference import (
    load_model,
    predict_single_image,
    predict_batch,
    visualize_segmentation,
    export_mask_to_png,
    export_to_json,
    create_legend,
    CITYSCAPES_CLASSES,
    CITYSCAPES_COLORS,
)

def main():
    # 1. Load trained model
    model = load_model(
        checkpoint_path="checkpoints/best-cityscapes.ckpt",
        backbone="resnet50",
        num_classes=19,
        head_type="deeplabv3plus",
        image_size=512,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")

    # 2. Single image inference
    print("\n=== Single Image Inference ===")
    result = predict_single_image(
        model=model,
        image_path="data/street.jpg",
    )

    print(f"Mask shape: {result['mask'].shape}")
    print(f"Classes found: {sorted(set(result['mask'].flatten().tolist()))}")

    # 3. Visualize with different alpha values
    print("\n=== Visualizing with different transparency ===")
    for alpha in [0.3, 0.5, 0.7]:
        visualize_segmentation(
            image_path="data/street.jpg",
            mask=result["mask"],
            output_path=f"outputs/overlay_alpha_{alpha}.jpg",
            color_palette=CITYSCAPES_COLORS,
            alpha=alpha,
        )

    # 4. Export masks
    print("\n=== Exporting masks ===")
    # Colored mask
    export_mask_to_png(
        result["mask"],
        "outputs/mask_colored.png",
        color_palette=CITYSCAPES_COLORS,
    )

    # Grayscale mask (class indices)
    export_mask_to_png(
        result["mask"],
        "outputs/mask_grayscale.png",
        color_palette=None,
    )

    # 5. Export statistics
    print("\n=== Exporting statistics ===")
    export_to_json(
        result["mask"],
        "outputs/statistics.json",
        class_names=CITYSCAPES_CLASSES,
    )

    # 6. Create legend
    print("\n=== Creating legend ===")
    create_legend(
        CITYSCAPES_CLASSES,
        CITYSCAPES_COLORS,
        "outputs/legend.png",
    )

    # 7. Batch inference
    print("\n=== Batch Inference ===")
    image_paths = [
        "data/street1.jpg",
        "data/street2.jpg",
        "data/street3.jpg",
    ]

    batch_results = predict_batch(
        model=model,
        image_paths=image_paths,
        batch_size=2,
    )

    # 8. Process batch results
    print("\n=== Processing batch results ===")
    for i, (path, result) in enumerate(zip(image_paths, batch_results)):
        visualize_segmentation(
            image_path=path,
            mask=result["mask"],
            output_path=f"outputs/batch_{i}.jpg",
            color_palette=CITYSCAPES_COLORS,
            alpha=0.5,
        )

    # 9. Export batch statistics
    masks = [r["mask"] for r in batch_results]
    export_to_json(
        masks,
        "outputs/batch_statistics.json",
        image_paths=image_paths,
        class_names=CITYSCAPES_CLASSES,
    )

    print("\n=== Inference complete! ===")
    print("Results saved to outputs/ directory")


if __name__ == "__main__":
    main()
```

---

## Running the Demo

The example script includes a standalone demo:

```bash
# Run the demo script
python examples/logging_inference/segmentation_inference.py
```

The demo demonstrates:
1. Model loading with and without checkpoints
2. Image preprocessing and data configuration
3. Single image inference
4. Visualization with color overlays
5. Mask export to PNG (colored and grayscale)
6. Statistics export to JSON
7. Legend creation

---

## Advanced Usage

### Using model.preprocess() Directly

If you prefer to use the model's built-in preprocessing:

```python
from PIL import Image
import torch

# Load image
image = Image.open("test.jpg").convert("RGB")

# Preprocess using model's config
input_tensor = model.preprocess(image)  # Returns (1, 3, H, W) tensor

# Predict
with torch.inference_mode():
    logits = model.predict(input_tensor)  # Returns (1, num_classes, H, W)

# Get class predictions
mask = logits.argmax(dim=1)[0].cpu().numpy()  # (H, W)

# Get probabilities
probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (C, H, W)
```

### Custom Preprocessing

For custom preprocessing pipelines:

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

# Define custom transform
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Apply to image
image = Image.open("test.jpg").convert("RGB")
image_np = np.array(image)
transformed = transform(image=image_np)
input_tensor = transformed["image"].unsqueeze(0)  # Add batch dimension

# Predict
with torch.inference_mode():
    logits = model(input_tensor)
    mask = logits.argmax(dim=1)[0].cpu().numpy()
```

---

## Model Loading Options

### From Checkpoint

```python
model = load_model(
    checkpoint_path="best-segmentor.ckpt",
    backbone="resnet50",
    num_classes=19,
    head_type="deeplabv3plus",
    image_size=512,
)
```

### Creating New Model (for testing)

```python
model = load_model(
    checkpoint_path=None,  # Creates untrained model
    backbone="resnet50",
    num_classes=19,
    head_type="deeplabv3plus",
    image_size=512,
)
```

### Direct Loading with SemanticSegmentor

```python
import autotimm as at  # recommended alias
from autotimm import SemanticSegmentor, MetricConfig, TransformConfig

metrics = [
    MetricConfig(
        name="mIoU",
        backend="torchmetrics",
        metric_class="JaccardIndex",
        params={
            "task": "multiclass",
            "num_classes": 19,
            "average": "macro",
            "ignore_index": 255,
        },
        stages=["val", "test"],
    ),
]

model = SemanticSegmentor.load_from_checkpoint(
    "checkpoint.ckpt",
    backbone="resnet50",
    num_classes=19,
    head_type="deeplabv3plus",
    compile_model=False,                                # skip compilation for inference
    metrics=metrics,                                    # not saved in checkpoint
    transform_config=TransformConfig(image_size=512),   # not saved in checkpoint
)
model.eval()
```

---

## Troubleshooting

For semantic segmentation inference issues, see the [Troubleshooting - Export & Inference](../../troubleshooting/deployment/export-inference.md) including:

- Mask size mismatch
- Out of memory
- Custom ignore index handling

---

## See Also

- [Semantic Segmentation Examples](../../examples/tasks/semantic-segmentation.md) - Training examples
- [Semantic Segmentation Model Guide](../models/semantic-segmentation.md) - Model architecture details
- [Segmentation Data Loading](../data-loading/segmentation-data.md) - Dataset preparation
- [Model Export Guide](model-export.md) - Export to TorchScript/ONNX
