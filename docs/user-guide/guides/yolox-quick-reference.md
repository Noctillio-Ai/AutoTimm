# YOLOX Quick Reference

Fast reference for using YOLOX in AutoTimm.

## Installation

```bash
pip install autotimm
```

## Basic Usage

### Official YOLOX (Recommended)

```python
from autotimm import YOLOXDetector, DetectionDataModule, AutoTrainer

model = YOLOXDetector(model_name="yolox-s", num_classes=80)
data = DetectionDataModule(data_dir="./coco", image_size=640, batch_size=64)
trainer = AutoTrainer(max_epochs=300, precision="16-mixed")
trainer.fit(model, datamodule=data)
```

### YOLOX-Style (Flexible Backbone)

```python
from autotimm import ObjectDetector, DetectionDataModule, AutoTrainer

model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    detection_arch="yolox",
    reg_loss_weight=5.0,
)
data = DetectionDataModule(data_dir="./coco", image_size=640, batch_size=16)
trainer = AutoTrainer(max_epochs=300)
trainer.fit(model, datamodule=data)
```

## Available Models

| Model | Params | mAP | Use Case |
|-------|--------|-----|----------|
| yolox-nano | 0.9M | 25.8 | Edge/Mobile |
| yolox-tiny | 5.1M | 32.8 | Resource-constrained |
| yolox-s | 9.0M | 40.5 | Balanced (recommended) |
| yolox-m | 25.3M | 47.2 | Medium |
| yolox-l | 54.2M | 50.1 | High accuracy |
| yolox-x | 99.1M | 51.5 | Maximum accuracy |

## List Models

```python
from autotimm import list_yolox_models, get_yolox_model_info

# List all models
models = list_yolox_models()

# Detailed table
list_yolox_models(verbose=True)

# Get specific model info
info = get_yolox_model_info("yolox-s")
```

## Official Settings

```python
model = YOLOXDetector(
    model_name="yolox-s",
    lr=0.01,              # Base LR for batch 64
    weight_decay=5e-4,
    optimizer="sgd",      # SGD with momentum
    scheduler="yolox",    # Warmup + cosine
    total_epochs=300,
    warmup_epochs=5,
    no_aug_epochs=15,
    reg_loss_weight=5.0,
)
```

## Custom Settings

```python
model = YOLOXDetector(
    model_name="yolox-s",
    lr=0.001,            # Custom LR
    optimizer="adamw",   # Different optimizer
    scheduler="cosine",  # Standard cosine
    total_epochs=100,    # Fewer epochs
)
```

## Inference

```python
import torch
from autotimm import YOLOXDetector

model = YOLOXDetector.load_from_checkpoint("checkpoint.ckpt", compile_model=False)
model.eval()

images = torch.randn(1, 3, 640, 640)
predictions = model.predict(images)

for pred in predictions:
    boxes = pred["boxes"]    # [N, 4]
    scores = pred["scores"]  # [N]
    labels = pred["labels"]  # [N]
```

## Multi-GPU

```python
trainer = AutoTrainer(
    max_epochs=300,
    devices=8,
    strategy="ddp",
    precision="16-mixed",
)

data = DetectionDataModule(
    data_dir="./coco",
    batch_size=8,  # Per GPU
)
```

## Learning Rate Scaling

```python
base_lr = 0.01  # For batch size 64
batch_size = 32
lr = base_lr * (batch_size / 64)  # Scale linearly
```

## Common Commands

```python
# List all YOLOX models
autotimm.list_yolox_models()

# List backbones
autotimm.list_yolox_backbones()

# List necks
autotimm.list_yolox_necks()

# Get model info
autotimm.get_yolox_model_info("yolox-s")

# Explore interactively
python examples/computer_vision/explore_yolox_models.py
```

## Examples

```bash
# Official YOLOX training
python examples/computer_vision/yolox_official.py --model-name yolox-s --batch-size 64

# YOLOX-style with timm backbone
python examples/computer_vision/object_detection_yolox.py --backbone resnet50

# Explore models
python examples/computer_vision/explore_yolox_models.py
```

## Troubleshooting

Common YOLOX-specific issues include:

- CUDA out of memory
- Slow training
- Poor performance
- Batch size and settings optimization

## Quick Comparison

| Feature | YOLOXDetector | ObjectDetector |
|---------|---------------|----------------|
| Backbone | CSPDarknet | Any timm model |
| Settings | Official | Flexible |
| Use for | Production | Experimentation |

## Links

- [Complete YOLOX Guide](../models/yolox-detector.md)
- [Official YOLOX Repo](https://github.com/Megvii-BaseDetection/YOLOX)
- [AutoTimm Docs](https://noctillio-ai.github.io/AutoTimm/)
- [Object Detector Guide](../models/object-detector.md)

## Cheat Sheet

```python
# Official YOLOX (one-liner)
from autotimm import YOLOXDetector, DetectionDataModule, AutoTrainer
AutoTrainer(max_epochs=300).fit(
    YOLOXDetector(model_name="yolox-s", num_classes=80),
    DetectionDataModule(data_dir="./coco", image_size=640, batch_size=64)
)

# List and explore
from autotimm import list_yolox_models, get_yolox_model_info
list_yolox_models(verbose=True)
info = get_yolox_model_info("yolox-s")

# Inference
model = YOLOXDetector.load_from_checkpoint("checkpoint.ckpt", compile_model=False)
predictions = model.predict(images)
```
