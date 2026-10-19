# CLI Examples

Train AutoTimm models from the command line using YAML configuration files.

## Overview

The AutoTimm CLI is built on [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) and supports `fit`, `validate`, `test`, and `predict` subcommands.

```bash
# Install (jsonargparse included automatically)
pip install autotimm
```

## Classification

**Config:** [`examples/cli/classification.yaml`](https://github.com/Noctillio-Ai/AutoTimm/blob/main/examples/cli/classification.yaml)

```yaml
model:
  class_path: autotimm.ImageClassifier
  init_args:
    backbone: resnet18
    num_classes: 10

data:
  class_path: autotimm.ImageDataModule
  init_args:
    dataset_name: CIFAR10
    data_dir: ./data
    batch_size: 32
    image_size: 224
    num_workers: 4

trainer:
  max_epochs: 10
  accelerator: auto
  devices: auto
  precision: 32
  enable_checkpointing: false
  logger: false
  tuner_config: false
```

```bash
# Train
autotimm fit --config examples/cli/classification.yaml

# Quick smoke test
autotimm fit --config examples/cli/classification.yaml --trainer.fast_dev_run true

# Override learning rate
autotimm fit --config examples/cli/classification.yaml --model.init_args.lr 0.001
```

## Object Detection

**Config:** [`examples/cli/detection.yaml`](https://github.com/Noctillio-Ai/AutoTimm/blob/main/examples/cli/detection.yaml)

```yaml
model:
  class_path: autotimm.ObjectDetector
  init_args:
    backbone: resnet50
    num_classes: 80
    detection_arch: fcos
    fpn_channels: 256
    lr: 0.01

data:
  class_path: autotimm.DetectionDataModule
  init_args:
    data_dir: ./coco
    image_size: 640
    batch_size: 8
    num_workers: 4

trainer:
  max_epochs: 50
  accelerator: auto
  devices: auto
  precision: 32
  enable_checkpointing: false
  logger: false
  tuner_config: false
```

```bash
autotimm fit --config examples/cli/detection.yaml
```

## Semantic Segmentation

**Config:** [`examples/cli/segmentation.yaml`](https://github.com/Noctillio-Ai/AutoTimm/blob/main/examples/cli/segmentation.yaml)

```yaml
model:
  class_path: autotimm.SemanticSegmentor
  init_args:
    backbone: resnet50
    num_classes: 19
    head_type: deeplabv3plus
    lr: 0.01

data:
  class_path: autotimm.SegmentationDataModule
  init_args:
    data_dir: ./cityscapes
    format: cityscapes
    image_size: 512
    batch_size: 4
    num_workers: 4

trainer:
  max_epochs: 100
  accelerator: auto
  devices: auto
  precision: 32
  enable_checkpointing: false
  logger: false
  tuner_config: false
```

```bash
autotimm fit --config examples/cli/segmentation.yaml
```

## Common Patterns

### Override Any Parameter

```bash
# Change backbone and learning rate
autotimm fit --config examples/cli/classification.yaml \
    --model.init_args.backbone efficientnet_b0 \
    --model.init_args.lr 0.0005

# Change trainer settings
autotimm fit --config examples/cli/classification.yaml \
    --trainer.max_epochs 50 \
    --trainer.precision "bf16-mixed"
```

### Validate and Test

```bash
autotimm validate --config config.yaml --ckpt_path path/to/checkpoint.ckpt
autotimm test --config config.yaml --ckpt_path path/to/checkpoint.ckpt
```

### Print Resolved Config

```bash
autotimm fit --config config.yaml --print_config
```

### Use HuggingFace Hub Backbones

```yaml
model:
  class_path: autotimm.ImageClassifier
  init_args:
    backbone: "hf-hub:timm/convnext_base.fb_in22k_ft_in1k"
    num_classes: 100
```

---

See also: [CLI User Guide](../../user-guide/training/cli.md) | [CLI API Reference](../../api/cli.md)
