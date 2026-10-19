# AutoTimmCLI

YAML-config-driven command-line interface built on LightningCLI.

## AutoTimmCLI

A subclass of `LightningCLI` that uses `AutoTrainer` by default and discovers all AutoTimm task and data module classes automatically.

### API Reference

::: autotimm.cli.AutoTimmCLI
    options:
      show_source: true
      members:
        - __init__

### Usage

```bash
# Train with a YAML config
autotimm fit --config config.yaml

# Or via python module
python -m autotimm fit --config config.yaml

# Override parameters
autotimm fit --config config.yaml --trainer.max_epochs 20

# Show help
autotimm --help
autotimm fit --help
```

### Subcommands

| Subcommand | Description |
|------------|-------------|
| `fit` | Train the model |
| `validate` | Run one validation epoch |
| `test` | Run one test epoch |
| `predict` | Run inference |

---

## main()

Entry point function for the CLI.

::: autotimm.cli.main
    options:
      show_source: true

### Usage

Called automatically when using the `autotimm` command or `python -m autotimm`.

Can also be called programmatically:

```python
from autotimm.cli import main

main()
```

---

## Config File Format

The CLI expects a YAML config with `model`, `data`, and `trainer` sections:

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

trainer:
  max_epochs: 10
  accelerator: auto
```

### Supported Model Classes

| Class | Import Path |
|-------|-------------|
| `ImageClassifier` | `autotimm.ImageClassifier` |
| `ObjectDetector` | `autotimm.ObjectDetector` |
| `YOLOXDetector` | `autotimm.YOLOXDetector` |
| `SemanticSegmentor` | `autotimm.SemanticSegmentor` |
| `InstanceSegmentor` | `autotimm.InstanceSegmentor` |

### Supported Data Modules

| Class | Import Path |
|-------|-------------|
| `ImageDataModule` | `autotimm.ImageDataModule` |
| `MultiLabelImageDataModule` | `autotimm.MultiLabelImageDataModule` |
| `DetectionDataModule` | `autotimm.DetectionDataModule` |
| `SegmentationDataModule` | `autotimm.SegmentationDataModule` |
| `InstanceSegmentationDataModule` | `autotimm.InstanceSegmentationDataModule` |

---

## Standalone CLI Modules

In addition to the `autotimm` CLI, AutoTimm provides standalone CLI modules for specific tasks:

### export_jit — TorchScript Export

Export a trained checkpoint to TorchScript (JIT) format:

```bash
python -m autotimm.export.export_jit \
    --checkpoint path/to/checkpoint.ckpt \
    --output model.pt \
    --task-class ImageClassifier \
    --input-size 224
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--checkpoint` | Yes | — | Path to `.ckpt` file |
| `--output` | Yes | — | Output `.pt` file path |
| `--task-class` | No | `ImageClassifier` | Task class name |
| `--input-size` | No | `224` | Input image size (auto-detected from hparams) |

See also: [Export API Reference](export.md)

### interpret_cli — Model Interpretation

Run interpretation methods on a trained checkpoint from the command line:

```bash
python -m autotimm.cli.interpret_cli \
    --checkpoint path/to/checkpoint.ckpt \
    --image path/to/image.jpg \
    --methods gradcam,gradcampp,integrated_gradients \
    --output-dir ./interpretations \
    --task-class ImageClassifier
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--checkpoint` | Yes | — | Path to `.ckpt` file |
| `--image` | Yes | — | Path to input image |
| `--methods` | No | All 6 methods | Comma-separated method names |
| `--output-dir` | Yes | — | Directory for output heatmap PNGs |
| `--task-class` | No | `ImageClassifier` | Task class name |

**Output:** JSON to stdout with heatmap file paths, predicted class, and per-method errors.

```json
{
  "results": {"gradcam": "/path/to/gradcam.png", ...},
  "predicted_class": 5,
  "errors": {}
}
```

**Available methods:** `gradcam`, `gradcampp`, `integrated_gradients`, `smoothgrad`, `attention_rollout`, `attention_flow`

---

See also: [CLI User Guide](../user-guide/training/cli.md) | [AutoTrainer](trainer.md) | [Export API](export.md) | [Example Configs](https://github.com/Noctillio-Ai/AutoTimm/tree/main/examples/cli)
