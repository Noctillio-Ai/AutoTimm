# HuggingFace Integration Overview

This guide compares the different approaches for using HuggingFace models with AutoTimm, PyTorch Lightning, and AutoTrainer.

## Integration Approaches

```mermaid
graph TD
    A[HuggingFace Models] --> A1[Access Model Zoo]
    A1 --> A2[Choose Integration]
    A2 --> B{Integration Type}
    
    B -->|Option 1| C[HF Hub timm]
    C --> C1[Search timm Models]
    C1 --> C2[hf-hub:timm/model]
    C2 --> F1[timm + hf_hub]
    F1 --> F1a[Load Pretrained]
    F1a --> F1b[timm.create_model]
    F1b --> G1[ImageClassifier]
    
    G1 --> G1a[Pass backbone]
    G1a --> G1b[Add Task Head]
    G1b --> G1c[Configure Metrics]
    G1c --> H1[AutoTrainer :material-check:]
    H1 --> H1a[Full Integration]
    H1a --> H1b[Easy Training]
    
    B -->|Option 2| D[HF Transformers Direct]
    D --> D1[Import Specific Model]
    D1 --> D2[ViTModel/DeiTModel]
    D2 --> F2[transformers]
    F2 --> F2a[Load Config]
    F2a --> F2b[Initialize Model]
    F2b --> G2[Custom LightningModule]
    
    G2 --> G2a[Define training_step]
    G2a --> G2b[Define configure_optimizers]
    G2b --> G2c[Add Task Head]
    G2c --> H2[AutoTrainer :material-check:]
    H2 --> H2a[Flexible Control]
    H2a --> H2b[Manual Setup]
    
    B -->|Option 3| E[HF Auto Classes]
    E --> E1[Generic Interface]
    E1 --> E2[AutoModel/AutoConfig]
    E2 --> F3[transformers]
    F3 --> F3a[Auto Detection]
    F3a --> F3b[Load Model]
    F3b --> G3[Custom LightningModule]
    
    G3 --> G3a[Define training_step]
    G3a --> G3b[Define configure_optimizers]
    G3b --> G3c[Add Task Head]
    G3c --> H3[AutoTrainer :material-check:]
    H3 --> H3a[Quick Prototyping]
    H3a --> H3b[Less Control]
    
    style C fill:#2196F3,stroke:#1976D2
    style D fill:#1976D2,stroke:#1565C0
    style E fill:#2196F3,stroke:#1976D2
    style G1 fill:#1976D2,stroke:#1565C0
    style H1 fill:#2196F3,stroke:#1976D2
    style H2 fill:#1976D2,stroke:#1565C0
    style H3 fill:#2196F3,stroke:#1976D2
```

## Quick Reference

AutoTimm provides three distinct ways to work with HuggingFace models:

| Approach | Best For | Libraries | AutoTrainer | Integration |
|----------|----------|-----------|-------------|-------------|
| **HF Hub timm** | CNNs, Production | timm, hf_hub | :material-check-circle: Full | Native |
| **HF Direct** | Vision Transformers, Custom | transformers | :material-check-circle: Full | Manual |
| **HF Auto** | Prototyping | transformers | :material-check-circle: Full | Manual |

**All approaches are fully compatible with PyTorch Lightning and AutoTrainer!**

## Three Integration Approaches

AutoTimm supports three distinct ways to work with HuggingFace models, each suited for different use cases.

### Approach 1: HF Hub timm Models (via AutoTimm)

**What it is:** Loading timm models from HuggingFace Hub using AutoTimm's native integration.

```python
import autotimm as at  # recommended alias
from autotimm import ImageClassifier

# Load timm model from HF Hub
model = ImageClassifier(
    backbone="hf-hub:timm/resnet50.a1_in1k",
    num_classes=10,
)
```

**Status:** :material-check-circle: **Fully Compatible** (Recommended for timm models)

**Requirements:**

- `timm` library
- `huggingface_hub` library
- Does NOT require `transformers`

**Best for:**

- CNN backbones (ResNet, EfficientNet, ConvNeXt)
- timm-compatible models on HF Hub
- Direct integration with AutoTimm tasks
- Production deployments with proven architectures

**Advantages:**

- Seamless AutoTimm integration
- Access to 1000+ timm models
- Version control via HF Hub
- Same API as standard timm models
- Works with all AutoTimm tasks

---

### Approach 2: HF Transformers Models (Direct Classes)

**What it is:** Using HuggingFace transformers models directly with specific model classes (no Auto classes).

```python
from transformers import ViTModel, ViTConfig
import pytorch_lightning as pl

class ViTClassifier(pl.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        # Use specific model class
        self.vit = ViTModel(ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            image_size=224,
            patch_size=16,
        ))
        self.classifier = torch.nn.Linear(768, num_classes)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return self.classifier(outputs.pooler_output)
```

**Status:** :material-check-circle: **Fully Compatible** (Recommended for transformers)

**Requirements:**

- `transformers` library

**Best for:**

- Vision Transformers (ViT, DeiT, BEiT, Swin)
- Full control over model architecture
- Custom configurations
- Production use with explicit specifications

**Advantages:**

- Full control and transparency
- No abstraction layer
- Better IDE support and type hints
- Easy to customize and extend
- Explicit configuration

---

### Approach 3: HF Transformers Models (Auto Classes)

**What it is:** Using AutoModel, AutoConfig, and AutoImageProcessor for automatic model detection.

```python
from transformers import AutoModel, AutoConfig, AutoImageProcessor

# Auto classes automatically detect model type
model = AutoModel.from_pretrained("google/vit-base-patch16-224")
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

**Status:** :material-check-circle: **Compatible** (but not necessary)

**Requirements:**

- `transformers` library

**Best for:**

- Quick prototyping
- Dynamic model selection
- Multi-model pipelines
- Rapid experimentation

**Note:** Auto classes work but are **not required**. Approach #2 (Direct classes) provides better control and is recommended for production.

---

## Comparison Matrix

| Feature | HF Hub timm | HF Direct | HF Auto |
|---------|-------------|-----------|---------|
| **PyTorch Lightning** | :material-check-circle: Full | :material-check-circle: Full | :material-check-circle: Full |
| **Training/Val/Test** | :material-check-circle: | :material-check-circle: | :material-check-circle: |
| **Checkpointing** | :material-check-circle: | :material-check-circle: | :material-check-circle: |
| **Distributed (DDP)** | :material-check-circle: | :material-check-circle: | :material-check-circle: |
| **Mixed Precision** | :material-check-circle: | :material-check-circle: | :material-check-circle: |
| **AutoTimm Integration** | :material-check-circle: Native | :material-alert: Manual | :material-alert: Manual |
| **Model Type** | timm models | transformers | transformers |
| **Library Required** | timm, hf_hub | transformers | transformers |
| **Explicit Config** | :material-check-circle: | :material-check-circle: | :material-alert: Auto-detected |
| **Type Safety** | :material-check-circle: | :material-check-circle: | :material-alert: Limited |
| **IDE Support** | :material-check-circle: | :material-check-circle: | :material-alert: Limited |

## Quick Decision Guide

### Use HF Hub timm (Approach 1) when:

- :material-check-circle: Working with CNN architectures (ResNet, EfficientNet, ConvNeXt)
- :material-check-circle: Want seamless AutoTimm integration
- :material-check-circle: Need AutoTimm's task-specific heads
- :material-check-circle: Using timm ecosystem
- :material-check-circle: Building production systems

**Example:**
```python
from autotimm import ImageClassifier, SemanticSegmentor

# Classification
classifier = ImageClassifier(
    backbone="hf-hub:timm/convnext_base.fb_in22k_ft_in1k",
    num_classes=100,
)

# Segmentation
segmentor = SemanticSegmentor(
    backbone="hf-hub:timm/resnet50.a1_in1k",
    num_classes=19,
    head_type="deeplabv3plus",
)
```

### Use HF Direct (Approach 2) when:

- :material-check-circle: Using Vision Transformers (ViT, DeiT, BEiT, Swin)
- :material-check-circle: Need full control over architecture
- :material-check-circle: Want explicit, transparent configuration
- :material-check-circle: Building custom models
- :material-check-circle: Production deployments

**Example:**
```python
from transformers import ViTModel, ViTConfig
import pytorch_lightning as pl

class CustomViTClassifier(pl.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        # Explicit configuration
        config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            image_size=224,
            patch_size=16,
            hidden_dropout_prob=0.1,
        )
        self.vit = ViTModel(config)
        self.classifier = torch.nn.Linear(768, num_classes)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return self.classifier(outputs.pooler_output)
```

### Use HF Auto (Approach 3) when:

- :material-check-circle: Rapid prototyping and experimentation
- :material-check-circle: Trying different model architectures quickly
- :material-check-circle: Model type determined at runtime
- :material-check-circle: Quick proof-of-concept

**Example:**
```python
from transformers import AutoModel

# Quick experimentation
for model_name in ["google/vit-base-patch16-224", "microsoft/beit-base-patch16-224"]:
    model = AutoModel.from_pretrained(model_name)
    # Test and compare...
```

## Complete Usage Example

Here's how all three approaches work together:

```python
import pytorch_lightning as pl
import torch
from transformers import ViTModel, ViTConfig, AutoModel
import autotimm as at  # recommended alias
from autotimm import ImageClassifier, AutoTrainer, ImageDataModule

# Prepare data
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    image_size=224,
    batch_size=32,
)

# === Approach 1: HF Hub timm via AutoTimm ===
autotimm_model = ImageClassifier(
    backbone="hf-hub:timm/resnet50.a1_in1k",
    num_classes=10,
)

# === Approach 2: HF transformers Direct ===
class ViTClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel(ViTConfig(hidden_size=768))
        self.classifier = torch.nn.Linear(768, 10)
        self.save_hyperparameters()

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return self.classifier(outputs.pooler_output)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

vit_model = ViTClassifier()

# === Approach 3: HF Auto (optional) ===
class AutoViTWrapper(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("google/vit-base-patch16-224")
        self.classifier = torch.nn.Linear(768, 10)

    # ... similar implementation

# All three work with PyTorch Lightning!
trainer = AutoTrainer(max_epochs=10)

trainer.fit(autotimm_model, datamodule=data)  # :material-check-circle: Works
trainer.fit(vit_model, datamodule=data)       # :material-check-circle: Works
# trainer.fit(auto_vit, datamodule=data)       # :material-check-circle: Works
```

## Key Insights

### 1. AutoTimm Uses timm, Not transformers

AutoTimm is built on the `timm` library:

```python
# AutoTimm dependencies
:material-check-circle: timm (for models)
:material-check-circle: huggingface_hub (for downloading from HF Hub)
:material-close-circle: transformers (different library, optional)
```

You can use **both** in the same project:
- Use AutoTimm for timm backbones (CNNs, timm ViTs)
- Use transformers for HF-specific models (ViT, DeiT, etc.)
- Both work seamlessly with PyTorch Lightning!

### 2. No Auto Classes Required

For production use with transformers:

```python
# :material-close-circle: Not necessary
from transformers import AutoModel, AutoImageProcessor, AutoConfig

# :material-check-circle: Recommended: Use specific classes
from transformers import ViTModel, ViTConfig, ViTImageProcessor
```

Benefits:
- Better type hints and IDE support
- More explicit and controlled
- Easier to debug and maintain

### 3. All Approaches Are Fully Compatible

Every approach supports **all** PyTorch Lightning features:

- :material-check-circle: Training, validation, testing
- :material-check-circle: Checkpointing and resume
- :material-check-circle: Distributed training (DDP, FSDP)
- :material-check-circle: Mixed precision (FP16, BF16)
- :material-check-circle: All callbacks (EarlyStopping, ModelCheckpoint, etc.)
- :material-check-circle: All loggers (TensorBoard, MLflow, W&B)
- :material-check-circle: Gradient accumulation
- :material-check-circle: Multi-GPU training

## Performance Comparison

All three approaches have **identical runtime performance**:

| Metric | HF Hub timm | HF Direct | HF Auto |
|--------|-------------|-----------|---------|
| **Training Speed** | :material-lightning-bolt: Fast | :material-lightning-bolt: Fast | :material-lightning-bolt: Fast |
| **Inference Speed** | :material-lightning-bolt: Fast | :material-lightning-bolt: Fast | :material-lightning-bolt: Fast |
| **Memory Usage** | :material-database: Model-dependent | :material-database: Model-dependent | :material-database: Model-dependent |
| **Load Time** | First: Slow (download)<br>After: Fast (cached) | First: Slow (download)<br>After: Fast (cached) | First: Slow (download)<br>After: Fast (cached) |
| **Abstraction Overhead** | None | None | Minimal |

**Conclusion:** Choose based on your use case and requirements, not performance.

## Recommendations

### For Production Systems

1. **CNNs and timm models**: Use **Approach 1** (HF Hub timm via AutoTimm)
   - Best integration with AutoTimm
   - Proven architectures
   - Simple, clean API

2. **Vision Transformers**: Use **Approach 2** (HF Direct classes)
   - Full control and transparency
   - Explicit configuration
   - Better maintainability

### For Research and Prototyping

- Quick experiments: **Approach 3** (Auto classes)
- Serious development: Switch to **Approach 1 or 2**

## Resources

### Documentation
- [HF Hub Integration Guide](huggingface-hub-integration.md)
- [HF Transformers Integration Guide](huggingface-transformers-integration.md)
- [HF Hub Examples](../../examples/integration/huggingface-hub.md)

### Test Suites
- HF Hub Lightning compatibility: `tests/test_hf_hub_lightning_compatibility.py`
- HF Direct models: `tests/test_hf_direct_models.py`
- HF Hub backbones: `tests/test_hf_hub_backbones.py`

### Examples
- [HF Hub Classification](https://github.com/Noctillio-Ai/AutoTimm/blob/main/examples/huggingface/hf_hub_classification.py)
- [HF Hub Segmentation](https://github.com/Noctillio-Ai/AutoTimm/blob/main/examples/huggingface/hf_hub_segmentation.py)
- [HF Hub Object Detection](https://github.com/Noctillio-Ai/AutoTimm/blob/main/examples/huggingface/hf_hub_object_detection.py)
- [HF Direct with Lightning](https://github.com/Noctillio-Ai/AutoTimm/blob/main/examples/huggingface/hf_direct_models_lightning.py)

## Summary

:material-check-circle: **Three approaches, all fully compatible with PyTorch Lightning**

:material-check-circle: **Choose based on use case:**

- CNNs → HF Hub timm via AutoTimm
- Vision Transformers → HF Direct classes
- Quick prototyping → HF Auto classes

:material-check-circle: **No Auto classes required** for production use

:material-check-circle: **AutoTimm and transformers work together** in the same project

:material-check-circle: **All Lightning features supported** across all approaches

## AutoTrainer Compatibility

All HuggingFace integration approaches are **fully compatible** with AutoTimm's AutoTrainer class. AutoTrainer extends PyTorch Lightning's Trainer with additional features, and all of them work seamlessly with HF models.

### Verified AutoTrainer Features

All the following AutoTrainer-specific features have been tested and verified:

- :material-check-circle: Basic training with AutoTrainer
- :material-check-circle: Checkpoint monitoring and saving
- :material-check-circle: Early stopping callbacks
- :material-check-circle: Multiple callbacks simultaneously
- :material-check-circle: Validation and testing
- :material-check-circle: Gradient accumulation
- :material-check-circle: Mixed precision training (FP16/BF16)
- :material-check-circle: ImageDataModule integration
- :material-check-circle: LoggerManager support
- :material-check-circle: TunerConfig for auto LR/batch size finding

### AutoTrainer with HF Hub timm

```python
from autotimm import AutoTrainer, ImageClassifier, ImageDataModule, MetricConfig
import pytorch_lightning as pl

# Model with HF Hub backbone
model = ImageClassifier(
    backbone="hf-hub:timm/resnet50.a1_in1k",
    num_classes=10,
    metrics=[
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass", "num_classes": 10},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
    ],
    lr=1e-3,
)

# DataModule
datamodule = ImageDataModule(
    data_dir="./data",
    image_size=224,
    batch_size=32,
)

# AutoTrainer with callbacks
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath="./checkpoints",
    filename="best-{epoch:02d}-{val/accuracy:.2f}",
    monitor="val/accuracy",
    mode="max",
    save_top_k=3,
)

early_stop = pl.callbacks.EarlyStopping(
    monitor="val/accuracy",
    patience=10,
    mode="max",
)

trainer = AutoTrainer(
    max_epochs=100,
    accelerator="auto",
    precision="16-mixed",
    accumulate_grad_batches=2,
    callbacks=[checkpoint_callback, early_stop],
)

# Train
trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)
```

### AutoTrainer with HF Transformers Direct

```python
from autotimm import AutoTrainer
import pytorch_lightning as pl
from transformers import ViTModel, ViTConfig
import torch.nn.functional as F

class ViTClassifier(pl.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.save_hyperparameters()

        config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            image_size=224,
            patch_size=16,
        )

        self.vit = ViTModel(config)
        self.classifier = torch.nn.Linear(768, num_classes)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        return self.classifier(outputs.pooler_output)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

# Use with AutoTrainer
model = ViTClassifier(num_classes=10)
trainer = AutoTrainer(
    max_epochs=100,
    callbacks=[checkpoint_callback, early_stop],
)
trainer.fit(model, train_loader, val_loader)
```

### AutoTrainer Advanced Features

#### Automatic Hyperparameter Tuning

```python
from autotimm import AutoTrainer, TunerConfig

trainer = AutoTrainer(
    max_epochs=10,
    tuner_config=TunerConfig(
        auto_lr=True,  # Automatic learning rate finding
        auto_batch_size=True,  # Automatic batch size scaling
    ),
)

# AutoTrainer will automatically run LR finder and batch size scaling
trainer.fit(model, datamodule=datamodule)
```

#### Multiple Loggers

```python
from autotimm import AutoTrainer, LoggerConfig

trainer = AutoTrainer(
    max_epochs=10,
    logger=[
        LoggerConfig(backend="tensorboard", params={"save_dir": "./logs"}),
        LoggerConfig(backend="wandb", params={"project": "my-project"}),
    ],
)

trainer.fit(model, datamodule=datamodule)
```

### AutoTrainer Compatibility Matrix

| Feature | HF Hub timm | HF Direct | HF Auto |
|---------|-------------|-----------|---------|
| **Basic Training** | :material-check-circle: | :material-check-circle: | :material-check-circle: |
| **Checkpointing** | :material-check-circle: | :material-check-circle: | :material-check-circle: |
| **Early Stopping** | :material-check-circle: | :material-check-circle: | :material-check-circle: |
| **Multiple Callbacks** | :material-check-circle: | :material-check-circle: | :material-check-circle: |
| **Gradient Accumulation** | :material-check-circle: | :material-check-circle: | :material-check-circle: |
| **Mixed Precision** | :material-check-circle: | :material-check-circle: | :material-check-circle: |
| **Validation/Testing** | :material-check-circle: | :material-check-circle: | :material-check-circle: |
| **ImageDataModule** | :material-check-circle: | :material-check-circle: | :material-check-circle: |
| **LoggerManager** | :material-check-circle: | :material-check-circle: | :material-check-circle: |
| **TunerConfig** | :material-check-circle: | :material-check-circle: | :material-check-circle: |

### Production Workflow

Complete production-ready workflow with AutoTrainer:

```python
import pytorch_lightning as pl
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    MetricConfig,
    TunerConfig,
)

# Metrics
metrics = [
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass", "num_classes": 100},
        stages=["train", "val", "test"],
        prog_bar=True,
    ),
    MetricConfig(
        name="top5_acc",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass", "num_classes": 100, "top_k": 5},
        stages=["val", "test"],
    ),
]

# Model with HF Hub backbone
model = ImageClassifier(
    backbone="hf-hub:timm/convnext_base.fb_in22k_ft_in1k",
    num_classes=100,
    metrics=metrics,
    lr=1e-3,
    optimizer="adamw",
    scheduler="cosine",
)

# DataModule
datamodule = ImageDataModule(
    data_dir="./data",
    image_size=224,
    batch_size=32,
)

# Callbacks
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="val/accuracy",
    mode="max",
    save_top_k=3,
)

early_stop = pl.callbacks.EarlyStopping(
    monitor="val/accuracy",
    patience=10,
    mode="max",
)

# AutoTrainer with all features
trainer = AutoTrainer(
    max_epochs=100,
    accelerator="auto",
    precision="16-mixed",
    callbacks=[checkpoint, early_stop],
    tuner_config=TunerConfig(auto_lr=True),
)

# Train and test
trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)
```

**Result:** :material-check-circle: Everything works perfectly with full AutoTrainer support!

## Summary & Recommendations

### :material-check-circle: All Approaches Fully Compatible

- **PyTorch Lightning**: All features work (training, validation, testing, checkpointing, DDP, AMP, callbacks)
- **AutoTrainer**: All features work (checkpoint monitoring, early stopping, gradient accumulation, mixed precision, auto LR/batch size, multiple loggers)
- **Production Ready**: All approaches are stable and tested

### :material-chart-bar: Test Coverage

| Component | Tests | Status |
|-----------|-------|--------|
| HF Hub Lightning | 12 tests | :material-check-circle: 100% pass |
| HF Direct Models | 10 tests | :material-check-circle: 100% pass |
| HF Hub Backbones | 13 tests | :material-check-circle: 100% pass |
| HF AutoTrainer | 12 tests | :material-check-circle: 100% pass |
| **Total** | **47 tests** | **:material-check-circle: 100% pass** |

### :material-bullseye: Quick Decision Guide

**Choose HF Hub timm when:**

- Working with CNNs (ResNet, EfficientNet, ConvNeXt)
- Want seamless AutoTimm integration
- Need all AutoTimm tasks (classification, detection, segmentation)
- Building production systems

**Choose HF Direct when:**

- Using Vision Transformers (ViT, DeiT, BEiT, Swin)
- Need full control over architecture
- Want explicit configuration
- Building custom models

**Choose HF Auto when:**

- Rapid prototyping
- Quick experimentation
- Testing multiple architectures

### :material-lightbulb: Key Insights

1. **No Auto classes required** - Use specific classes (`ViTModel`, `DeiTModel`) for better control
2. **AutoTimm uses timm, not transformers** - Both libraries can be used together
3. **AutoTrainer adds value** - Extended features beyond pl.Trainer
4. **Performance is identical** - Choose based on use case, not speed
5. **All features work** - No workarounds or special handling needed
