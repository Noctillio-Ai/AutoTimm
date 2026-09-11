# Training Callbacks

PyTorch Lightning callbacks for automatic interpretation and feature monitoring during training.

## Overview

AutoTimm provides two callbacks for integrating interpretation into your training workflow:

- **`InterpretationCallback`**: Automatically generate and log explanations during training
- **`FeatureMonitorCallback`**: Track feature statistics across layers

Both callbacks integrate seamlessly with PyTorch Lightning loggers (TensorBoard, Weights & Biases, MLflow).

---

## InterpretationCallback

Automatically generate explanations for sample images during training and log them to your tracking platform.

### Class: `InterpretationCallback`

```python
import autotimm as at  # recommended alias
from autotimm.interpretation import InterpretationCallback

callback = InterpretationCallback(
    sample_images: Union[torch.Tensor, List[torch.Tensor], List[str]],
    sample_labels: Optional[List[int]] = None,
    method: Literal['gradcam', 'gradcam++', 'integrated_gradients'] = 'gradcam',
    target_layer: Optional[Union[str, torch.nn.Module]] = None,
    log_every_n_epochs: int = 5,
    log_every_n_steps: Optional[int] = None,
    num_samples: int = 8,
    colormap: str = "viridis",
    alpha: float = 0.4,
    prefix: str = "interpretation"
)
```

**Parameters:**

- `sample_images`: Images to explain during training. Can be torch tensors, list of tensors, or list of file paths. Will be sampled down to `num_samples` if more provided
- `sample_labels` (Optional[List[int]]): Ground truth labels for sample images
- `method` (str): Interpretation method to use. Options: `'gradcam'` (default, fast), `'gradcam++'` (better for multiple objects), `'integrated_gradients'` (pixel-level attributions)
- `target_layer` (Optional): Layer to use for interpretation (None = auto-detect)
- `log_every_n_epochs` (int): Generate explanations every N epochs (default: 5)
- `log_every_n_steps` (Optional[int]): Alternative: log every N steps (overrides epochs)
- `num_samples` (int): Number of images to explain (default: 8)
- `colormap` (str): Matplotlib colormap for heatmaps (default: "viridis")
- `alpha` (float): Overlay transparency (0-1, default: 0.4)
- `prefix` (str): Prefix for logged images (default: "interpretation")

### Basic Usage

```python
from autotimm import AutoTrainer, ImageClassifier, LoggerConfig, ImageDataModule, LoggerConfig
from autotimm.interpretation import InterpretationCallback
from PIL import Image

# Prepare sample images for monitoring
sample_images = [Image.open(f"samples/img_{i}.jpg") for i in range(8)]

# Create callback
interp_callback = InterpretationCallback(
    sample_images=sample_images,
    method="gradcam",
    log_every_n_epochs=5,
    num_samples=8,
)

# Create model and trainer
model = ImageClassifier(backbone="resnet50", num_classes=10)
data = ImageDataModule(data_dir="./data")  # expects data/train and data/val

# Train with automatic interpretation
trainer = AutoTrainer(
    max_epochs=100,
    callbacks=[interp_callback],
    logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
)

trainer.fit(model, datamodule=data)
```

### With TensorBoard

```python
from autotimm.core.loggers import LoggerConfig

# Configure TensorBoard logger
logger_config = LoggerConfig(
    backend="tensorboard",
    save_dir="logs/",
    name="my_experiment",
)

# Create callback
interp_callback = InterpretationCallback(
    sample_images=sample_images,
    method="gradcam",
    log_every_n_epochs=5,
)

# Train
trainer = AutoTrainer(
    max_epochs=100,
    callbacks=[interp_callback],
    logger=logger_config.create_logger(),
)
trainer.fit(model, datamodule=data)
```

**View in TensorBoard:**
```bash
tensorboard --logdir logs/
```

Navigate to the "IMAGES" tab to see interpretations logged during training.

### With Weights & Biases

```python
import wandb
from pytorch_lightning.loggers import WandbLogger

# Initialize W&B
wandb.init(project="my_project", name="experiment_1")

# Create callback
interp_callback = InterpretationCallback(
    sample_images=sample_images,
    method="gradcam++",
    log_every_n_epochs=5,
)

# Train
trainer = AutoTrainer(
    max_epochs=100,
    callbacks=[interp_callback],
    logger=WandbLogger(),
)
trainer.fit(model, datamodule=data)
```

### With MLflow

```python
from pytorch_lightning.loggers import MLFlowLogger

# Create callback
interp_callback = InterpretationCallback(
    sample_images=sample_images,
    method="gradcam",
    log_every_n_epochs=10,
)

# Train
trainer = AutoTrainer(
    max_epochs=100,
    callbacks=[interp_callback],
    logger=MLFlowLogger(experiment_name="my_experiment"),
)
trainer.fit(model, datamodule=data)
```

### Step-Based Logging

Log explanations based on training steps instead of epochs:

```python
interp_callback = InterpretationCallback(
    sample_images=sample_images,
    method="gradcam",
    log_every_n_steps=1000,  # Log every 1000 steps
)
```

### Multiple Methods

Compare different interpretation methods during training:

```python
callbacks = [
    InterpretationCallback(
        sample_images=sample_images,
        method="gradcam",
        log_every_n_epochs=5,
        prefix="interp_gradcam",
    ),
    InterpretationCallback(
        sample_images=sample_images,
        method="gradcam++",
        log_every_n_epochs=5,
        prefix="interp_gradcampp",
    ),
]

trainer = AutoTrainer(max_epochs=100, callbacks=callbacks)
```

### Custom Visualization

Customize colormap and overlay settings:

```python
interp_callback = InterpretationCallback(
    sample_images=sample_images,
    method="gradcam",
    colormap="hot",      # Use 'hot' colormap
    alpha=0.5,           # More transparent overlay
    log_every_n_epochs=5,
)
```

---

## FeatureMonitorCallback

Monitor feature statistics during training to understand how features evolve.

### Class: `FeatureMonitorCallback`

```python
from autotimm.interpretation import FeatureMonitorCallback

callback = FeatureMonitorCallback(
    layer_names: List[str],
    log_every_n_epochs: int = 1,
    num_batches: int = 10
)
```

**Parameters:**

- `layer_names` (List[str]): Names of layers to monitor (e.g., ["backbone.layer2", "backbone.layer4"])
- `log_every_n_epochs` (int): Log statistics every N epochs (default: 1)
- `num_batches` (int): Number of batches to accumulate statistics over (default: 10)

### Basic Usage

```python
from autotimm import AutoTrainer, ImageClassifier, LoggerConfig
from autotimm.interpretation import FeatureMonitorCallback

# Create callback
feature_callback = FeatureMonitorCallback(
    layer_names=["backbone.layer2", "backbone.layer3", "backbone.layer4"],
    log_every_n_epochs=1,
    num_batches=10,
)

# Create model
model = ImageClassifier(backbone="resnet50", num_classes=10)

# Train with feature monitoring
trainer = AutoTrainer(
    max_epochs=100,
    callbacks=[feature_callback],
    logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
)

trainer.fit(model, datamodule=data)
```

### Logged Metrics

The callback logs the following metrics for each monitored layer:

- `features/{layer_name}/mean`: Mean activation
- `features/{layer_name}/std`: Standard deviation
- `features/{layer_name}/sparsity`: Fraction of zero activations
- `features/{layer_name}/max`: Maximum activation

**Example in TensorBoard:**
```
features/backbone.layer2/mean: 0.234
features/backbone.layer2/std: 0.156
features/backbone.layer2/sparsity: 0.345
features/backbone.layer2/max: 2.456
```

### Monitoring Specific Layers

Monitor only the final layers for efficiency:

```python
feature_callback = FeatureMonitorCallback(
    layer_names=["backbone.layer4"],  # Only final layer
    log_every_n_epochs=1,
)
```

### Fine-Grained Monitoring

Monitor every layer for detailed analysis:

```python
# Get all conv layer names
layer_names = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        layer_names.append(name)

feature_callback = FeatureMonitorCallback(
    layer_names=layer_names,
    log_every_n_epochs=1,
)
```

### Adjusting Batch Count

Control how many batches to use for statistics:

```python
feature_callback = FeatureMonitorCallback(
    layer_names=["backbone.layer2", "backbone.layer3", "backbone.layer4"],
    log_every_n_epochs=1,
    num_batches=20,  # More batches = more accurate but slower
)
```

---

## Combined Usage

Use both callbacks together for comprehensive monitoring:

```python
from autotimm import AutoTrainer, ImageClassifier, LoggerConfig
from autotimm.interpretation import InterpretationCallback, FeatureMonitorCallback

# Sample images for interpretation
sample_images = [Image.open(f"samples/img_{i}.jpg") for i in range(8)]

# Interpretation callback
interp_callback = InterpretationCallback(
    sample_images=sample_images,
    method="gradcam",
    log_every_n_epochs=5,
    num_samples=8,
)

# Feature monitoring callback
feature_callback = FeatureMonitorCallback(
    layer_names=["backbone.layer2", "backbone.layer3", "backbone.layer4"],
    log_every_n_epochs=1,
    num_batches=10,
)

# Create model and trainer
model = ImageClassifier(backbone="resnet50", num_classes=10)

trainer = AutoTrainer(
    max_epochs=100,
    callbacks=[interp_callback, feature_callback],
    logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
)

trainer.fit(model, datamodule=data)
```

---

## Use Cases

### 1. Debugging Model Learning

Check if model is learning meaningful patterns:

```python
# Monitor early in training
interp_callback = InterpretationCallback(
    sample_images=sample_images,
    method="gradcam",
    log_every_n_epochs=1,  # Log every epoch
)
```

**What to look for:**
- Epoch 1-10: Random/noisy heatmaps
- Epoch 10-30: Heatmaps start focusing on relevant regions
- Epoch 30+: Clear, focused heatmaps on discriminative features

### 2. Detecting Overfitting

Monitor feature sparsity to detect overfitting:

```python
feature_callback = FeatureMonitorCallback(
    layer_names=["backbone.layer4"],
    log_every_n_epochs=1,
)
```

**Warning signs:**
- Rapidly increasing sparsity (>70%): Model becoming too specialized
- Decreasing mean activation: Features dying out
- High variance in statistics: Unstable training

### 3. Comparing Architectures

Compare how different models learn:

```python
models_to_test = ["resnet18", "resnet50", "efficientnet_b0"]

for backbone in models_to_test:
    model = ImageClassifier(backbone=backbone, num_classes=10)

    feature_callback = FeatureMonitorCallback(
        layer_names=["backbone.layer4"],
        log_every_n_epochs=1,
    )

    trainer = AutoTrainer(
        max_epochs=50,
        callbacks=[feature_callback],
        logger=WandbLogger(project="architecture_comparison", name=backbone),
    )

    trainer.fit(model, datamodule=data)
```

### 4. Transfer Learning Analysis

Monitor how features adapt during fine-tuning:

```python
# Start with pretrained model
model = ImageClassifier(backbone="resnet50", num_classes=10, pretrained=True)

# Monitor features during fine-tuning
callbacks = [
    InterpretationCallback(sample_images, log_every_n_epochs=1),
    FeatureMonitorCallback(["backbone.layer4"], log_every_n_epochs=1),
]

trainer = AutoTrainer(max_epochs=30, callbacks=callbacks)
trainer.fit(model, datamodule=data)
```

**What to look for:**
- Early epochs: Gradual shift in heatmap focus
- Feature statistics: Small changes (good), large changes (may need lower LR)

### 5. Curriculum Learning

Adjust sample images as training progresses:

```python
class DynamicInterpretationCallback(InterpretationCallback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Update sample images based on current performance
        if trainer.current_epoch % 10 == 0:
            # Load harder examples
            self.sample_images = load_challenging_samples()
        super().on_train_epoch_start(trainer, pl_module)
```

---

## Performance Considerations

### Callback Overhead

**InterpretationCallback:**
- Forward pass per image: ~10-50ms (depending on model size)
- Total overhead per log: `num_samples * forward_time + visualization_time`
- Recommendation: Log every 5-10 epochs for balance

**FeatureMonitorCallback:**
- Hook overhead: Minimal (<1% of training time)
- Accumulation: Proportional to `num_batches`
- Recommendation: Monitor 3-5 key layers, use 10-20 batches

### Optimizing Logging Frequency

```python
# For quick experiments (low overhead)
interp_callback = InterpretationCallback(
    sample_images=sample_images[:4],  # Fewer samples
    log_every_n_epochs=10,            # Less frequent
)

# For detailed analysis (higher overhead)
interp_callback = InterpretationCallback(
    sample_images=sample_images,
    log_every_n_epochs=1,
)
```

### Memory Management

```python
# Reduce memory usage
feature_callback = FeatureMonitorCallback(
    layer_names=["backbone.layer4"],  # Monitor fewer layers
    num_batches=5,                     # Use fewer batches
)
```

---

## Troubleshooting

For interpretation callback issues, see the [Troubleshooting - Interpretation](../../troubleshooting/task-specific/interpretation.md) including:

- Interpretations not logged
- Feature monitoring not working
- High memory usage
- Slow training with callbacks
    log_every_n_epochs=10,  # Less frequent
)

# Monitor fewer layers
feature_callback = FeatureMonitorCallback(
    layer_names=["backbone.layer4"],  # Only final layer
)
```

---

## Advanced Customization

### Custom Interpretation Method

```python
from autotimm.interpretation import InterpretationCallback
from autotimm.interpretation import IntegratedGradients

class CustomInterpretationCallback(InterpretationCallback):
    def on_train_start(self, trainer, pl_module):
        # Use custom method
        self.explainer = IntegratedGradients(
            pl_module,
            baseline='blur',
            steps=30,
        )
```

### Custom Logging Logic

```python
class CustomFeatureCallback(FeatureMonitorCallback):
    def _compute_and_log_statistics(self, trainer):
        super()._compute_and_log_statistics(trainer)

        # Add custom metrics
        for name, acts in self.activations.items():
            if len(acts) > 0:
                all_acts = torch.cat(acts, dim=0)
                # Custom metric: L1 norm
                l1_norm = all_acts.abs().mean().item()
                trainer.logger.log_metrics(
                    {f"features/{name}/l1_norm": l1_norm},
                    step=trainer.global_step
                )
```

---

## See Also

- [Feature Visualization](feature-visualization.md) - Analyze features after training
- [Interpretation Methods](methods.md) - Available interpretation methods
- [Main Guide](index.md) - Overview and quick start
