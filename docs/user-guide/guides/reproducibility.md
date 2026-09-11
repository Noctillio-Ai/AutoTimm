# Reproducibility Guide

AutoTimm is designed with reproducibility as a first-class feature. This guide covers everything you need to know about achieving consistent, reproducible results in your experiments.

## Reproducibility Workflow

### Seed Sources

```mermaid
graph LR
    A{Seed Sources} --> B[Model<br/>seed=42 opt-in]
    A --> C[Trainer<br/>seed=42 opt-in]
    A --> D[Manual<br/>seed_everything]
    B --> E[seed_everything<br/>Python, NumPy, PyTorch, CUDA]
    C --> E
    D --> E

    style A fill:#FF9800,stroke:#F57C00
    style B fill:#1976D2,stroke:#1565C0
    style C fill:#1976D2,stroke:#1565C0
    style D fill:#1976D2,stroke:#1565C0
    style E fill:#1565C0,stroke:#0D47A1
```

### Deterministic Pipeline

```mermaid
graph LR
    A[Seed Everything] --> B[Deterministic Mode<br/>cuDNN + Benchmark + Workers]
    B --> C[Reproducible Results<br/>Same init, data order,<br/>augmentations, gradients]
    C --> D{Verify}
    D --> E1[Same seed = same results]
    D --> E2[Different seed = different results]

    style A fill:#1565C0,stroke:#0D47A1
    style B fill:#1976D2,stroke:#1565C0
    style C fill:#1976D2,stroke:#1565C0
    style D fill:#FF9800,stroke:#F57C00
    style E1 fill:#4CAF50,stroke:#388E3C
    style E2 fill:#4CAF50,stroke:#388E3C
```

## Why Reproducibility Matters

- **Research**: Essential for publishing papers and validating results
- **Debugging**: Makes it easier to isolate and fix issues
- **Collaboration**: Team members get identical results
- **Production**: Ensures consistent model behavior

## Quick Start

AutoTimm supports opt-in reproducibility by setting `seed` explicitly:

```python
import autotimm as at  # recommended alias
from autotimm import ImageClassifier, AutoTrainer

# Opt-in to reproducibility with seed=42
model = ImageClassifier(backbone="resnet50", num_classes=10, seed=42)
trainer = AutoTrainer(max_epochs=10, seed=42)

# Results will be identical across runs!
```

## Default Behavior

By default, seeding is **disabled** (`seed=None`). Set a seed explicitly for reproducibility:

- **Default seed**: `None` (no seeding)
- **Deterministic mode**: `True` for models, `False` for trainer
- **Components seeded** (when seed is set): Python, NumPy, PyTorch, cuDNN

```python
from autotimm import ImageClassifier

# Explicit opt-in to reproducibility
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    seed=42,              # set explicitly
    deterministic=True,   # default for models
)
```

## Seeding Levels

AutoTimm provides seeding at multiple levels:

### 1. Model-Level Seeding

Seeds when the model is created:

```python
from autotimm import ImageClassifier

model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    seed=42,
    deterministic=True,
)
```

### 2. Trainer-Level Seeding

Seeds before training starts:

```python
from autotimm import AutoTrainer

trainer = AutoTrainer(
    max_epochs=10,
    seed=42,
    deterministic=True,
)
```

### 3. Manual Seeding

Complete control over seeding:

```python
from autotimm import seed_everything

# Seed everything manually
seed_everything(42, deterministic=True)

# Now create models
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    seed=None,  # Don't seed again
)
```

## Custom Seeds

Use custom seeds for different experiments:

```python
# Experiment 1: seed=42
model1 = ImageClassifier(backbone="resnet50", num_classes=10, seed=42)
trainer1 = AutoTrainer(max_epochs=10, seed=42)

# Experiment 2: seed=123
model2 = ImageClassifier(backbone="resnet50", num_classes=10, seed=123)
trainer2 = AutoTrainer(max_epochs=10, seed=123)

# Different seeds = different random initializations
```

## Deterministic Mode

Control the trade-off between speed and reproducibility:

### Full Reproducibility (Default)

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    seed=42,
    deterministic=True,  # Full reproducibility
)

trainer = AutoTrainer(
    max_epochs=10,
    seed=42,
    deterministic=True,
)
```

**What it does:**

- Sets `torch.backends.cudnn.deterministic = True`
- Sets `torch.backends.cudnn.benchmark = False`
- Uses `torch.use_deterministic_algorithms(True)` (PyTorch 1.8+)

**Pros:**

- :material-check-circle: Fully reproducible results
- :material-check-circle: Identical outputs across runs
- :material-check-circle: Perfect for research

**Cons:**

- :material-close-circle: Slower training
- :material-close-circle: May impact performance by 10-30%

### Faster Training

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    seed=42,
    deterministic=False,  # Faster training
)

trainer = AutoTrainer(
    max_epochs=10,
    seed=42,
    deterministic=False,
)
```

**What it does:**

- Enables cuDNN benchmark mode
- Allows non-deterministic algorithms
- Still seeds random number generators

**Pros:**

- :material-check-circle: Faster training
- :material-check-circle: Better GPU utilization
- :material-check-circle: Partially reproducible

**Cons:**

- :material-close-circle: Results may vary slightly between runs
- :material-close-circle: Small differences in final metrics

## Trainer Seeding Options

AutoTrainer supports two seeding approaches:

### PyTorch Lightning Seeding (Default)

```python
trainer = AutoTrainer(
    max_epochs=10,
    seed=42,
    use_autotimm_seeding=False,  # Default
)
```

Uses Lightning's built-in `seed_everything()`:
- Standard Lightning behavior
- Includes dataloader worker seeding
- Good integration with Lightning ecosystem

### AutoTimm Custom Seeding

```python
trainer = AutoTrainer(
    max_epochs=10,
    seed=42,
    use_autotimm_seeding=True,
)
```

Uses AutoTimm's custom `seed_everything()`:
- More comprehensive seeding
- Explicit control over deterministic mode
- Sets additional environment variables

**Both options work well!** Choose based on your preference.

## Complete Reproducible Workflow

```python
from autotimm import (
    ImageClassifier,
    AutoTrainer,
    ImageDataModule,
    MetricConfig,
    seed_everything,
)

# 1. Set global seed (optional but recommended)
SEED = 42
seed_everything(SEED, deterministic=True)

# 2. Create reproducible data module
data = ImageDataModule(
    data_dir="./data",
    batch_size=32,
    num_workers=4,
)

# 3. Create reproducible model
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    seed=SEED,
    deterministic=True,
    metrics=[
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass", "num_classes": 10},
            stages=["val"],
        )
    ],
)

# 4. Create reproducible trainer
trainer = AutoTrainer(
    max_epochs=10,
    seed=SEED,
    deterministic=True,
)

# 5. Train - results will be identical across runs!
trainer.fit(model, datamodule=data)
```

## Disabling Seeding

Sometimes you want to explore model variance:

```python
# Disable seeding completely
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    seed=None,            # No seeding
    deterministic=False,  # Avoid warning
)

trainer = AutoTrainer(
    max_epochs=10,
    seed=None,            # No seeding
    deterministic=False,  # Avoid warning
)

# Results will vary between runs
```

!!! warning "seed=None with deterministic=True"

    Setting `seed=None` with `deterministic=True` (the default) emits a warning because
    deterministic mode requires seeding to be effective. Either set a seed or set
    `deterministic=False` when disabling seeding.

## Reproducible Inference

Ensure consistent predictions:

```python
import torch
from autotimm import ImageClassifier

# Create model with seeding
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    seed=42,
    compile_model=False,  # Disable for consistency
)
model.eval()

# Load same model again
model2 = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    seed=42,
    compile_model=False,
)
model2.eval()

# Same input
x = torch.randn(1, 3, 224, 224)

# Same predictions
with torch.inference_mode():
    pred1 = model(x)
    pred2 = model2(x)

assert torch.allclose(pred1, pred2)  # True!
```

## Research Paper Setup

For maximum reproducibility in research:

```python
from autotimm import seed_everything, ImageClassifier, AutoTrainer

# Strict reproducibility settings
SEED = 42

# 1. Global seeding with deterministic mode
seed_everything(SEED, deterministic=True)

# 2. Model with strict settings
model = ImageClassifier(
    backbone="resnet50",
    num_classes=1000,
    seed=SEED,
    deterministic=True,
    compile_model=False,  # Disable for consistency
    mixup_alpha=0.0,      # Disable stochastic augmentations
)

# 3. Trainer with strict settings
trainer = AutoTrainer(
    max_epochs=100,
    seed=SEED,
    deterministic=True,
    precision=32,  # Full precision for reproducibility
)
```

**Additional tips for research:**

- Document PyTorch/CUDA versions
- Fix library versions in `requirements.txt`
- Disable stochastic augmentations if needed
- Use full precision (32-bit) instead of mixed precision
- Test reproducibility on multiple runs

## What Gets Seeded

AutoTimm's `seed_everything()` seeds:

### Random Number Generators
- Python's built-in `random` module
- NumPy's `np.random`
- PyTorch's `torch.random`
- PyTorch CUDA random: `torch.cuda.manual_seed_all()`

### Environment Variables
- `PYTHONHASHSEED` - Python hash randomization

### Backend Settings (when `deterministic=True`)
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `torch.use_deterministic_algorithms(True)` (PyTorch 1.8+)

## Verification

Test that seeding works:

```python
import torch
from autotimm import seed_everything

# Test 1: Same seed = same random numbers
seed_everything(42)
x1 = torch.randn(5)

seed_everything(42)
x2 = torch.randn(5)

assert torch.allclose(x1, x2)  # True!

# Test 2: Different seeds = different random numbers
seed_everything(42)
x3 = torch.randn(5)

seed_everything(123)
x4 = torch.randn(5)

assert not torch.allclose(x3, x4)  # True!
```

## Common Issues

Common reproducibility issues include:

- Results still vary slightly
- Training is too slow with deterministic mode
- Different results on different GPUs

## Performance Impact

Expected training time impact:

| Setting | Speed | Reproducibility |
|---------|-------|-----------------|
| `deterministic=True` | Baseline | 100% |
| `deterministic=False` | +10-30% faster | ~95% |
| `seed=None` | +10-30% faster | Variable |

## Best Practices

### For Research
- :material-check-circle: Use `seed=42, deterministic=True`
- :material-check-circle: Document all versions
- :material-check-circle: Test on multiple seeds
- :material-check-circle: Report mean ± std across seeds

### For Production
- :material-check-circle: Use `seed=42, deterministic=False`
- :material-check-circle: Prioritize speed
- :material-check-circle: Test on multiple seeds to ensure robustness

### For Debugging
- :material-check-circle: Use `seed=42, deterministic=True`
- :material-check-circle: Minimal training runs
- :material-check-circle: Compare outputs step-by-step

### For Exploration
- :material-check-circle: Use `seed=None` to explore variance
- :material-check-circle: Try multiple random initializations
- :material-check-circle: Analyze result distribution

## Examples

See complete working examples in the repository:

- `examples/utilities/reproducibility.py` - Comprehensive examples (9 patterns)
- `examples/utilities/test_seeding.py` - Seeding verification tests

## API Reference

- [`seed_everything()`](../../api/utils.md) - Manual seeding function
- [`ImageClassifier`](../models/image-classifier.md) - Model seeding parameters
- [`AutoTrainer`](../../api/trainer.md) - Trainer seeding parameters

## Summary

**Key Takeaways:**

1. AutoTimm supports opt-in reproducibility (set `seed=42` explicitly)
2. Use `deterministic=True` for full reproducibility
3. Use `deterministic=False` for faster training
4. Both model and trainer support seeding
5. Choose seeding approach based on your use case

**Quick Reference:**



```python
# Research (strict reproducibility)
model = ImageClassifier(..., seed=42, deterministic=True)
trainer = AutoTrainer(..., seed=42, deterministic=True)

# Production (speed + partial reproducibility)
model = ImageClassifier(..., seed=42, deterministic=False)
trainer = AutoTrainer(..., seed=42, deterministic=False)

# Exploration (no reproducibility)
model = ImageClassifier(..., seed=None)
trainer = AutoTrainer(..., seed=None)
```

Happy reproducible training! :material-bullseye:
