# Training Utilities Examples

This page demonstrates training optimization utilities including auto-tuning and multi-GPU training.

## Training Optimization Strategies

```mermaid
graph TD
    A[Training Config] --> A1[Define Hyperparams]
    A1 --> A2[Set Requirements]
    A2 --> B{Optimizations}
    
    B -->|Auto-tuning| C1[LR Finder]
    C1 --> C1a[Range Test]
    C1a --> C1b[Plot Loss Curve]
    C1b --> C1c[Select Optimal LR]
    
    B -->|Auto-tuning| C2[Batch Size Finder]
    C2 --> C2a[Binary Search]
    C2a --> C2b[Test Memory]
    C2b --> C2c[Find Max Batch Size]
    
    B -->|Multi-GPU| C3[DDP Strategy]
    C3 --> C3a[Initialize Process Group]
    C3a --> C3b[Distribute Model]
    C3b --> C3c[Sync Gradients]
    
    B -->|Precision| C4[Mixed Precision]
    C4 --> C4a[FP16/BF16]
    C4a --> C4b[Loss Scaling]
    C4b --> C4c[Autocast]
    
    B -->|Memory| C5[Gradient Accumulation]
    C5 --> C5a[Accumulate Steps]
    C5a --> C5b[Delayed Update]
    C5b --> C5c[Effective Batch]
    
    C1c --> D[Optimal LR]
    D --> D1[Set Learning Rate]
    D1 --> D2[Configure Optimizer]
    
    C2c --> E[Optimal Batch Size]
    E --> E1[Set Batch Size]
    E1 --> E2[Configure DataLoader]
    
    C3c --> F[Parallel Training]
    F --> F1[Multi-GPU Speedup]
    F1 --> F2[Reduced Time]
    
    C4c --> G[Faster Training]
    G --> G1[Lower Memory]
    G1 --> G2[Faster Computation]
    
    C5c --> H[Larger Effective Batch]
    H --> H1[Better Gradients]
    H1 --> H2[Stable Training]
    
    D2 --> I[Better Convergence]
    E2 --> I
    H2 --> I
    
    F2 --> J[Faster Training]
    G2 --> J
    
    I --> K[Final Model]
    J --> K
    
    K --> K1[Optimized Training]
    K1 --> K2[Best Performance]
    K2 --> K3[Production Ready]
    
    style A fill:#2196F3,stroke:#1976D2
    style C1 fill:#1976D2,stroke:#1565C0
    style C2 fill:#2196F3,stroke:#1976D2
    style C3 fill:#1976D2,stroke:#1565C0
    style C4 fill:#2196F3,stroke:#1976D2
    style C5 fill:#1976D2,stroke:#1565C0
    style I fill:#2196F3,stroke:#1976D2
    style J fill:#1976D2,stroke:#1565C0
    style K fill:#2196F3,stroke:#1976D2
```

## Auto-Tuning

Automatically find optimal learning rate and batch size.

```python
import autotimm as at  # recommended alias
from autotimm import AutoTrainer, TunerConfig


def main():
    trainer = AutoTrainer(
        max_epochs=10,
        tuner_config=TunerConfig(
            auto_lr=True,
            auto_batch_size=True,
            lr_find_kwargs={"min_lr": 1e-6, "max_lr": 1.0, "num_training": 100},
            batch_size_kwargs={"mode": "power", "init_val": 16},
        ),
    )

    trainer.fit(model, datamodule=data)  # Runs tuning before training


if __name__ == "__main__":
    main()
```

**TunerConfig Options:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `auto_lr` | Enable automatic learning rate finding | `False` |
| `auto_batch_size` | Enable automatic batch size finding | `False` |
| `lr_find_kwargs` | Arguments for LR finder | `{"min_lr": 1e-6, "max_lr": 1.0}` |
| `batch_size_kwargs` | Arguments for batch size finder | `{"mode": "power"}` |

---

## Multi-GPU Training

Distributed training across multiple GPUs.

```python
from autotimm import AutoTrainer


def main():
    trainer = AutoTrainer(
        max_epochs=10,
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        precision="bf16-mixed",
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

**Multi-GPU Strategies:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| `ddp` | Distributed Data Parallel | Most use cases |
| `ddp_spawn` | DDP with process spawning | Debugging |
| `fsdp` | Fully Sharded Data Parallel | Very large models |

**Precision Options:**

| Precision | Speed | Memory |
|-----------|-------|--------|
| `32` | Slowest | Highest |
| `16-mixed` | Faster | Lower |
| `bf16-mixed` | Faster | Lower |

---

## Transform Backend Recommendation

Get a recommended transform backend and preset for your task.

```python
from autotimm import recommend_backend, ImageDataModule


def main():
    # Get a recommendation for your use case
    rec = recommend_backend(
        task="classification",
        needs_advanced_augmentation=False,
    )
    print(rec)

    # Convert the recommendation into a TransformConfig
    config = rec.to_config(image_size=224)

    data = ImageDataModule(
        data_dir="./data",
        transform_config=config,
        backbone="resnet50",
    )


if __name__ == "__main__":
    main()
```

**Recommendation Features:**

- **Task-aware**: Suggests albumentations for detection/segmentation
- **Preset selection**: Picks a matching augmentation preset
- **Reasoning**: Explains why a backend was chosen
- **Comparison**: Use `compare_backends()` to see both side by side

---

## Performance Optimization

Optimize training performance with various techniques.

```python
from autotimm import AutoTrainer, ImageClassifier, ImageDataModule
import torch


def main():
    # Enable performance optimizations
    torch.set_float32_matmul_precision('high')  # Use TensorFloat-32
    
    # Data optimizations
    data = ImageDataModule(
        data_dir="./data",
        batch_size=64,
        num_workers=8,  # Parallel data loading
        pin_memory=True,  # Fast GPU transfer
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2,  # Prefetch batches
    )
    
    # Model with optimizations
    model = ImageClassifier(
        backbone="resnet50",
        num_classes=10,
        compile_model=True,  # torch.compile (PyTorch 2.0+)
    )
    
    # Trainer with performance settings
    trainer = AutoTrainer(
        max_epochs=50,
        precision="bf16-mixed",  # BFloat16 for speed
        accelerator="gpu",
        devices=1,
        benchmark=True,  # cuDNN autotuner
        deterministic=False,  # Allow non-deterministic ops
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,  # Gradient accumulation
    )
    
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

**Performance Optimization Techniques:**

| Technique | Speed Gain | Trade-off |
|-----------|------------|-----------|
| Mixed precision (bf16) | 2-3x | Minimal accuracy impact |
| Channels last | 10-20% | None |
| torch.compile | 20-40% | Longer startup time |
| Persistent workers | 5-15% | More memory |
| Gradient accumulation | Enable larger batch | Slower updates |
| cuDNN benchmark | 5-10% | Non-deterministic |

**Best Practices:**

1. **Start simple**: Enable one optimization at a time
2. **Profile first**: Use PyTorch Profiler to identify bottlenecks
3. **Monitor accuracy**: Ensure optimizations don't hurt model quality
4. **Test thoroughly**: Some optimizations are hardware-specific

---

## Running Examples

```bash
python examples/data_training/auto_tuning.py
python examples/data_training/multi_gpu_training.py
python examples/data_training/preset_manager.py
python examples/data_training/performance_optimization_demo.py
```

**See Also:**

- [Training User Guide](../../user-guide/training/training.md) - Full training documentation
- [Inference Guide](../../user-guide/inference/index.md) - Model inference and deployment
