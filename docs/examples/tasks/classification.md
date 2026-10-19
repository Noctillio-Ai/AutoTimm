# Classification Examples

This page demonstrates image classification tasks using AutoTimm.

## Classification Workflow

```mermaid
graph TD
    A[Dataset] --> A1[Download/Locate]
    A1 --> A2[Verify Structure]
    A2 --> B[ImageDataModule]
    B --> B1[Configure Transforms]
    B1 --> B2[Setup DataLoaders]
    
    C[Backbone] --> C1[Select Architecture]
    C1 --> C2[Load Pretrained]
    C2 --> D[ImageClassifier]
    D --> D1[Add Task Head]
    
    E[MetricConfig] --> E1[Choose Metrics]
    E1 --> E2[Set Parameters]
    E2 --> F[MetricManager]
    F --> F1[Initialize Metrics]
    F1 --> D
    
    B2 --> G[AutoTrainer]
    D --> G
    H[LoggerConfig] --> H1[Configure Backend]
    H1 --> H2[Set Parameters]
    H2 --> G
    
    G --> G1[Setup Callbacks]
    G1 --> G2[Configure GPUs]
    G2 --> I[Training]
    
    I --> I1[Training Loop]
    I1 --> I2[Log Metrics]
    I2 --> I3[Save Checkpoints]
    I3 --> J[Validation]
    
    J --> J1[Compute Metrics]
    J1 --> J2[Early Stopping Check]
    J2 --> J3{Continue?}
    J3 -->|Yes| I1
    J3 -->|No| K[Testing]
    
    K --> K1[Load Best Model]
    K1 --> K2[Test Evaluation]
    K2 --> L[Results]
    L --> L1[Generate Reports]
    L1 --> L2[Save Predictions]
    L2 --> L3[Plot Metrics]
    
    style B fill:#2196F3,stroke:#1976D2
    style D fill:#1976D2,stroke:#1565C0
    style F fill:#2196F3,stroke:#1976D2
    style G fill:#1976D2,stroke:#1565C0
    style I fill:#2196F3,stroke:#1976D2
    style K fill:#1976D2,stroke:#1565C0
    style L fill:#2196F3,stroke:#1976D2
```

## CIFAR-10 Classification

Basic training with ResNet-18 on CIFAR-10 using MetricManager.

```python
import autotimm as at  # recommended alias
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    MetricConfig,
    MetricManager,
)


def main():
    # Data
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=64,
    )

    # Metrics
    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
    ]

    # Create MetricManager
    metric_manager = MetricManager(configs=metric_configs, num_classes=10)

    # Model
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        metrics=metric_manager,
        lr=1e-3,
    )

    # Train
    trainer = AutoTrainer(
        max_epochs=10,
        logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
        checkpoint_monitor="val/accuracy",
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
```

---

## Custom Folder Dataset

Training on your own images organized in folders.

```python
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    MetricConfig,
    MetricManager,
)


def main():
    # Your data structure:
    # dataset/
    #   train/
    #     class_a/...
    #     class_b/...
    #   val/
    #     class_a/...
    #     class_b/...

    data = ImageDataModule(
        data_dir="./dataset",
        image_size=384,
        batch_size=16,
    )
    data.setup("fit")

    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
    ]

    metric_manager = MetricManager(configs=metric_configs, num_classes=data.num_classes)

    model = ImageClassifier(
        backbone="efficientnet_b3",
        num_classes=data.num_classes,
        metrics=metric_manager,
    )

    trainer = AutoTrainer(
        max_epochs=20,
        logger=[LoggerConfig(backend="wandb", params={"project": "my-project"})],
        checkpoint_monitor="val/accuracy",
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

---

## ViT Fine-Tuning

Two-phase fine-tuning for Vision Transformers with MetricManager.

```python
from autotimm import AutoTrainer, ImageClassifier, MetricConfig, MetricManager


def main():
    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
        ),
    ]

    metric_manager = MetricManager(configs=metric_configs, num_classes=data.num_classes)

    # Phase 1: Linear probe (frozen backbone)
    model = ImageClassifier(
        backbone="vit_base_patch16_224",
        num_classes=data.num_classes,
        metrics=metric_manager,
        freeze_backbone=True,
        lr=1e-2,
    )

    trainer = AutoTrainer(max_epochs=5)
    trainer.fit(model, datamodule=data)

    # Phase 2: Full fine-tune
    for param in model.backbone.parameters():
        param.requires_grad = True

    model._lr = 1e-4
    trainer = AutoTrainer(max_epochs=20, gradient_clip_val=1.0)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

---

## Multi-Label Classification

Multi-label classification where each image can belong to multiple classes simultaneously.
Uses `MultiLabelImageDataModule` for CSV-based data and `ImageClassifier` with `multi_label=True`.

```python
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    MetricConfig,
    MultiLabelImageDataModule,
)


def main():
    # Data - CSV with columns: image_path, cat, dog, outdoor, indoor
    data = MultiLabelImageDataModule(
        train_csv="train.csv",
        val_csv="val.csv",
        image_dir="./images",
        image_size=224,
        batch_size=32,
    )
    data.setup("fit")

    num_labels = data.num_labels  # auto-detected from CSV

    # Multilabel metrics
    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="MultilabelAccuracy",
            params={"num_labels": num_labels},
            stages=["train", "val"],
            prog_bar=True,
        ),
        MetricConfig(
            name="f1",
            backend="torchmetrics",
            metric_class="MultilabelF1Score",
            params={"num_labels": num_labels, "average": "macro"},
            stages=["val"],
        ),
    ]

    # Model - multi_label=True switches to BCEWithLogitsLoss + sigmoid
    # Use loss_fn to override the default loss (e.g., loss_fn="focal" or a custom nn.Module)
    model = ImageClassifier(
        backbone="resnet50",
        num_classes=num_labels,
        multi_label=True,
        threshold=0.5,
        metrics=metrics,
        lr=1e-3,
    )

    # Train
    trainer = AutoTrainer(max_epochs=10)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

**Key differences from single-label:**

- Data: CSV with multi-hot label columns instead of ImageFolder directories
- Model: `multi_label=True` uses `BCEWithLogitsLoss` and sigmoid predictions
- Metrics: Use `Multilabel*` metrics (e.g., `MultilabelAccuracy`, `MultilabelF1Score`)
- `predict_step` returns per-label sigmoid probabilities (each in [0, 1], don't sum to 1)

---

## Running Classification Examples

```bash
git clone https://github.com/Noctillio-Ai/AutoTimm.git
cd AutoTimm
pip install -e ".[all]"

# Run examples
python examples/getting_started/classify_cifar10.py
python examples/getting_started/classify_custom_folder.py
python examples/getting_started/vit_finetuning.py
python examples/data_training/multilabel_classification.py
```

## See Also

- [Image Classifier Guide](../../user-guide/models/image-classifier.md)
- [Multi-Label Data Loading](../../user-guide/data-loading/image-classification-data.md#multi-label-classification-data)
- [Classification Inference](../../user-guide/inference/classification-inference.md)
- [API Reference](../../api/classifier.md)
