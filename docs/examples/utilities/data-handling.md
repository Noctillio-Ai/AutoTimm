# Data Handling Examples

This page demonstrates data loading, augmentation, and balanced sampling techniques.

## Balanced Sampling

Handle imbalanced datasets with weighted sampling.

```python
import autotimm as at  # recommended alias
from autotimm import ImageDataModule


def main():
    data = ImageDataModule(
        data_dir="./imbalanced_dataset",
        balanced_sampling=True,  # Oversamples minority classes
    )


if __name__ == "__main__":
    main()
```

This feature automatically computes class weights and uses a `WeightedRandomSampler` to ensure balanced training batches even when your dataset has imbalanced class distributions.

---

## Albumentations (Strong Augmentation)

Use strong augmentations with albumentations preset.

```python
from autotimm import ImageDataModule


def main():
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        transform_backend="albumentations",
        augmentation_preset="strong",
    )


if __name__ == "__main__":
    main()
```

The `strong` preset includes:
- Random resized crop
- Horizontal flip
- Color jitter
- Random brightness/contrast
- Random rotation
- Cutout/CoarseDropout
- Normalization

---

## Custom Albumentations Pipeline

Define custom transforms for maximum control:

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
from autotimm import ImageDataModule

custom_train = A.Compose([
    A.RandomResizedCrop(height=224, width=224),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.8),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

custom_val = A.Compose([
    A.Resize(height=256, width=256),
    A.CenterCrop(height=224, width=224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

data = ImageDataModule(
    data_dir="./dataset",
    transform_backend="albumentations",
    train_transforms=custom_train,
    eval_transforms=custom_val,
)
```

---

## Data Module Configuration

Key configuration options:

```python
from autotimm import ImageDataModule

data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",  # or None for folder structure
    image_size=224,
    batch_size=32,
    num_workers=4,
    transform_backend="albumentations",  # or "torchvision"
    augmentation_preset="strong",  # "default", "light", or None
    balanced_sampling=False,  # Enable for imbalanced datasets
    val_split=0.1,
)

data.setup("fit")
print(f"Classes: {data.num_classes}, Training: {len(data.train_dataset)}")
```

---

## Running Examples

```bash
python examples/data_training/balanced_sampling.py
python examples/data_training/albumentations_cifar10.py
python examples/data_training/albumentations_custom_folder.py
```

**See Also:**

- [Transforms User Guide](../../user-guide/data-loading/transforms.md) - Full transform documentation
- [Data Loading Guide](../../user-guide/data-loading/index.md) - Complete data loading options
