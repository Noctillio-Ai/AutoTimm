"""Example: Custom Datasets & Advanced Augmentation with HuggingFace Models.

This example demonstrates advanced data handling techniques:
- Loading custom image datasets (folder structure, CSV manifest)
- Advanced augmentation strategies (AutoAugment, RandAugment, TrivialAugment)
- Handling imbalanced datasets (weighted sampling, focal loss)
- Multi-class vs multi-label classification
- Data validation and quality checks
- Custom collate functions

Usage:
    python examples/hf_custom_data.py
"""

from __future__ import annotations


def example_1_custom_folder_dataset():
    """Example 1: Loading custom folder-based datasets."""
    print("=" * 80)
    print("Example 1: Custom Folder Dataset (ImageFolder format)")
    print("=" * 80)

    print("\nExpected directory structure:")
    print("""
    /path/to/dataset/
      train/
        class_1/
          image_001.jpg
          image_002.jpg
          ...
        class_2/
          image_001.jpg
          ...
        class_3/
          ...
      val/
        class_1/
          ...
        class_2/
          ...
    """)

    print("Loading custom dataset with ImageDataModule:")
    example_code = """
from autotimm import ImageDataModule

# Method 1: Using ImageDataModule (simplest)
data = ImageDataModule(
    data_dir="/path/to/dataset",
    image_size=224,
    batch_size=32,
    num_workers=4,
    # Optional: specify transforms
    train_preset="augmentation_heavy",
    val_preset="inference",
)

# Setup and discover classes
data.setup("fit")
print(f"Discovered {data.num_classes} classes")
print(f"Train size: {len(data.train_dataset)}")
print(f"Val size: {len(data.val_dataset)}")

# Create model
from autotimm import ImageClassifier
model = ImageClassifier(
    backbone="hf-hub:timm/convnext_tiny.fb_in22k_ft_in1k",
    num_classes=data.num_classes,
)

# Train
from autotimm import AutoTrainer
trainer = AutoTrainer(max_epochs=10)
trainer.fit(model, datamodule=data)
"""

    print(example_code)

    print("\n✓ ImageDataModule automatically:")
    print("  • Discovers classes from folder names")
    print("  • Applies appropriate transforms")
    print("  • Handles train/val/test splits")
    print("  • Supports all timm data configs")


def example_2_csv_manifest_dataset():
    """Example 2: Loading from CSV manifest."""
    print("\n" + "=" * 80)
    print("Example 2: CSV Manifest Dataset")
    print("=" * 80)

    print("\nCSV format example (train.csv):")
    print("""
    image_path,label
    /data/images/img001.jpg,0
    /data/images/img002.jpg,2
    /data/images/img003.jpg,1
    /data/images/img004.jpg,0
    ...
    """)

    print("\nCustom PyTorch Dataset:")
    example_code = '''
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class CSVDataset(Dataset):
    """Dataset loading from CSV manifest."""

    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image = Image.open(row['image_path']).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = row['label']
        return image, label

# Create datasets
from autotimm.data import get_transforms

train_transform = get_transforms(224, "augmentation_medium", "torchvision")
val_transform = get_transforms(224, "inference", "torchvision")

train_dataset = CSVDataset("train.csv", transform=train_transform)
val_dataset = CSVDataset("val.csv", transform=val_transform)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
)
'''

    print(example_code)

    print("\n✓ CSV approach benefits:")
    print("  • Flexible: store metadata, image paths, multiple labels")
    print("  • Easy filtering and sampling")
    print("  • Good for complex datasets")


def example_3_advanced_augmentation():
    """Example 3: Advanced augmentation strategies."""
    print("\n" + "=" * 80)
    print("Example 3: Advanced Augmentation Strategies")
    print("=" * 80)

    print("\nAutoAugment: Automatically learned augmentation policies")
    print("  • Policies learned from CIFAR-10, ImageNet, or SVHN")
    print("  • Combines multiple augmentations with learned magnitude")

    print("\nRandAugment: Random augmentation with controllable magnitude")
    print("  • N: number of augmentation operations (typically 2)")
    print("  • M: magnitude of augmentations (0-30)")

    print("\nTrivialAugment: Simple random augmentation")
    print("  • One random operation per image")
    print("  • Random magnitude")
    print("  • Surprisingly effective despite simplicity")

    example_code = """
import torchvision.transforms as T

# AutoAugment (learned policies)
autoaugment_transform = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# RandAugment (N=2, M=10)
randaugment_transform = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.RandAugment(num_ops=2, magnitude=10),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# TrivialAugment (simplest, often best)
trivialaugment_transform = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.TrivialAugmentWide(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# CutMix and MixUp (best done as collate_fn)
from autotimm.data.mixup import Mixup

mixup_fn = Mixup(
    mixup_alpha=0.2,        # MixUp alpha
    cutmix_alpha=1.0,       # CutMix alpha
    prob=0.5,               # Probability to apply
    num_classes=num_classes,
)

# Use in training loop
for images, labels in train_loader:
    if training:
        images, labels = mixup_fn(images, labels)
    outputs = model(images)
    loss = criterion(outputs, labels)
"""

    print(example_code)

    print("\n✓ When to use each:")
    print("  • Small dataset (<10k): TrivialAugment or RandAugment")
    print("  • Medium dataset (10k-100k): RandAugment or AutoAugment")
    print("  • Large dataset (>100k): BasicAugment is often sufficient")
    print("  • Always try: MixUp or CutMix (1-2% improvement)")


def example_4_imbalanced_datasets():
    """Example 4: Handling imbalanced datasets."""
    print("\n" + "=" * 80)
    print("Example 4: Handling Imbalanced Datasets")
    print("=" * 80)

    print("\nProblem: Class imbalance hurts performance")
    print("  • Model biased toward majority class")
    print("  • Poor performance on minority classes")
    print("  • Common in real-world datasets")

    print("\nSolution 1: Weighted Random Sampling")
    example_code = """
from torch.utils.data import WeightedRandomSampler
import numpy as np

# Compute class weights
class_counts = np.bincount(train_labels)
class_weights = 1. / class_counts
sample_weights = class_weights[train_labels]

# Create weighted sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

# Use with DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,  # Note: don't use shuffle=True with sampler
    num_workers=4,
)
"""

    print(example_code)

    print("\nSolution 2: Focal Loss")
    example_code2 = '''
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Use with model
model = ImageClassifier(
    backbone="hf-hub:timm/resnet50.a1_in1k",
    num_classes=num_classes,
    loss_fn=FocalLoss(alpha=0.25, gamma=2.0),
)
'''

    print(example_code2)

    print("\nSolution 3: Class Weights in Loss")
    example_code3 = """
import torch

# Compute class weights
class_counts = torch.bincount(torch.tensor(train_labels))
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum() * len(class_weights)

# Use with CrossEntropy
criterion = nn.CrossEntropyLoss(weight=class_weights)
"""

    print(example_code3)

    print("\n✓ Which approach to use:")
    print("  • Mild imbalance (1:10): Weighted sampling")
    print("  • Severe imbalance (1:100): Focal Loss + sampling")
    print("  • Very severe (1:1000): Resample + data augmentation + Focal Loss")


def example_5_multi_label_classification():
    """Example 5: Multi-label classification."""
    print("\n" + "=" * 80)
    print("Example 5: Multi-label Classification")
    print("=" * 80)

    print("\nMulti-class vs Multi-label:")
    print("  • Multi-class: Each image has ONE label (cat OR dog OR bird)")
    print("  • Multi-label: Each image can have MULTIPLE labels (cat AND outdoor)")

    example_code = '''
import torch
import torch.nn as nn

# Custom dataset for multi-label
class MultiLabelDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            labels: List of label lists, e.g., [[0,2,5], [1], [3,4]]
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.num_classes = max(max(label_list) for label_list in labels) + 1

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        # Convert to multi-hot encoding
        label_vector = torch.zeros(self.num_classes)
        for label in self.labels[idx]:
            label_vector[label] = 1.0

        return image, label_vector

# Model for multi-label (use sigmoid, not softmax!)
model = ImageClassifier(
    backbone="hf-hub:timm/convnext_tiny.fb_in1k",
    num_classes=num_labels,
)

# Override forward to use sigmoid
class MultiLabelClassifier(ImageClassifier):
    def forward(self, x):
        logits = super().forward(x)
        if isinstance(logits, dict):
            logits = logits['logits']
        return torch.sigmoid(logits)  # Sigmoid for multi-label!

# Use BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()

# Training
for images, labels in train_loader:
    outputs = model(images)
    loss = criterion(outputs, labels)

# Inference (threshold = 0.5)
with torch.inference_mode():
    outputs = model(images)
    predictions = (outputs > 0.5).float()
'''

    print(example_code)

    print("\n✓ Key differences:")
    print("  • Use sigmoid (not softmax)")
    print("  • Use BCEWithLogitsLoss (not CrossEntropyLoss)")
    print("  • Labels are multi-hot vectors (not single integers)")
    print("  • Metrics: hamming distance, subset accuracy, F1-micro/macro")


def example_6_data_validation():
    """Example 6: Data validation and quality checks."""
    print("\n" + "=" * 80)
    print("Example 6: Data Validation & Quality Checks")
    print("=" * 80)

    print("\nCommon data issues:")
    print("  • Corrupted images")
    print("  • Wrong image sizes")
    print("  • Grayscale instead of RGB")
    print("  • Class imbalance")
    print("  • Duplicate images")

    example_code = '''
from PIL import Image
from pathlib import Path
from collections import Counter
import hashlib

def validate_dataset(data_dir):
    """Validate dataset quality."""
    issues = []
    image_paths = list(Path(data_dir).rglob("*.jpg")) + \\
                  list(Path(data_dir).rglob("*.png"))

    print(f"Found {len(image_paths)} images")

    # Track statistics
    sizes = []
    channels = []
    corrupted = []
    hashes = set()
    duplicates = []

    for img_path in image_paths:
        try:
            # Try to load image
            img = Image.open(img_path)
            sizes.append(img.size)
            channels.append(len(img.getbands()))

            # Check for duplicates (via hash)
            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            if img_hash in hashes:
                duplicates.append(img_path)
            hashes.add(img_hash)

        except Exception as e:
            corrupted.append((img_path, str(e)))

    # Report findings
    print(f"\\nValidation Results:")
    print(f"  • Corrupted: {len(corrupted)}")
    print(f"  • Duplicates: {len(duplicates)}")

    # Check image sizes
    size_counter = Counter(sizes)
    print(f"  • Unique sizes: {len(size_counter)}")
    print(f"    Most common: {size_counter.most_common(3)}")

    # Check channels
    channel_counter = Counter(channels)
    print(f"  • Channel distributions: {dict(channel_counter)}")
    if 1 in channel_counter:
        print(f"    WARNING: Found {channel_counter[1]} grayscale images!")

    return {
        'corrupted': corrupted,
        'duplicates': duplicates,
        'sizes': sizes,
        'channels': channels,
    }

# Run validation
results = validate_dataset("/path/to/dataset")

# Remove corrupted images
for corrupted_path, error in results['corrupted']:
    print(f"Removing corrupted: {corrupted_path}")
    # corrupted_path.unlink()  # Uncomment to actually delete
'''

    print(example_code)

    print("\n✓ Best practices:")
    print("  • Always validate before training")
    print("  • Check for corrupted/duplicate images")
    print("  • Verify class balance")
    print("  • Ensure consistent image formats (RGB)")
    print("  • Document data statistics")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Custom Datasets & Advanced Augmentation with HuggingFace Models")
    print("=" * 80)
    print("\nThis example demonstrates advanced data handling techniques")
    print("for real-world datasets.\n")

    # Run examples
    example_1_custom_folder_dataset()
    example_2_csv_manifest_dataset()
    example_3_advanced_augmentation()
    example_4_imbalanced_datasets()
    example_5_multi_label_classification()
    example_6_data_validation()

    print("\n" + "=" * 80)
    print("Key Takeaways")
    print("=" * 80)
    print("\n1. Dataset loading:")
    print("   • ImageFolder: simplest for folder-based datasets")
    print("   • CSV manifest: flexible for complex metadata")
    print("   • Custom Dataset: ultimate flexibility")

    print("\n2. Augmentation strategies:")
    print("   • TrivialAugment: best default choice")
    print("   • RandAugment: when you need control")
    print("   • MixUp/CutMix: almost always beneficial")

    print("\n3. Imbalanced data:")
    print("   • Use weighted sampling or focal loss")
    print("   • Combine multiple techniques for severe imbalance")
    print("   • Monitor per-class metrics")

    print("\n4. Multi-label classification:")
    print("   • Use sigmoid activation (not softmax)")
    print("   • Use BCEWithLogitsLoss")
    print("   • Evaluate with appropriate metrics")

    print("\n5. Data validation:")
    print("   • Always validate before training")
    print("   • Check for corrupted/duplicate images")
    print("   • Document data statistics")

    print("\nNext steps:")
    print("• Validate your dataset with checks above")
    print("• Try TrivialAugment or RandAugment")
    print("• Implement weighted sampling for imbalanced data")
    print("• Consider MixUp for regularization")
    print("• Monitor per-class metrics during training")


if __name__ == "__main__":
    main()
