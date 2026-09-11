"""Example: Using Hugging Face Hub models with AutoTimm.

This example demonstrates how to use timm-compatible models from Hugging Face Hub
for image classification. HF Hub provides access to thousands of pretrained models
that can be used directly with AutoTimm.

Usage:
    python examples/huggingface_hub_models.py
"""

from __future__ import annotations

import autotimm as at  # recommended alias
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    MetricConfig,
    list_hf_hub_backbones,
)


def discover_hf_hub_models():
    """Discover available models on Hugging Face Hub."""
    print("=" * 80)
    print("Discovering Hugging Face Hub Models")
    print("=" * 80)

    # List official timm models on HF Hub
    print("\nOfficial timm models (first 10):")
    timm_models = list_hf_hub_backbones(limit=10)
    for model in timm_models:
        print(f"  - {model}")

    # Search for specific model types
    print("\nSearching for ResNet models on HF Hub:")
    resnet_models = list_hf_hub_backbones(model_name="resnet", limit=5)
    for model in resnet_models:
        print(f"  - {model}")

    print("\nSearching for ConvNeXt models on HF Hub:")
    convnext_models = list_hf_hub_backbones(model_name="convnext", limit=5)
    for model in convnext_models:
        print(f"  - {model}")


def train_with_hf_hub_model():
    """Train an image classifier using a Hugging Face Hub model."""
    print("\n" + "=" * 80)
    print("Training with Hugging Face Hub Model")
    print("=" * 80)

    # Data setup
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=32,
        num_workers=4,
    )

    # Metrics configuration
    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass", "num_classes": 10},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
        MetricConfig(
            name="top5_acc",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass", "num_classes": 10, "top_k": 5},
            stages=["val", "test"],
            prog_bar=False,
        ),
    ]

    # Model using a Hugging Face Hub backbone
    # You can use any timm-compatible model from HF Hub with the hf-hub: prefix
    print("\nCreating model with HF Hub backbone: hf-hub:timm/resnet50.a1_in1k")

    model = ImageClassifier(
        backbone="hf-hub:timm/resnet50.a1_in1k",  # HF Hub model
        num_classes=10,
        metrics=metrics,
        lr=1e-3,
        optimizer="adamw",
    )

    # Print model info
    print(f"Total parameters: {at.count_parameters(model):,}")
    print(f"Trainable parameters: {at.count_parameters(model, trainable_only=True):,}")

    # Trainer setup
    trainer = AutoTrainer(
        max_epochs=5,
        accelerator="auto",
        precision="16-mixed",
        logger_configs=[],  # Disable loggers for this example
    )

    # Train the model
    print("\nStarting training...")
    trainer.fit(model, datamodule=data)

    # Test the model
    print("\nTesting model...")
    trainer.test(model, datamodule=data)


def compare_timm_vs_hf_hub():
    """Compare using a standard timm model vs HF Hub model."""
    print("\n" + "=" * 80)
    print("Comparison: Timm vs HF Hub Models")
    print("=" * 80)

    # Create both types of models
    print("\n1. Creating standard timm model (resnet18)...")
    timm_model = at.create_backbone("resnet18")
    print(f"   Features: {timm_model.num_features}")
    print(f"   Parameters: {at.count_parameters(timm_model, trainable_only=False):,}")

    print("\n2. Creating HF Hub model (hf-hub:timm/resnet18.a1_in1k)...")
    hf_model = at.create_backbone("hf-hub:timm/resnet18.a1_in1k")
    print(f"   Features: {hf_model.num_features}")
    print(f"   Parameters: {at.count_parameters(hf_model, trainable_only=False):,}")

    print("\nBoth models have the same architecture and can be used interchangeably!")
    print("HF Hub models provide:")
    print("  - Centralized model hosting")
    print("  - Version control and model cards")
    print("  - Easy sharing and collaboration")
    print("  - Community-contributed models")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Hugging Face Hub Models with AutoTimm")
    print("=" * 80)

    # Example 1: Discover models
    discover_hf_hub_models()

    # Example 2: Compare model types
    compare_timm_vs_hf_hub()

    # Example 3: Train with HF Hub model (optional - can be slow)
    # Uncomment the line below to run training
    # train_with_hf_hub_model()

    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Use 'hf-hub:' prefix to load models from Hugging Face Hub")
    print("  2. HF Hub models work seamlessly with all AutoTimm tasks")
    print("  3. Use list_hf_hub_backbones() to discover available models")
    print("  4. Both timm and HF Hub models use the same API")


if __name__ == "__main__":
    main()
