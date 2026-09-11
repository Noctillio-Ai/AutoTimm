"""Reproducibility with AutoTimm: Comprehensive seeding examples.

This example demonstrates how to achieve fully reproducible training and inference
using AutoTimm's built-in seeding capabilities.
"""

import torch
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    MetricConfig,
    seed_everything,
)


def example_1_default_reproducibility():
    """Example 1: Default behavior (reproducible by default)."""
    print("\n" + "=" * 70)
    print("Example 1: Default Reproducibility")
    print("=" * 70)

    # Both model and trainer use seed=42 by default
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        # seed=42 (default)
        # deterministic=True (default)
    )

    trainer = AutoTrainer(
        max_epochs=10,
        # seed=42 (default)
        # deterministic=True (default)
    )

    print("✓ Model created with default seed=42, deterministic=True")
    print("✓ Trainer created with default seed=42, deterministic=True")
    print("✓ Training will be fully reproducible!")


def example_2_custom_seed():
    """Example 2: Using a custom seed."""
    print("\n" + "=" * 70)
    print("Example 2: Custom Seed")
    print("=" * 70)

    # Use seed=123 for both model and trainer
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        seed=123,
        deterministic=True,
    )

    trainer = AutoTrainer(
        max_epochs=10,
        seed=123,
        deterministic=True,
    )

    print("✓ Using seed=123 for reproducibility")
    print("✓ Same seed will produce identical results across runs")


def example_3_faster_training():
    """Example 3: Disable deterministic mode for faster training."""
    print("\n" + "=" * 70)
    print("Example 3: Faster Training (Non-deterministic)")
    print("=" * 70)

    # Disable deterministic mode for speed
    model = ImageClassifier(
        backbone="resnet50",
        num_classes=1000,
        seed=42,
        deterministic=False,  # Enables cuDNN benchmark mode
    )

    trainer = AutoTrainer(
        max_epochs=100,
        seed=42,
        deterministic=False,  # Faster training
    )

    print("✓ deterministic=False for faster training")
    print("✓ cuDNN benchmark mode enabled")
    print("⚠ Results may vary slightly between runs")
    print("✓ Useful for production training where speed is critical")


def example_4_disable_seeding():
    """Example 4: Disable seeding completely."""
    print("\n" + "=" * 70)
    print("Example 4: Disable Seeding")
    print("=" * 70)

    # No seeding (set deterministic=False to avoid warning)
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        seed=None,  # Disable seeding
        deterministic=False,  # Must be False when seed=None
    )

    trainer = AutoTrainer(
        max_epochs=10,
        seed=None,  # Disable seeding
        deterministic=False,  # Must be False when seed=None
    )

    print("✓ Seeding disabled (seed=None)")
    print("✓ Results will vary between runs")
    print("✓ Useful for exploring model variance")


def example_5_manual_seeding():
    """Example 5: Manual seeding with seed_everything()."""
    print("\n" + "=" * 70)
    print("Example 5: Manual Seeding")
    print("=" * 70)

    # Generate some random numbers with seeding
    seed_everything(42, deterministic=True)
    random1 = torch.randn(5)
    print(f"Random tensor 1: {random1[:3].tolist()}")

    # Reset seed - should get same values
    print("\nResetting to seed=42...")
    seed_everything(42, deterministic=True)
    random2 = torch.randn(5)
    print(f"Random tensor 2: {random2[:3].tolist()}")

    if torch.allclose(random1, random2):
        print("✓ Manual seeding works - tensors are identical!")

    # Manual seeding before creating model
    print("\nUsing manual seeding before model creation...")
    seed_everything(123, deterministic=True)
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        seed=None,  # Don't seed again in model
        deterministic=False,  # Already set by manual seed_everything
    )
    print("✓ Model created with manual seed")


def example_6_reproducible_inference():
    """Example 6: Reproducible inference."""
    print("\n" + "=" * 70)
    print("Example 6: Reproducible Inference")
    print("=" * 70)

    # Create model with seeding
    model1 = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        seed=42,
        compile_model=False,
    )
    model1.eval()

    # Create another model with same seed
    model2 = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        seed=42,
        compile_model=False,
    )
    model2.eval()

    # Test with same input
    x = torch.randn(1, 3, 224, 224)

    with torch.inference_mode():
        out1 = model1(x)
        out2 = model2(x)

    if torch.allclose(out1, out2, rtol=1e-5):
        print("✓ Inference is reproducible!")
        print(f"  Output 1: {out1[0, :3].tolist()}")
        print(f"  Output 2: {out2[0, :3].tolist()}")
    else:
        print("✗ Outputs differ")


def example_7_trainer_seeding_options():
    """Example 7: Trainer seeding options."""
    print("\n" + "=" * 70)
    print("Example 7: Trainer Seeding Options")
    print("=" * 70)

    # Option 1: Use Lightning's built-in seeding (default)
    trainer1 = AutoTrainer(
        max_epochs=10,
        seed=42,
        use_autotimm_seeding=False,  # Default: uses Lightning's seeding
    )
    print("✓ Trainer 1: Using PyTorch Lightning's built-in seeding")

    # Option 2: Use AutoTimm's custom seeding
    trainer2 = AutoTrainer(
        max_epochs=10,
        seed=42,
        use_autotimm_seeding=True,  # Use AutoTimm's seed_everything
    )
    print("✓ Trainer 2: Using AutoTimm's custom seed_everything()")

    print("\nBoth options provide reproducibility:")
    print("  - Lightning's seeding: Standard Lightning behavior")
    print("  - AutoTimm's seeding: More comprehensive control")


def example_8_complete_workflow():
    """Example 8: Complete reproducible training workflow."""
    print("\n" + "=" * 70)
    print("Example 8: Complete Reproducible Workflow")
    print("=" * 70)

    # Set seed for the entire workflow
    seed = 42

    # 1. Create model with seeding
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        seed=seed,
        deterministic=True,
        compile_model=False,
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

    # 2. Create data module with seeding (commented out - requires data)
    # data = ImageDataModule(
    #     data_dir="./imagenette2-160",
    #     batch_size=32,
    #     num_workers=4,
    #     image_size=224,
    #     seed=seed,  # Seed for data splitting
    # )

    # 3. Create trainer with seeding
    trainer = AutoTrainer(
        max_epochs=10,
        seed=seed,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        fast_dev_run=True,  # Quick test
    )

    print(f"✓ Complete workflow seeded with seed={seed}")
    print("✓ Model, data, and trainer all use consistent seeding")
    print("✓ Results will be identical across multiple runs")
    print("\nRun this workflow multiple times - you'll get the same results!")

    # Uncomment to actually train:
    # trainer.fit(model, datamodule=data)


def example_9_research_paper_setup():
    """Example 9: Setup for research papers (strict reproducibility)."""
    print("\n" + "=" * 70)
    print("Example 9: Research Paper Setup")
    print("=" * 70)

    print("For maximum reproducibility in research:")
    print()

    # Strict reproducibility settings
    SEED = 42

    # 1. Global seeding
    seed_everything(SEED, deterministic=True)

    # 2. Model with deterministic mode
    model = ImageClassifier(
        backbone="resnet50",
        num_classes=1000,
        seed=SEED,
        deterministic=True,
        compile_model=False,  # Disable for consistency
    )

    # 3. Trainer with strict settings
    trainer = AutoTrainer(
        max_epochs=100,
        seed=SEED,
        deterministic=True,
        precision=32,  # Use full precision for maximum reproducibility
    )

    print(f"✓ Seed: {SEED}")
    print("✓ Deterministic mode: Enabled")
    print("✓ torch.compile: Disabled for consistency")
    print("✓ Precision: 32 (full precision)")
    print("✓ cuDNN deterministic: Enabled")
    print("✓ cuDNN benchmark: Disabled")
    print()
    print("This setup ensures:")
    print("  - Identical results across runs")
    print("  - Reproducible results on same hardware")
    print("  - Suitable for research paper submissions")


def main():
    """Run all examples."""
    print("=" * 70)
    print("AutoTimm Reproducibility Examples")
    print("=" * 70)

    # Run all examples
    example_1_default_reproducibility()
    example_2_custom_seed()
    example_3_faster_training()
    example_4_disable_seeding()
    example_5_manual_seeding()
    example_6_reproducible_inference()
    example_7_trainer_seeding_options()
    example_8_complete_workflow()
    example_9_research_paper_setup()

    print("\n" + "=" * 70)
    print("All Examples Complete!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  1. AutoTimm is reproducible by default (seed=42)")
    print("  2. Use deterministic=True for strict reproducibility")
    print("  3. Use deterministic=False for faster training")
    print("  4. Both model and trainer support seeding")
    print("  5. Use seed_everything() for manual control")
    print("=" * 70)


if __name__ == "__main__":
    main()
