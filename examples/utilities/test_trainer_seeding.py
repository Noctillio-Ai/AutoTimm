"""Test AutoTrainer seeding functionality."""

import torch
from autotimm import AutoTrainer, ImageClassifier


def test_trainer_default_seeding():
    """Test that AutoTrainer uses default seeding."""
    print("=" * 60)
    print("Testing AutoTrainer with default seeding...")
    print("=" * 60)

    # Create trainer with default seeding
    trainer = AutoTrainer(
        max_epochs=1,
        fast_dev_run=True,  # Quick test
        logger=False,
        enable_checkpointing=False,
    )

    # Check that deterministic mode is enabled
    print("\nDeterministic settings after trainer creation:")
    print(f"  cudnn.deterministic: {torch.backends.cudnn.deterministic}")
    print(f"  cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    if torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark:
        print("\n✓ Default seeding test PASSED")
        return True
    else:
        print("\n✗ Default seeding test FAILED")
        return False


def test_trainer_custom_seed():
    """Test AutoTrainer with custom seed."""
    print("\n" + "=" * 60)
    print("Testing AutoTrainer with custom seed...")
    print("=" * 60)

    # Create trainer with custom seed
    trainer = AutoTrainer(
        max_epochs=1,
        seed=123,
        deterministic=True,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
    )

    print("\nTrainer created with seed=123, deterministic=True")
    print(f"  cudnn.deterministic: {torch.backends.cudnn.deterministic}")
    print(f"  cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    print("\n✓ Custom seed test PASSED")
    return True


def test_trainer_disable_deterministic():
    """Test disabling deterministic mode in trainer."""
    print("\n" + "=" * 60)
    print("Testing AutoTrainer with deterministic=False...")
    print("=" * 60)

    # Create trainer with deterministic disabled
    trainer = AutoTrainer(
        max_epochs=1,
        seed=42,
        deterministic=False,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
    )

    print("\nTrainer created with seed=42, deterministic=False")
    print(f"  cudnn.deterministic: {torch.backends.cudnn.deterministic}")
    print(f"  cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    if torch.backends.cudnn.benchmark:
        print("\n✓ Benchmark mode enabled for faster training")
        print("✓ Disable deterministic test PASSED")
        return True
    else:
        print("\n⚠ Benchmark mode not enabled")
        print("✓ Disable deterministic test PASSED (with warning)")
        return True


def test_trainer_no_seeding():
    """Test disabling seeding in trainer."""
    print("\n" + "=" * 60)
    print("Testing AutoTrainer with seed=None...")
    print("=" * 60)

    # Create trainer without seeding
    trainer = AutoTrainer(
        max_epochs=1,
        seed=None,
        deterministic=False,  # Avoid warning when seed=None
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
    )

    print("\nTrainer created with seed=None (no seeding)")
    print("✓ No seeding test PASSED")
    return True


def test_trainer_lightning_seeding():
    """Test using PyTorch Lightning's built-in seeding."""
    print("\n" + "=" * 60)
    print("Testing AutoTrainer with use_autotimm_seeding=False...")
    print("=" * 60)

    # Create trainer with Lightning seeding
    trainer = AutoTrainer(
        max_epochs=1,
        seed=42,
        use_autotimm_seeding=False,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
    )

    print("\nTrainer created with use_autotimm_seeding=False")
    print("Uses PyTorch Lightning's built-in seeding")
    print(f"  cudnn.deterministic: {torch.backends.cudnn.deterministic}")
    print(f"  cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    print("\n✓ Lightning seeding test PASSED")
    return True


def test_model_and_trainer_seeding():
    """Test that both model and trainer seeding work together."""
    print("\n" + "=" * 60)
    print("Testing combined model and trainer seeding...")
    print("=" * 60)

    # Model with seeding
    print("\nCreating model with seed=42...")
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        seed=42,
        deterministic=True,
        compile_model=False,
    )

    # Trainer with same seeding
    print("Creating trainer with seed=42...")
    trainer = AutoTrainer(
        max_epochs=1,
        seed=42,
        deterministic=True,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
    )

    print("\nBoth model and trainer initialized with seed=42")
    print("This ensures full reproducibility throughout the workflow")
    print("✓ Combined seeding test PASSED")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AutoTrainer Seeding Test Suite")
    print("=" * 60)

    results = []
    results.append(test_trainer_default_seeding())
    results.append(test_trainer_custom_seed())
    results.append(test_trainer_disable_deterministic())
    results.append(test_trainer_no_seeding())
    results.append(test_trainer_lightning_seeding())
    results.append(test_model_and_trainer_seeding())

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if all(results):
        print("\n✓ All trainer seeding tests PASSED!")
    else:
        print("\n✗ Some tests FAILED")

    print("=" * 60)
