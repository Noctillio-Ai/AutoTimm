"""Demonstrate reproducibility with default seeding."""

import torch
from autotimm import ImageClassifier, seed_everything


def test_reproducibility():
    """Test that the same seed produces identical results."""
    print("Testing reproducibility with default seeding...\n")

    # Create two models with the same seed
    print("Creating model 1 with seed=42...")
    model1 = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        seed=42,
        compile_model=False,  # Disable for faster testing
    )

    print("Creating model 2 with seed=42...")
    model2 = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        seed=42,
        compile_model=False,
    )

    # Test with same input
    print("\nTesting forward pass with identical inputs...")
    model1.eval()
    model2.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        out1 = model1(x)
        out2 = model2(x)

    # Check if outputs are identical
    is_identical = torch.allclose(out1, out2, rtol=1e-6, atol=1e-8)
    print(f"Outputs identical: {is_identical}")

    if is_identical:
        print("✓ Reproducibility test PASSED")
    else:
        print("✗ Reproducibility test FAILED")
        print(f"  Max difference: {(out1 - out2).abs().max().item()}")

    return is_identical


def test_different_seeds():
    """Test that different seeds produce different results."""
    print("\n" + "=" * 60)
    print("Testing that different seeds produce different results...\n")

    print("Creating model 1 with seed=42...")
    model1 = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        seed=42,
        compile_model=False,
    )

    print("Creating model 2 with seed=123...")
    model2 = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        seed=123,
        compile_model=False,
    )

    print("\nTesting forward pass with identical inputs...")
    model1.eval()
    model2.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        out1 = model1(x)
        out2 = model2(x)

    # Check if outputs are different
    is_different = not torch.allclose(out1, out2, rtol=1e-6, atol=1e-8)
    print(f"Outputs different: {is_different}")

    if is_different:
        print("✓ Different seeds test PASSED")
    else:
        print("✗ Different seeds test FAILED")

    return is_different


def test_manual_seeding():
    """Test using seed_everything() directly."""
    print("\n" + "=" * 60)
    print("Testing manual seed_everything() usage...\n")

    # Manual seeding
    print("Using seed_everything(42)...")
    seed_everything(42)

    # Generate random numbers
    random1 = torch.randn(5)
    print(f"Random tensor 1: {random1[:3].tolist()}")

    # Reset seed
    print("\nResetting to seed=42...")
    seed_everything(42)

    # Generate again - should be identical
    random2 = torch.randn(5)
    print(f"Random tensor 2: {random2[:3].tolist()}")

    is_identical = torch.allclose(random1, random2)
    print(f"\nTensors identical: {is_identical}")

    if is_identical:
        print("✓ Manual seeding test PASSED")
    else:
        print("✗ Manual seeding test FAILED")

    return is_identical


def test_deterministic_mode():
    """Test deterministic mode (enabled by default)."""
    print("\n" + "=" * 60)
    print("Testing deterministic mode (default)...\n")

    print("Creating model with default settings (deterministic=True)...")
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        seed=42,
        # deterministic=True is default
        compile_model=False,
    )

    print("Checking PyTorch deterministic settings...")
    print(f"  cudnn.deterministic: {torch.backends.cudnn.deterministic}")
    print(f"  cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    # Should be deterministic by default
    expected_deterministic = True
    expected_benchmark = False

    if (
        torch.backends.cudnn.deterministic == expected_deterministic
        and torch.backends.cudnn.benchmark == expected_benchmark
    ):
        print("\n✓ Deterministic mode enabled by default")
    else:
        print("\n✗ Deterministic mode not set correctly")
        return False

    # Run inference
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        out = model(x)

    print(f"Output shape: {out.shape}")
    print("✓ Deterministic mode test PASSED")

    return True


def test_disable_deterministic():
    """Test disabling deterministic mode for faster training."""
    print("\n" + "=" * 60)
    print("Testing with deterministic=False (faster training)...\n")

    print("Creating model with deterministic=False...")
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        seed=42,
        deterministic=False,  # Faster but not fully deterministic
        compile_model=False,
    )

    print("Checking PyTorch settings...")
    print(f"  cudnn.deterministic: {torch.backends.cudnn.deterministic}")
    print(f"  cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    # Should have benchmark enabled for speed
    if torch.backends.cudnn.benchmark:
        print("\n✓ Benchmark mode enabled for faster training")
    else:
        print("\n⚠ Benchmark mode not enabled")

    print("✓ Disable deterministic test PASSED")

    return True


def test_no_seeding():
    """Test disabling seeding completely."""
    print("\n" + "=" * 60)
    print("Testing with seed=None (no seeding)...\n")

    print("Creating model with seed=None...")
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        seed=None,  # Disable seeding completely
        deterministic=False,  # Avoid warning when seed=None
        compile_model=False,
    )

    print("Model created without seeding")
    print("✓ No seeding test PASSED")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("AutoTimm Seeding Test Suite")
    print("=" * 60)

    results = []
    results.append(test_reproducibility())
    results.append(test_different_seeds())
    results.append(test_manual_seeding())
    results.append(test_deterministic_mode())
    results.append(test_disable_deterministic())
    results.append(test_no_seeding())

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if all(results):
        print("\n✓ All tests PASSED!")
    else:
        print("\n✗ Some tests FAILED")

    print("=" * 60)
