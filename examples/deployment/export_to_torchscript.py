"""Export trained models to TorchScript for production deployment.

This example demonstrates how to export AutoTimm models to TorchScript format,
which enables deployment without Python dependencies to C++, mobile, and edge devices.
"""

import torch

from autotimm import ImageClassifier, export_to_torchscript, load_torchscript
from autotimm.core.metrics import MetricConfig


def example_basic_export():
    """Basic TorchScript export example."""
    print("=" * 60)
    print("Example 1: Basic TorchScript Export")
    print("=" * 60)

    # Create and train a model (or load from checkpoint)
    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["val"],
        )
    ]

    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        metrics=metrics,
    )
    model.eval()

    # Export to TorchScript
    example_input = torch.randn(1, 3, 224, 224)
    scripted_model = export_to_torchscript(
        model=model,
        save_path="resnet18_classifier.pt",
        example_input=example_input,
        method="trace",  # Recommended method
    )

    print("✓ Model exported to: resnet18_classifier.pt")

    # Load and test
    loaded_model = load_torchscript("resnet18_classifier.pt")
    with torch.inference_mode():
        output = loaded_model(example_input)
    print(f"✓ Loaded model output shape: {output.shape}")
    print()


def example_convenience_method():
    """Using the convenience to_torchscript() method."""
    print("=" * 60)
    print("Example 2: Using Convenience Method")
    print("=" * 60)

    model = ImageClassifier(
        backbone="efficientnet_b0",
        num_classes=100,
    )
    model.eval()

    # One-line export using convenience method
    scripted_model = model.to_torchscript("efficientnet_classifier.pt")

    print("✓ Model exported using to_torchscript() method")
    print("✓ File: efficientnet_classifier.pt")
    print()


def example_checkpoint_export():
    """Export from checkpoint file."""
    print("=" * 60)
    print("Example 3: Export from Checkpoint")
    print("=" * 60)

    # First, create and save a checkpoint
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
    )

    # In practice, you would train the model first
    # For this example, we'll just demonstrate the export process

    print("Note: In production, load your trained checkpoint:")
    print("  model = ImageClassifier.load_from_checkpoint('model.ckpt')")
    print("  model.to_torchscript('model.pt')")
    print()


def example_with_validation():
    """Export with validation to ensure correctness."""
    print("=" * 60)
    print("Example 4: Export with Validation")
    print("=" * 60)

    from autotimm.export import validate_torchscript_export

    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
    )
    model.eval()

    # Export
    example_input = torch.randn(1, 3, 224, 224)
    scripted_model = export_to_torchscript(
        model=model,
        save_path="validated_model.pt",
        example_input=example_input,
    )

    # Validate that outputs match
    is_valid = validate_torchscript_export(
        original_model=model,
        scripted_model=scripted_model,
        example_input=example_input,
        rtol=1e-5,
        atol=1e-8,
    )

    print(f"✓ Export validation: {'PASSED' if is_valid else 'FAILED'}")
    print("✓ Model exported to: validated_model.pt")
    print()


def example_production_inference():
    """Production inference with TorchScript model."""
    print("=" * 60)
    print("Example 5: Production Inference")
    print("=" * 60)

    # First export a model
    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()
    example_input = torch.randn(1, 3, 224, 224)
    export_to_torchscript(model, "production_model.pt", example_input)

    # Production deployment code (no AutoTimm required!)
    print("\n--- Production Code (no AutoTimm dependency) ---")
    print("""
import torch

# Load model
model = torch.jit.load("production_model.pt")
model.eval()

# Prepare input (e.g., from image file)
# image = preprocess_image("image.jpg")
image = torch.randn(1, 3, 224, 224)

# Run inference
with torch.inference_mode():
    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

print(f"Predicted class: {predicted_class.item()}")
print(f"Confidence: {probabilities.max().item():.2%}")
    """)

    print("✓ TorchScript model can be deployed without AutoTimm!")
    print()


def example_different_image_sizes():
    """Export models for different input sizes."""
    print("=" * 60)
    print("Example 6: Different Input Sizes")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    # Export for different input sizes
    sizes = [(224, 224), (299, 299), (384, 384)]

    for h, w in sizes:
        example_input = torch.randn(1, 3, h, w)
        save_path = f"model_{h}x{w}.pt"

        export_to_torchscript(
            model=model,
            save_path=save_path,
            example_input=example_input,
        )
        print(f"✓ Exported model for {h}x{w} input: {save_path}")

    print("\nNote: Each exported model is optimized for its specific input size")
    print()


def example_batch_inference():
    """Export model that supports batch inference."""
    print("=" * 60)
    print("Example 7: Batch Inference Support")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    # Export with batch size > 1
    batch_size = 8
    example_input = torch.randn(batch_size, 3, 224, 224)

    export_to_torchscript(
        model=model,
        save_path="batch_model.pt",
        example_input=example_input,
    )

    # Test batch inference
    loaded_model = torch.jit.load("batch_model.pt")
    test_batch = torch.randn(16, 3, 224, 224)

    with torch.inference_mode():
        output = loaded_model(test_batch)

    print(f"✓ Exported model with batch size: {batch_size}")
    print(f"✓ Tested with batch size: {test_batch.shape[0]}")
    print(f"✓ Output shape: {output.shape}")
    print()


def example_optimization_options():
    """Export with different optimization options."""
    print("=" * 60)
    print("Example 8: Optimization Options")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()
    example_input = torch.randn(1, 3, 224, 224)

    # Export with optimization (default)
    export_to_torchscript(
        model=model,
        save_path="optimized_model.pt",
        example_input=example_input,
        optimize=True,  # Default: applies inference optimizations
    )
    print("✓ Exported with optimization: optimized_model.pt")

    # Export without optimization (for debugging)
    export_to_torchscript(
        model=model,
        save_path="debug_model.pt",
        example_input=example_input,
        optimize=False,  # Keep original for debugging
    )
    print("✓ Exported without optimization: debug_model.pt")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("AutoTimm TorchScript Export Examples")
    print("=" * 60 + "\n")

    example_basic_export()
    example_convenience_method()
    example_checkpoint_export()
    example_with_validation()
    example_production_inference()
    example_different_image_sizes()
    example_batch_inference()
    example_optimization_options()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train your model with AutoTrainer")
    print("2. Export to TorchScript: model.to_torchscript('model.pt')")
    print("3. Deploy to production with just PyTorch")
    print("\nFor more information, see:")
    print("- docs/user-guide/deployment/torchscript-export.md")
    print("- TORCHSCRIPT_EXPORT.md")


if __name__ == "__main__":
    main()
