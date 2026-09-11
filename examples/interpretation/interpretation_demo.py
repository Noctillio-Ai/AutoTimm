"""
Demonstration of model interpretation capabilities in AutoTimm.

This example shows how to use GradCAM and other interpretation methods
to visualize what your model is looking at when making predictions.
"""

import numpy as np
from PIL import Image

from autotimm import ImageClassifier
from autotimm.interpretation import (
    GradCAM,
    compare_methods,
    explain_prediction,
    quick_explain,
)


def create_sample_image():
    """Create a sample image for demonstration."""
    # Create a simple gradient image
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        img[i, :, 0] = int(255 * i / 224)  # Red gradient
        img[:, i, 1] = int(255 * i / 224)  # Green gradient
    img[:, :, 2] = 128  # Constant blue
    return Image.fromarray(img)


def example_1_quick_explain():
    """Example 1: Quickest way to explain a prediction."""
    print("\n" + "=" * 60)
    print("Example 1: Quick Explanation")
    print("=" * 60)

    # Create a model (using random weights for demo)
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
    )
    model.eval()

    # Create test image
    image = create_sample_image()

    # One-liner explanation
    result = quick_explain(model, image, save_path="quick_explanation.png")

    print(f"✓ Predicted class: {result['predicted_class']}")
    print(f"✓ Method used: {result['method']}")
    print(f"✓ Target layer: {result['target_layer']}")
    print("✓ Saved to: quick_explanation.png")


def example_2_gradcam_basic():
    """Example 2: Basic GradCAM usage."""
    print("\n" + "=" * 60)
    print("Example 2: Basic GradCAM")
    print("=" * 60)

    # Create a model
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
    )
    model.eval()

    # Create explainer
    explainer = GradCAM(model, target_layer="backbone.layer4")

    # Create test image
    image = create_sample_image()

    # Generate heatmap
    heatmap = explainer(image, target_class=5)

    print(f"✓ Generated heatmap with shape: {heatmap.shape}")
    print(f"✓ Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    print(f"✓ Using layer: {explainer.get_target_layer_name()}")


def example_3_compare_methods():
    """Example 3: Compare GradCAM and GradCAM++."""
    print("\n" + "=" * 60)
    print("Example 3: Compare Interpretation Methods")
    print("=" * 60)

    # Create a model
    model = ImageClassifier(
        backbone="resnet34",
        num_classes=10,
    )
    model.eval()

    # Create test image
    image = create_sample_image()

    # Compare methods
    results = compare_methods(
        model=model,
        image=image,
        methods=["gradcam", "gradcam++"],
        layout="horizontal",
        save_path="method_comparison.png",
    )

    print(f"✓ Compared {len(results)} methods")
    for method, result in results.items():
        print(f"  - {method.upper()}: class {result['predicted_class']}")
    print("✓ Saved comparison to: method_comparison.png")


def example_4_high_level_api():
    """Example 4: High-level API with all features."""
    print("\n" + "=" * 60)
    print("Example 4: High-Level API")
    print("=" * 60)

    # Create a model
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
    )
    model.eval()

    # Create test image
    image = create_sample_image()

    # Explain prediction with all options
    result = explain_prediction(
        model=model,
        image=image,
        method="gradcam++",
        target_class=7,  # Explain specific class
        colormap="viridis",  # Use colorblind-friendly colormap
        alpha=0.5,  # Overlay transparency
        save_path="detailed_explanation.png",
        show_original=True,
        dpi=150,  # High quality
        return_heatmap=True,  # Get raw heatmap too
    )

    print(f"✓ Predicted class: {result['predicted_class']}")
    print(f"✓ Explained class: {result['target_class']}")
    print(f"✓ Method: {result['method']}")
    print(f"✓ Heatmap shape: {result['heatmap'].shape}")
    print("✓ Saved to: detailed_explanation.png")


def example_5_custom_layer():
    """Example 5: Visualize different layers."""
    print("\n" + "=" * 60)
    print("Example 5: Compare Different Layers")
    print("=" * 60)

    # Create a model
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
    )
    model.eval()

    # Create test image
    image = create_sample_image()

    # Compare different layers
    layers = ["backbone.layer2", "backbone.layer3", "backbone.layer4"]

    for layer in layers:
        explainer = GradCAM(model, target_layer=layer)
        heatmap = explainer(image)
        print(f"✓ Layer {layer}: heatmap mean = {heatmap.mean():.3f}")


def example_6_batch_processing():
    """Example 6: Batch processing multiple images."""
    print("\n" + "=" * 60)
    print("Example 6: Batch Processing")
    print("=" * 60)

    # Create a model
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
    )
    model.eval()

    # Create multiple test images
    images = [create_sample_image() for _ in range(3)]

    # Batch explain
    explainer = GradCAM(model)
    heatmaps = explainer.explain_batch(images)

    print(f"✓ Processed {len(heatmaps)} images")
    print(f"✓ Heatmap shapes: {[h.shape for h in heatmaps]}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("AutoTimm Model Interpretation Demo")
    print("=" * 60)

    # Run examples
    example_1_quick_explain()
    example_2_gradcam_basic()
    example_3_compare_methods()
    example_4_high_level_api()
    example_5_custom_layer()
    example_6_batch_processing()

    print("\n" + "=" * 60)
    print("✓ All examples completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - quick_explanation.png")
    print("  - method_comparison.png")
    print("  - detailed_explanation.png")
    print("\nTry these methods with your own trained models!")


if __name__ == "__main__":
    main()
