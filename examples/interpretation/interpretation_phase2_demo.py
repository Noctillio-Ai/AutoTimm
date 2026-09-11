"""
Phase 2: Advanced Model Interpretation Demo

Demonstrates Integrated Gradients, SmoothGrad, Attention Visualization,
and task-specific adapters for detection and segmentation.
"""

import numpy as np
from PIL import Image

from autotimm import ImageClassifier
from autotimm.interpretation import (
    GradCAM,
    IntegratedGradients,
    SmoothGrad,
    explain_prediction,
)


def create_sample_image():
    """Create a sample image for demonstration."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        img[i, :, 0] = int(255 * i / 224)
        img[:, i, 1] = int(255 * i / 224)
    img[:, :, 2] = 128
    return Image.fromarray(img)


def example_1_integrated_gradients():
    """Example 1: Integrated Gradients."""
    print("\n" + "=" * 60)
    print("Example 1: Integrated Gradients")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    image = create_sample_image()

    # Integrated Gradients with different baselines
    for baseline in ["black", "white", "blur"]:
        ig = IntegratedGradients(model, baseline=baseline, steps=50)
        attribution = ig(image, target_class=5)

        print(
            f"✓ IG with {baseline} baseline: shape={attribution.shape}, "
            f"range=[{attribution.min():.3f}, {attribution.max():.3f}]"
        )

    # Visualize polarity
    ig = IntegratedGradients(model, baseline="black", steps=30)
    attribution = ig(image, target_class=5)
    ig.visualize_polarity(attribution, image, save_path="ig_polarity.png")
    print("✓ Saved polarity visualization to: ig_polarity.png")

    # Check completeness
    completeness = ig.get_completeness_score(image, attribution, target_class=5)
    print(f"✓ Completeness error: {completeness:.6f} (should be close to 0)")


def example_2_smoothgrad():
    """Example 2: SmoothGrad for noise reduction."""
    print("\n" + "=" * 60)
    print("Example 2: SmoothGrad")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    image = create_sample_image()

    # Base GradCAM
    base_explainer = GradCAM(model)
    base_heatmap = base_explainer(image, target_class=5)

    # SmoothGrad wrapper
    smooth_explainer = SmoothGrad(
        base_explainer,
        noise_level=0.15,
        num_samples=20,  # Using fewer samples for speed
    )
    smooth_heatmap = smooth_explainer(image, target_class=5)

    print(f"✓ Base GradCAM: shape={base_heatmap.shape}")
    print(f"✓ SmoothGrad: shape={smooth_heatmap.shape}")
    print(
        f"✓ Variance reduction: {base_heatmap.std():.3f} → {smooth_heatmap.std():.3f}"
    )


def example_3_high_level_api_ig():
    """Example 3: Integrated Gradients via high-level API."""
    print("\n" + "=" * 60)
    print("Example 3: High-Level API with Integrated Gradients")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    image = create_sample_image()

    # Use Integrated Gradients through high-level API
    result = explain_prediction(
        model=model,
        image=image,
        method="integrated_gradients",
        target_class=7,
        save_path="ig_explanation.png",
        return_heatmap=True,
    )

    print(f"✓ Predicted class: {result['predicted_class']}")
    print(f"✓ Explained class: {result['target_class']}")
    print(f"✓ Method: {result['method']}")
    print(f"✓ Heatmap shape: {result['heatmap'].shape}")
    print("✓ Saved to: ig_explanation.png")


def example_4_attention_visualization():
    """Example 4: Attention Visualization (ViT)."""
    print("\n" + "=" * 60)
    print("Example 4: Attention Visualization for ViT")
    print("=" * 60)
    print("⚠ Note: This example requires a Vision Transformer model.")
    print("        Skipping for now with standard ResNet model.")
    print("        To use: model = ImageClassifier(backbone='vit_base_patch16_224')")

    # Example code (commented out since we're using ResNet):
    """
    from autotimm.interpretation import AttentionRollout, AttentionFlow

    model = ImageClassifier(backbone="vit_base_patch16_224", num_classes=10)
    model.eval()

    image = create_sample_image()

    # Attention Rollout
    attention = AttentionRollout(model, head_fusion='mean')
    attention_map = attention(image)

    # Visualize
    viz = attention.visualize(
        attention_map,
        image,
        save_path="attention_rollout.png"
    )

    # Attention Flow
    flow = AttentionFlow(model)
    flow_map = flow(image, from_patch=50, layer_idx=-1)
    """


def example_5_compare_all_methods():
    """Example 5: Compare all interpretation methods."""
    print("\n" + "=" * 60)
    print("Example 5: Compare Multiple Methods")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet34", num_classes=10)
    model.eval()

    image = create_sample_image()

    # Individual explanations
    methods = {
        "GradCAM": GradCAM(model),
        "GradCAM++ (Smooth)": SmoothGrad(GradCAM(model), num_samples=10),
        "IG (Black)": IntegratedGradients(model, baseline="black", steps=30),
        "IG (Blur)": IntegratedGradients(model, baseline="blur", steps=30),
    }

    print(f"Comparing {len(methods)} methods...")
    for name, explainer in methods.items():
        heatmap = explainer(image, target_class=5)
        print(f"  ✓ {name}: mean={heatmap.mean():.3f}, std={heatmap.std():.3f}")


def main():
    """Run all Phase 2 examples."""
    print("\n" + "=" * 60)
    print("AutoTimm Phase 2: Advanced Interpretation Demo")
    print("=" * 60)

    # Run examples
    example_1_integrated_gradients()
    example_2_smoothgrad()
    example_3_high_level_api_ig()
    example_4_attention_visualization()
    example_5_compare_all_methods()

    print("\n" + "=" * 60)
    print("✓ All Phase 2 examples completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - ig_polarity.png")
    print("  - ig_explanation.png")
    print("\nPhase 2 Features:")
    print("  ✓ Integrated Gradients (multiple baselines)")
    print("  ✓ SmoothGrad (noise reduction)")
    print("  ✓ Attention Visualization (for ViT models)")
    print("  ✓ Task-specific adapters (detection, segmentation)")
    print("\nFor task-specific examples:")
    print("  - explain_detection() for object detection")
    print("  - explain_segmentation() for semantic segmentation")


if __name__ == "__main__":
    main()
