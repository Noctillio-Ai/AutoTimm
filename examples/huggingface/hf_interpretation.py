"""Example: Model Interpretation with HuggingFace Hub Models.

This example demonstrates comprehensive interpretation and explainability techniques
for models loaded from HuggingFace Hub, including:
- GradCAM and GradCAM++ for CNNs
- Attention visualization for Vision Transformers
- Feature visualization across different architectures
- Quantitative evaluation metrics
- Interactive visualizations

Usage:
    python examples/hf_interpretation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from autotimm import ImageClassifier, list_hf_hub_backbones
from autotimm.interpretation import (
    AttentionVisualizer,
    GradCAM,
    GradCAMPlusPlus,
    IntegratedGradients,
)
from autotimm.interpretation.metrics import ExplanationMetrics
from autotimm.interpretation.visualization.heatmap import save_heatmap

try:
    from autotimm.interpretation.interactive import InteractiveVisualizer

    INTERACTIVE_AVAILABLE = True
except ImportError:
    INTERACTIVE_AVAILABLE = False
    print(
        "⚠ Interactive visualization not available. Install plotly for interactive features."
    )


def create_sample_image(size: int = 224) -> Image.Image:
    """Create a sample image for demonstration."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Create a simple pattern
    for i in range(size):
        img[i, :, 0] = int(255 * i / size)
        img[:, i, 1] = int(255 * (size - i) / size)
    img[:, :, 2] = 128
    return Image.fromarray(img)


def discover_interpretable_hf_models():
    """Discover HuggingFace Hub models suitable for interpretation."""
    print("=" * 80)
    print("HuggingFace Hub Models for Interpretation")
    print("=" * 80)

    print("\n1. ResNet models (excellent for GradCAM):")
    resnets = list_hf_hub_backbones(model_name="resnet", limit=3)
    for model in resnets[:3]:
        print(f"   • {model}")

    print("\n2. ConvNeXt models (modern CNNs with clear feature hierarchies):")
    convnexts = list_hf_hub_backbones(model_name="convnext", limit=3)
    for model in convnexts[:3]:
        print(f"   • {model}")

    print("\n3. Vision Transformers (for attention visualization):")
    vits = list_hf_hub_backbones(model_name="vit", limit=3)
    for model in vits[:3]:
        print(f"   • {model}")

    print("\n4. DeiT models (distilled ViTs with enhanced interpretability):")
    deits = list_hf_hub_backbones(model_name="deit", limit=3)
    for model in deits[:3]:
        print(f"   • {model}")


def example_1_gradcam_with_hf_resnet():
    """Example 1: GradCAM with HuggingFace ResNet."""
    print("\n" + "=" * 80)
    print("Example 1: GradCAM with HF Hub ResNet")
    print("=" * 80)

    # Load ResNet from HF Hub
    model = ImageClassifier(
        backbone="hf-hub:timm/resnet18.a1_in1k",
        num_classes=10,
        freeze_backbone=False,
    )
    model.eval()

    print("✓ Loaded model: resnet18.a1_in1k from HuggingFace Hub")
    print(f"✓ Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create test image
    image = create_sample_image()

    # Create GradCAM explainer
    explainer = GradCAM(model, target_layer="backbone.layer4")
    print(f"✓ Created GradCAM explainer targeting: {explainer.get_target_layer_name()}")

    # Generate explanation
    heatmap = explainer(image, target_class=0)
    print(
        f"✓ Generated heatmap: shape={heatmap.shape}, range=[{heatmap.min():.3f}, {heatmap.max():.3f}]"
    )

    # Save visualization
    save_heatmap(
        heatmap,
        image,
        save_path="outputs/hf_resnet_gradcam.png",
        method="gradcam",
        show_original=True,
    )
    print("✓ Saved visualization to: outputs/hf_resnet_gradcam.png")


def example_2_compare_gradcam_variants():
    """Example 2: Compare GradCAM and GradCAM++ on HF models."""
    print("\n" + "=" * 80)
    print("Example 2: Compare GradCAM Variants")
    print("=" * 80)

    # Load ConvNeXt from HF Hub
    model = ImageClassifier(
        backbone="hf-hub:timm/convnext_tiny.fb_in1k",
        num_classes=10,
    )
    model.eval()

    print("✓ Loaded ConvNeXt-Tiny from HuggingFace Hub")

    # Create test image
    image = create_sample_image()

    # Compare methods
    methods = {
        "gradcam": GradCAM(model, target_layer="backbone.stages.3"),
        "gradcam++": GradCAMPlusPlus(model, target_layer="backbone.stages.3"),
    }

    print("\nGenerating explanations with different methods:")
    for method_name, explainer in methods.items():
        heatmap = explainer(image, target_class=0)
        save_heatmap(
            heatmap,
            image,
            save_path=f"outputs/hf_convnext_{method_name}.png",
            method=method_name,
        )
        print(
            f"  • {method_name:12s}: heatmap range [{heatmap.min():.3f}, {heatmap.max():.3f}]"
        )

    print("✓ Saved all comparison visualizations to outputs/")


def example_3_attention_visualization_vit():
    """Example 3: Attention visualization for Vision Transformers."""
    print("\n" + "=" * 80)
    print("Example 3: Attention Visualization with HF ViT")
    print("=" * 80)

    # Load ViT from HF Hub
    model = ImageClassifier(
        backbone="hf-hub:timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        num_classes=10,
    )
    model.eval()

    print("✓ Loaded ViT-Tiny from HuggingFace Hub")

    # Create test image
    image = create_sample_image()

    # Create attention visualizer
    try:
        visualizer = AttentionVisualizer(model)
        print("✓ Created attention visualizer for ViT model")

        # Get attention maps
        attention_maps = visualizer.get_attention_maps(image)
        print(f"✓ Extracted attention from {len(attention_maps)} layers")

        # Visualize attention rollout
        rollout = visualizer.attention_rollout(image)
        print(f"✓ Computed attention rollout: shape={rollout.shape}")

        # Save visualization
        save_heatmap(
            rollout,
            image,
            save_path="outputs/hf_vit_attention.png",
            method="attention",
        )
        print("✓ Saved attention visualization to: outputs/hf_vit_attention.png")

    except Exception as e:
        print(f"⚠ Attention visualization not supported for this model: {e}")
        print("  Using GradCAM as fallback...")

        # Fallback to GradCAM for ViT
        explainer = GradCAM(model, target_layer="backbone.blocks.11")
        heatmap = explainer(image, target_class=0)
        save_heatmap(
            heatmap,
            image,
            save_path="outputs/hf_vit_gradcam_fallback.png",
            method="gradcam",
        )
        print("✓ Saved GradCAM fallback to: outputs/hf_vit_gradcam_fallback.png")


def example_4_multi_architecture_comparison():
    """Example 4: Compare interpretation across different HF architectures."""
    print("\n" + "=" * 80)
    print("Example 4: Multi-Architecture Interpretation Comparison")
    print("=" * 80)

    architectures = {
        "ResNet-18": "hf-hub:timm/resnet18.a1_in1k",
        "EfficientNet-B0": "hf-hub:timm/efficientnet_b0.ra_in1k",
        "ConvNeXt-Tiny": "hf-hub:timm/convnext_tiny.fb_in1k",
    }

    # Create test image
    image = create_sample_image()

    print("\nComparing GradCAM across architectures:")
    print(f"{'Architecture':<20} {'Parameters':>12} {'Heatmap Range':>20}")
    print("-" * 80)

    for arch_name, backbone_name in architectures.items():
        try:
            # Create model
            model = ImageClassifier(backbone=backbone_name, num_classes=10)
            model.eval()

            # Count parameters
            n_params = sum(p.numel() for p in model.parameters())

            # Create explainer (try to find last conv layer)
            if "resnet" in backbone_name:
                target_layer = "backbone.layer4"
            elif "efficientnet" in backbone_name:
                target_layer = "backbone.conv_head"
            elif "convnext" in backbone_name:
                target_layer = "backbone.stages.3"
            else:
                target_layer = None

            if target_layer:
                explainer = GradCAM(model, target_layer=target_layer)
                heatmap = explainer(image, target_class=0)

                # Save visualization
                save_path = (
                    f"outputs/hf_comparison_{arch_name.lower().replace('-', '_')}.png"
                )
                save_heatmap(heatmap, image, save_path=save_path, method="gradcam")

                heatmap_range = f"[{heatmap.min():.3f}, {heatmap.max():.3f}]"
                print(f"{arch_name:<20} {n_params:>12,} {heatmap_range:>20}")

        except Exception as e:
            print(f"{arch_name:<20} Error: {str(e)[:40]}")

    print("\n✓ Saved all comparison visualizations to outputs/")


def example_5_quantitative_metrics():
    """Example 5: Quantitative evaluation of explanations."""
    print("\n" + "=" * 80)
    print("Example 5: Quantitative Explanation Metrics")
    print("=" * 80)

    # Load model
    model = ImageClassifier(
        backbone="hf-hub:timm/resnet18.a1_in1k",
        num_classes=10,
    )
    model.eval()

    print("✓ Loaded ResNet-18 from HuggingFace Hub")

    # Create test image
    image = create_sample_image()

    # Create explainer
    explainer = GradCAM(model, target_layer="backbone.layer4")
    heatmap = explainer(image, target_class=0)

    # Create metrics evaluator
    metrics = ExplanationMetrics(model)
    print("✓ Created metrics evaluator")

    # Evaluate different metrics
    print("\nEvaluating explanation quality:")

    try:
        # Insertion/Deletion metrics
        insertion_score = metrics.insertion(
            image, heatmap, target_class=0, num_steps=10
        )
        deletion_score = metrics.deletion(image, heatmap, target_class=0, num_steps=10)

        print(f"  • Insertion AUC:  {insertion_score:.4f} (higher is better)")
        print(f"  • Deletion AUC:   {deletion_score:.4f} (lower is better)")

        # Sensitivity metrics
        sensitivity = metrics.sensitivity(image, heatmap, target_class=0, n_samples=5)
        print(f"  • Sensitivity:    {sensitivity:.4f} (higher is better)")

    except Exception as e:
        print(f"  ⚠ Could not compute all metrics: {e}")

    print("\n✓ Completed quantitative evaluation")


def example_6_integrated_gradients():
    """Example 6: Integrated Gradients with HF models."""
    print("\n" + "=" * 80)
    print("Example 6: Integrated Gradients")
    print("=" * 80)

    # Load model
    model = ImageClassifier(
        backbone="hf-hub:timm/resnet18.a1_in1k",
        num_classes=10,
    )
    model.eval()

    print("✓ Loaded ResNet-18 from HuggingFace Hub")

    # Create test image
    image = create_sample_image()

    # Create Integrated Gradients explainer
    explainer = IntegratedGradients(model)
    print("✓ Created Integrated Gradients explainer")

    # Generate attribution
    attribution = explainer(image, target_class=0, n_steps=50)
    print(f"✓ Computed attributions: shape={attribution.shape}")

    # Convert to heatmap (average across channels)
    heatmap = np.abs(attribution).mean(axis=2)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Save visualization
    save_heatmap(
        heatmap,
        image,
        save_path="outputs/hf_integrated_gradients.png",
        method="integrated_gradients",
    )
    print(
        "✓ Saved Integrated Gradients visualization to: outputs/hf_integrated_gradients.png"
    )


def example_7_interactive_visualization():
    """Example 7: Interactive visualization with Plotly."""
    if not INTERACTIVE_AVAILABLE:
        print("\n⚠ Skipping interactive visualization (plotly not installed)")
        return

    print("\n" + "=" * 80)
    print("Example 7: Interactive Visualization")
    print("=" * 80)

    # Load model
    model = ImageClassifier(
        backbone="hf-hub:timm/resnet18.a1_in1k",
        num_classes=10,
    )
    model.eval()

    print("✓ Loaded ResNet-18 from HuggingFace Hub")

    # Create test image
    image = create_sample_image()

    # Create interactive visualizer
    viz = InteractiveVisualizer(model)
    print("✓ Created interactive visualizer")

    # Create explainer
    explainer = GradCAM(model, target_layer="backbone.layer4")

    # Generate interactive visualization
    viz.visualize_explanation(
        image,
        explainer,
        target_class=0,
        save_path="outputs/hf_interactive.html",
    )
    print("✓ Saved interactive visualization to: outputs/hf_interactive.html")
    print("  Open this file in a web browser to explore interactively!")


def main():
    """Run all examples."""
    # Create output directory
    Path("outputs").mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("Model Interpretation with HuggingFace Hub Models")
    print("=" * 80)
    print("\nThis example demonstrates various interpretation techniques")
    print("for models loaded from HuggingFace Hub.\n")

    # Discover available models
    discover_interpretable_hf_models()

    # Run examples
    try:
        example_1_gradcam_with_hf_resnet()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        example_2_compare_gradcam_variants()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        example_3_attention_visualization_vit()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    try:
        example_4_multi_architecture_comparison()
    except Exception as e:
        print(f"Example 4 failed: {e}")

    try:
        example_5_quantitative_metrics()
    except Exception as e:
        print(f"Example 5 failed: {e}")

    try:
        example_6_integrated_gradients()
    except Exception as e:
        print(f"Example 6 failed: {e}")

    try:
        example_7_interactive_visualization()
    except Exception as e:
        print(f"Example 7 failed: {e}")

    print("\n" + "=" * 80)
    print("All Examples Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("• HuggingFace Hub models work seamlessly with AutoTimm interpretation")
    print("• GradCAM works well with CNNs (ResNet, EfficientNet, ConvNeXt)")
    print("• Attention visualization is ideal for Vision Transformers")
    print("• Integrated Gradients provides pixel-level attributions")
    print("• Quantitative metrics help evaluate explanation quality")
    print("• Interactive visualizations enable deeper exploration")
    print("\nNext steps:")
    print("• Try with your own images")
    print("• Experiment with different target layers")
    print("• Compare explanations across model families")
    print("• Use metrics to select the best explanation method")


if __name__ == "__main__":
    main()
