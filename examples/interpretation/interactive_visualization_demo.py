"""
Interactive Visualization Demo

Demonstrates how to create interactive, explorable visualizations of model
interpretations using Plotly. Includes:
- Interactive heatmap overlays
- Method comparisons
- HTML report generation
"""

import numpy as np
from PIL import Image

from autotimm import ImageClassifier
from autotimm.interpretation import (
    GradCAM,
    GradCAMPlusPlus,
    IntegratedGradients,
    InteractiveVisualizer,
)


def create_sample_image():
    """Create a sample image for demonstration."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Create interesting pattern
    for i in range(224):
        img[i, :, 0] = int(255 * i / 224)
        img[:, i, 1] = int(255 * (1 - i / 224))
    # Add circular pattern
    center = 112
    for i in range(224):
        for j in range(224):
            dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            if dist < 50:
                img[i, j, 2] = int(255 * (1 - dist / 50))
    return Image.fromarray(img)


def example_1_basic_visualization():
    """Example 1: Basic interactive visualization."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Interactive Visualization")
    print("=" * 60)

    # Create model and explainer
    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    explainer = GradCAM(model)
    viz = InteractiveVisualizer(model, use_cuda=False)

    # Create sample image
    image = create_sample_image()

    print("\nGenerating interactive visualization...")
    viz.visualize_explanation(
        image,
        explainer,
        title="GradCAM Interactive Explanation",
        colorscale="Viridis",
        opacity=0.6,
        save_path="interactive_gradcam.html",
    )

    print("✓ Saved interactive visualization to: interactive_gradcam.html")
    print("  Open in your browser to explore:")
    print("  - Hover over heatmap to see importance values")
    print("  - Zoom and pan to explore details")
    print("  - Interactive colorbar")


def example_2_compare_methods():
    """Example 2: Compare multiple methods interactively."""
    print("\n" + "=" * 60)
    print("Example 2: Interactive Method Comparison")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    # Create multiple explainers
    explainers = {
        "GradCAM": GradCAM(model),
        "GradCAM++": GradCAMPlusPlus(model),
        "IntegratedGradients": IntegratedGradients(model, baseline="black", steps=30),
    }

    viz = InteractiveVisualizer(model, use_cuda=False)
    image = create_sample_image()

    print("\nComparing 3 explanation methods...")
    viz.compare_methods(
        image,
        explainers,
        title="Method Comparison",
        colorscale="Viridis",
        opacity=0.6,
        save_path="interactive_comparison.html",
        width=1400,
        height=500,
    )

    print("✓ Saved comparison to: interactive_comparison.html")
    print("  Open in browser to:")
    print("  - Compare methods side-by-side")
    print("  - Hover to see method-specific importance")
    print("  - Zoom all views synchronously")


def example_3_different_colorscales():
    """Example 3: Try different colorscales."""
    print("\n" + "=" * 60)
    print("Example 3: Different Colorscales")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    explainer = GradCAM(model)
    viz = InteractiveVisualizer(model, use_cuda=False)
    image = create_sample_image()

    colorscales = {
        "Viridis": "viridis_example.html",
        "Hot": "hot_example.html",
        "Jet": "jet_example.html",
        "Plasma": "plasma_example.html",
    }

    print("\nGenerating visualizations with different colorscales...")
    for colorscale, filename in colorscales.items():
        viz.visualize_explanation(
            image,
            explainer,
            title=f"GradCAM with {colorscale} colorscale",
            colorscale=colorscale,
            save_path=filename,
        )
        print(f"✓ Saved {colorscale} colorscale to: {filename}")

    print("\nColorscales available:")
    print("  - Viridis: Good default, perceptually uniform")
    print("  - Hot: Traditional heatmap look")
    print("  - Jet: High contrast, easy to see differences")
    print("  - Plasma: Alternative perceptually uniform")


def example_4_html_report():
    """Example 4: Generate comprehensive HTML report."""
    print("\n" + "=" * 60)
    print("Example 4: Comprehensive HTML Report")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    explainer = GradCAM(model)
    viz = InteractiveVisualizer(model, use_cuda=False)
    image = create_sample_image()

    print("\nGenerating comprehensive HTML report...")
    report_path = viz.create_report(
        image,
        explainer,
        include_statistics=True,
        include_top_regions=True,
        save_path="comprehensive_report.html",
        title="Model Interpretation Report",
    )

    print(f"✓ Saved comprehensive report to: {report_path}")
    print("  Report includes:")
    print("  - Prediction information and top-5 classes")
    print("  - Heatmap statistics (mean, std, sparsity)")
    print("  - Interactive explanation visualization")
    print("  - Importance distribution histogram")
    print("  - Professional styling")


def example_5_custom_opacity():
    """Example 5: Adjust opacity for better visibility."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Opacity Settings")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    explainer = GradCAM(model)
    viz = InteractiveVisualizer(model, use_cuda=False)
    image = create_sample_image()

    opacities = [0.3, 0.5, 0.7, 0.9]

    print("\nGenerating visualizations with different opacities...")
    for opacity in opacities:
        filename = f"opacity_{int(opacity*10)}.html"
        viz.visualize_explanation(
            image,
            explainer,
            title=f"GradCAM with opacity {opacity}",
            opacity=opacity,
            save_path=filename,
        )
        print(f"✓ Opacity {opacity}: saved to {filename}")

    print("\nOpacity guidelines:")
    print("  - 0.3-0.4: See original image clearly")
    print("  - 0.5-0.6: Balanced (recommended)")
    print("  - 0.7-0.8: Emphasize heatmap")
    print("  - 0.9+: Focus on heatmap only")


def example_6_production_workflow():
    """Example 6: Production deployment workflow."""
    print("\n" + "=" * 60)
    print("Example 6: Production Workflow")
    print("=" * 60)

    print("\nProduction workflow for interactive explanations:")
    print()
    print("1. Setup:")
    print("   ```python")
    print("   from autotimm.interpretation import GradCAM, InteractiveVisualizer")
    print("   ")
    print("   model = load_production_model()")
    print("   explainer = GradCAM(model)")
    print("   viz = InteractiveVisualizer(model)")
    print("   ```")
    print()
    print("2. Generate explanations on demand:")
    print("   ```python")
    print("   # For user requests")
    print("   fig = viz.visualize_explanation(")
    print("       user_image,")
    print("       explainer,")
    print("       save_path=f'explanations/{user_id}.html'")
    print("   )")
    print("   ```")
    print()
    print("3. Batch processing:")
    print("   ```python")
    print("   # Generate reports for dataset")
    print("   for idx, image in enumerate(validation_set):")
    print("       viz.create_report(")
    print("           image,")
    print("           explainer,")
    print("           save_path=f'reports/sample_{idx}.html'")
    print("       )")
    print("   ```")
    print()
    print("4. Deploy:")
    print("   - Serve HTML files via web server")
    print("   - Or embed in existing dashboard")
    print("   - Users can explore without additional tools")


def main():
    """Run all interactive visualization examples."""
    print("\n" + "=" * 60)
    print("Interactive Visualization Demo")
    print("=" * 60)
    print("\nThis demo requires plotly. Install with:")
    print("  pip install plotly")
    print("  or")
    print("  pip install autotimm[interactive]")
    print()

    try:
        # Run examples
        example_1_basic_visualization()
        example_2_compare_methods()
        example_3_different_colorscales()
        example_4_html_report()
        example_5_custom_opacity()
        example_6_production_workflow()

        print("\n" + "=" * 60)
        print("✓ All examples completed!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - interactive_gradcam.html")
        print("  - interactive_comparison.html")
        print("  - viridis_example.html, hot_example.html, etc.")
        print("  - comprehensive_report.html")
        print("  - opacity_*.html (various opacities)")
        print("\nFeatures demonstrated:")
        print("  ✓ Interactive heatmap overlays")
        print("  ✓ Method comparisons")
        print("  ✓ Custom colorscales")
        print("  ✓ Comprehensive HTML reports")
        print("  ✓ Opacity adjustments")
        print("\nAdvantages over static images:")
        print("  • Zoom and pan to explore details")
        print("  • Hover to see exact values")
        print("  • No additional tools needed (just browser)")
        print("  • Professional presentation")
        print("  • Easy to share")

    except ImportError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlotly is required for interactive visualizations.")
        print("Install with: pip install plotly")
        print("Or: pip install autotimm[interactive]")


if __name__ == "__main__":
    main()
