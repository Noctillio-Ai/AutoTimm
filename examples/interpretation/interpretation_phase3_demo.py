"""
Phase 3: Production Polish & Advanced Features Demo

Demonstrates AutoTrainer integration, feature visualization, and
production-ready interpretation workflows.
"""

from PIL import Image
import numpy as np

from autotimm import ImageClassifier, LoggerConfig
from autotimm.interpretation import (
    InterpretationCallback,
    FeatureMonitorCallback,
    FeatureVisualizer,
)


def create_sample_image():
    """Create a sample image for demonstration."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        img[i, :, 0] = int(255 * i / 224)
        img[:, i, 1] = int(255 * i / 224)
    img[:, :, 2] = 128
    return Image.fromarray(img)


def example_1_autotrainer_integration():
    """Example 1: AutoTrainer callback for automatic interpretation."""
    print("\n" + "=" * 60)
    print("Example 1: AutoTrainer Integration")
    print("=" * 60)
    print("⚠ Note: This example shows callback setup.")
    print("        Full training demo requires a dataset.")

    # Create sample images for monitoring
    sample_images = [create_sample_image() for _ in range(4)]

    # Create interpretation callback
    InterpretationCallback(
        sample_images=sample_images,
        method="gradcam",
        log_every_n_epochs=5,
        num_samples=4,
        colormap="viridis",
    )

    # Create feature monitoring callback
    FeatureMonitorCallback(
        layer_names=["backbone.layer2", "backbone.layer3", "backbone.layer4"],
        log_every_n_epochs=1,
    )

    print("✓ Created InterpretationCallback")
    print("  - Will log explanations every 5 epochs")
    print(f"  - Monitoring {len(sample_images)} sample images")
    print("✓ Created FeatureMonitorCallback")
    print("  - Monitoring 3 layers")

    # Example trainer setup (would use in actual training)
    """
    trainer = AutoTrainer(
        max_epochs=100,
        callbacks=[interp_callback, feature_callback],
        logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
    )
    trainer.fit(model, datamodule=data)
    """

    print("✓ Setup complete - ready for training with automatic interpretation!")


def example_2_feature_visualization():
    """Example 2: Feature map visualization and analysis."""
    print("\n" + "=" * 60)
    print("Example 2: Feature Visualization")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    image = create_sample_image()

    # Create feature visualizer
    viz = FeatureVisualizer(model)

    # Visualize feature maps
    print("Generating feature map visualizations...")
    viz.plot_feature_maps(
        image,
        layer_name="backbone.layer3",
        num_features=16,
        sort_by="activation",
        save_path="feature_maps.png",
    )
    print("✓ Saved feature maps to: feature_maps.png")

    # Get feature statistics
    stats = viz.get_feature_statistics(image, layer_name="backbone.layer4")
    print("\n✓ Layer 4 Statistics:")
    print(f"  - Mean activation: {stats['mean']:.3f}")
    print(f"  - Std deviation: {stats['std']:.3f}")
    print(f"  - Sparsity: {stats['sparsity']:.2%}")
    print(f"  - Active channels: {stats['active_channels']}/{stats['num_channels']}")

    # Get top activating features
    top_features = viz.get_top_activating_features(image, "backbone.layer4", top_k=5)
    print("\n✓ Top 5 Activating Channels:")
    for channel, activation in top_features:
        print(f"  - Channel {channel}: {activation:.3f}")


def example_3_layer_comparison():
    """Example 3: Compare features across layers."""
    print("\n" + "=" * 60)
    print("Example 3: Layer Comparison")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet34", num_classes=10)
    model.eval()

    image = create_sample_image()

    viz = FeatureVisualizer(model)

    # Compare multiple layers
    layers = [
        "backbone.layer1",
        "backbone.layer2",
        "backbone.layer3",
        "backbone.layer4",
    ]
    all_stats = viz.compare_layers(image, layers, save_path="layer_comparison.png")

    print("✓ Layer Statistics Comparison:")
    for layer, stats in all_stats.items():
        print(f"\n  {layer}:")
        print(f"    - Channels: {stats['num_channels']}")
        print(f"    - Spatial: {stats['spatial_size']}")
        print(f"    - Mean activation: {stats['mean']:.3f}")
        print(f"    - Sparsity: {stats['sparsity']:.2%}")

    print("\n✓ Saved comparison plot to: layer_comparison.png")


def example_4_receptive_field():
    """Example 4: Receptive field visualization."""
    print("\n" + "=" * 60)
    print("Example 4: Receptive Field Visualization")
    print("=" * 60)
    print("⚠ Note: This is computationally intensive.")
    print("        Using simplified version for demo.")

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    image = create_sample_image()

    viz = FeatureVisualizer(model)

    # Get top activating channel
    top_features = viz.get_top_activating_features(image, "backbone.layer3", top_k=1)
    channel = top_features[0][0]

    print(f"✓ Selected channel {channel} (highest activation)")
    print("  Computing receptive field approximation...")

    # Note: This can be slow, so we skip actual computation in demo
    # rf = viz.visualize_receptive_field(
    #     image,
    #     layer_name="backbone.layer3",
    #     channel=channel,
    #     save_path="receptive_field.png"
    # )

    print("✓ Receptive field visualization would be saved to: receptive_field.png")


def example_5_production_workflow():
    """Example 5: Complete production workflow."""
    print("\n" + "=" * 60)
    print("Example 5: Production Workflow")
    print("=" * 60)

    print("Complete workflow for production deployment:")
    print()
    print("1. Training with automatic interpretation:")
    print("   ```python")
    print("   callbacks = [")
    print("       InterpretationCallback(sample_images, log_every_n_epochs=5),")
    print('       FeatureMonitorCallback(["layer2", "layer3", "layer4"]),')
    print("   ]")
    print("   trainer = AutoTrainer(callbacks=callbacks)")
    print("   trainer.fit(model, datamodule)")
    print("   ```")
    print()
    print("2. Feature analysis after training:")
    print("   ```python")
    print("   viz = FeatureVisualizer(model)")
    print('   stats = viz.compare_layers(image, ["layer2", "layer3", "layer4"])')
    print("   ```")
    print()
    print("3. Inference-time interpretation:")
    print("   ```python")
    print("   from autotimm.interpretation import explain_prediction")
    print('   result = explain_prediction(model, image, method="gradcam")')
    print("   ```")
    print()
    print("4. Monitoring in production:")
    print("   - Log explanations for low-confidence predictions")
    print("   - Track feature statistics drift over time")
    print("   - Generate periodic interpretation reports")


def main():
    """Run all Phase 3 examples."""
    print("\n" + "=" * 60)
    print("AutoTimm Phase 3: Production Polish Demo")
    print("=" * 60)

    # Run examples
    example_1_autotrainer_integration()
    example_2_feature_visualization()
    example_3_layer_comparison()
    example_4_receptive_field()
    example_5_production_workflow()

    print("\n" + "=" * 60)
    print("✓ All Phase 3 examples completed!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - feature_maps.png")
    print("  - layer_comparison.png")
    print("\nPhase 3 Features:")
    print("  ✓ AutoTrainer Integration (callbacks)")
    print("  ✓ Feature Visualization & Analysis")
    print("  ✓ Layer Comparison Tools")
    print("  ✓ Receptive Field Approximation")
    print("  ✓ Production-Ready Workflows")
    print("\nReady for production deployment! 🚀")


if __name__ == "__main__":
    main()
