"""Tests for model interpretation functionality."""

from typing import ClassVar

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from autotimm.interpretation import (
    FeatureMonitorCallback,
    FeatureVisualizer,
    GradCAM,
    GradCAMPlusPlus,
    InterpretationCallback,
    compare_methods,
    explain_prediction,
    visualize_batch,
)
from autotimm.interpretation.visualization.heatmap import (
    apply_colormap,
    create_comparison_figure,
    overlay_heatmap,
)


# Test fixtures
@pytest.fixture
def simple_cnn():
    """Create a simple CNN for testing."""

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = SimpleCNN()
    model.eval()
    return model


@pytest.fixture
def test_image():
    """Create a test image."""
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


@pytest.fixture
def test_image_tensor():
    """Create a test image tensor."""
    return torch.randn(1, 3, 224, 224)


# Base Interpreter Tests
class TestBaseInterpreter:
    def test_layer_resolution_by_name(self, simple_cnn):
        """Test resolving layer by name."""
        explainer = GradCAM(simple_cnn, target_layer="conv3")
        assert explainer.target_layer is simple_cnn.conv3

    def test_layer_auto_detection(self, simple_cnn):
        """Test automatic layer detection."""
        explainer = GradCAM(simple_cnn, target_layer=None)
        # Should detect conv3 as last conv layer
        assert isinstance(explainer.target_layer, nn.Conv2d)
        assert explainer.target_layer is simple_cnn.conv3

    def test_layer_resolution_by_module(self, simple_cnn):
        """Test resolving layer by module reference."""
        explainer = GradCAM(simple_cnn, target_layer=simple_cnn.conv2)
        assert explainer.target_layer is simple_cnn.conv2

    def test_invalid_layer_name(self, simple_cnn):
        """Test error on invalid layer name."""
        with pytest.raises(AttributeError):
            GradCAM(simple_cnn, target_layer="invalid_layer")

    def test_device_selection_cpu(self, simple_cnn):
        """Test CPU device selection."""
        explainer = GradCAM(simple_cnn, use_cuda=False)
        assert explainer.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_selection_cuda(self, simple_cnn):
        """Test CUDA device selection."""
        explainer = GradCAM(simple_cnn, use_cuda=True)
        assert explainer.device.type == "cuda"


# GradCAM Tests
class TestGradCAM:
    def test_gradcam_basic(self, simple_cnn, test_image):
        """Test basic GradCAM functionality."""
        explainer = GradCAM(simple_cnn)
        heatmap = explainer(test_image)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.ndim == 2
        assert heatmap.min() >= 0
        assert heatmap.max() <= 1

    def test_gradcam_with_target_class(self, simple_cnn, test_image):
        """Test GradCAM with specific target class."""
        explainer = GradCAM(simple_cnn)
        heatmap = explainer(test_image, target_class=5)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (224, 224)

    def test_gradcam_tensor_input(self, simple_cnn, test_image_tensor):
        """Test GradCAM with tensor input."""
        explainer = GradCAM(simple_cnn)
        heatmap = explainer(test_image_tensor, target_class=3)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.ndim == 2

    def test_gradcam_numpy_input(self, simple_cnn):
        """Test GradCAM with numpy array input."""
        image = np.random.rand(224, 224, 3).astype(np.float32)
        explainer = GradCAM(simple_cnn)
        heatmap = explainer(image)

        assert isinstance(heatmap, np.ndarray)

    def test_gradcam_batch(self, simple_cnn, test_image):
        """Test GradCAM batch processing."""
        explainer = GradCAM(simple_cnn)
        images = [test_image] * 3
        heatmaps = explainer.explain_batch(images)

        assert len(heatmaps) == 3
        assert all(isinstance(h, np.ndarray) for h in heatmaps)

    def test_gradcam_change_target_layer(self, simple_cnn):
        """Test changing target layer dynamically."""
        # Use a non-uniform image so different layers produce different heatmaps
        torch.manual_seed(42)
        image_tensor = torch.randn(1, 3, 224, 224)

        explainer = GradCAM(simple_cnn, target_layer="conv3")
        heatmap1 = explainer(image_tensor)

        explainer.set_target_layer("conv2")
        heatmap2 = explainer(image_tensor)

        assert explainer.target_layer is simple_cnn.conv2
        # Heatmaps should be different
        assert not np.allclose(heatmap1, heatmap2)


# GradCAM++ Tests
class TestGradCAMPlusPlus:
    def test_gradcampp_basic(self, simple_cnn, test_image):
        """Test basic GradCAM++ functionality."""
        explainer = GradCAMPlusPlus(simple_cnn)
        heatmap = explainer(test_image)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.ndim == 2
        assert heatmap.min() >= 0
        assert heatmap.max() <= 1

    def test_gradcampp_vs_gradcam(self, simple_cnn, test_image):
        """Test that GradCAM++ produces different results from GradCAM."""
        gradcam = GradCAM(simple_cnn)
        gradcampp = GradCAMPlusPlus(simple_cnn)

        heatmap_cam = gradcam(test_image, target_class=0)
        heatmap_campp = gradcampp(test_image, target_class=0)

        # Results should be different (but both valid)
        assert not np.allclose(heatmap_cam, heatmap_campp)
        assert heatmap_campp.min() >= 0
        assert heatmap_campp.max() <= 1


# Visualization Tests
class TestVisualization:
    def test_apply_colormap(self):
        """Test colormap application."""
        heatmap = np.random.rand(224, 224)
        colored = apply_colormap(heatmap, colormap="viridis")

        assert colored.shape == (224, 224, 3)
        assert colored.dtype == np.uint8
        assert colored.max() <= 255

    def test_apply_colormap_different_colormaps(self):
        """Test different colormaps."""
        heatmap = np.random.rand(100, 100)

        for cmap in ["viridis", "jet", "plasma", "hot"]:
            colored = apply_colormap(heatmap, colormap=cmap)
            assert colored.shape == (100, 100, 3)

    def test_overlay_heatmap_pil(self, test_image):
        """Test heatmap overlay with PIL Image."""
        heatmap = np.random.rand(224, 224)
        overlayed = overlay_heatmap(test_image, heatmap, alpha=0.4)

        assert overlayed.shape == (224, 224, 3)
        assert overlayed.dtype == np.uint8

    def test_overlay_heatmap_numpy(self):
        """Test heatmap overlay with numpy array."""
        image = np.random.rand(224, 224, 3)
        heatmap = np.random.rand(224, 224)
        overlayed = overlay_heatmap(image, heatmap, alpha=0.5)

        assert overlayed.shape == (224, 224, 3)

    def test_overlay_heatmap_resize(self):
        """Test automatic heatmap resizing."""
        image = np.random.rand(512, 512, 3)
        heatmap = np.random.rand(256, 256)  # Different size
        overlayed = overlay_heatmap(image, heatmap, alpha=0.4, resize_heatmap=True)

        assert overlayed.shape[:2] == image.shape[:2]

    def test_create_comparison_figure(self, test_image):
        """Test comparison figure creation."""
        heatmaps = [np.random.rand(224, 224) for _ in range(3)]
        titles = ["Method 1", "Method 2", "Method 3"]

        fig = create_comparison_figure(
            test_image, heatmaps, titles, layout="horizontal"
        )

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_create_comparison_figure_layouts(self, test_image):
        """Test different layout options."""
        heatmaps = [np.random.rand(224, 224) for _ in range(2)]
        titles = ["Method 1", "Method 2"]

        for layout in ["grid", "horizontal", "vertical"]:
            fig = create_comparison_figure(test_image, heatmaps, titles, layout=layout)
            assert fig is not None
            import matplotlib.pyplot as plt

            plt.close(fig)


# High-Level API Tests
class TestHighLevelAPI:
    def test_explain_prediction_basic(self, simple_cnn, test_image, tmp_path):
        """Test basic explain_prediction functionality."""
        save_path = tmp_path / "explanation.png"
        result = explain_prediction(
            simple_cnn,
            test_image,
            method="gradcam",
            save_path=save_path,
        )

        assert "predicted_class" in result
        assert "target_class" in result
        assert "method" in result
        assert "target_layer" in result
        assert save_path.exists()

    def test_explain_prediction_with_path(self, simple_cnn, test_image, tmp_path):
        """Test explain_prediction with image path."""
        image_path = tmp_path / "test_image.png"
        test_image.save(image_path)

        result = explain_prediction(
            simple_cnn,
            image_path,
            method="gradcam",
        )

        assert "predicted_class" in result

    def test_explain_prediction_return_heatmap(self, simple_cnn, test_image):
        """Test returning raw heatmap."""
        result = explain_prediction(
            simple_cnn,
            test_image,
            method="gradcam",
            return_heatmap=True,
        )

        assert "heatmap" in result
        assert isinstance(result["heatmap"], np.ndarray)

    def test_explain_prediction_gradcampp(self, simple_cnn, test_image):
        """Test explain_prediction with GradCAM++."""
        result = explain_prediction(
            simple_cnn,
            test_image,
            method="gradcam++",
        )

        assert result["method"] == "gradcam++"

    def test_compare_methods(self, simple_cnn, test_image, tmp_path):
        """Test method comparison."""
        save_path = tmp_path / "comparison.png"
        results = compare_methods(
            simple_cnn,
            test_image,
            methods=["gradcam", "gradcam++"],
            save_path=save_path,
        )

        assert "gradcam" in results
        assert "gradcam++" in results
        assert save_path.exists()

    def test_visualize_batch(self, simple_cnn, test_image, tmp_path):
        """Test batch visualization."""
        # Create test images
        images = [test_image] * 3

        results = visualize_batch(
            simple_cnn,
            images,
            method="gradcam",
            output_dir=tmp_path,
        )

        assert len(results) == 3
        assert all("predicted_class" in r for r in results)
        # Check that files were saved
        saved_files = list(tmp_path.glob("*.png"))
        assert len(saved_files) == 3


# Integration Tests
class TestIntegration:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gradcam_cuda(self, simple_cnn, test_image):
        """Test GradCAM on CUDA."""
        explainer = GradCAM(simple_cnn, use_cuda=True)
        heatmap = explainer(test_image)

        assert isinstance(heatmap, np.ndarray)
        assert explainer.device.type == "cuda"

    def test_gradcam_with_dict_output(self, test_image):
        """Test GradCAM with model that returns dict."""

        class DictOutputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.fc = nn.Linear(16, 10)

            def forward(self, x):
                x = self.conv(x)
                x = torch.relu(x)
                x = F.adaptive_avg_pool2d(x, 1)
                x = x.view(x.size(0), -1)
                logits = self.fc(x)
                return {"logits": logits}

        model = DictOutputModel()
        model.eval()

        explainer = GradCAM(model)
        heatmap = explainer(test_image)

        assert isinstance(heatmap, np.ndarray)


# Phase 3: Feature Visualization Tests
class TestFeatureVisualizer:
    def test_feature_visualizer_init(self, simple_cnn):
        """Test FeatureVisualizer initialization."""
        viz = FeatureVisualizer(simple_cnn, use_cuda=False)
        assert viz.model is simple_cnn
        assert viz.device.type == "cpu"

    def test_get_features(self, simple_cnn, test_image):
        """Test feature extraction from a layer."""
        viz = FeatureVisualizer(simple_cnn, use_cuda=False)
        features = viz.get_features(test_image, layer_name="conv2")

        assert isinstance(features, torch.Tensor)
        assert features.ndim == 4  # (B, C, H, W)
        assert features.shape[0] == 1  # Batch size

    def test_get_feature_statistics(self, simple_cnn, test_image):
        """Test feature statistics computation."""
        viz = FeatureVisualizer(simple_cnn, use_cuda=False)
        stats = viz.get_feature_statistics(test_image, layer_name="conv3")

        assert "mean" in stats
        assert "std" in stats
        assert "sparsity" in stats
        assert "max" in stats
        assert "min" in stats
        assert "active_channels" in stats
        assert "num_channels" in stats
        assert "spatial_size" in stats

        # Check value ranges
        assert stats["sparsity"] >= 0 and stats["sparsity"] <= 1
        assert stats["num_channels"] == 64  # conv3 has 64 channels

    def test_plot_feature_maps(self, simple_cnn, test_image, tmp_path):
        """Test feature map plotting."""
        viz = FeatureVisualizer(simple_cnn, use_cuda=False)
        save_path = tmp_path / "features.png"

        fig = viz.plot_feature_maps(
            test_image,
            layer_name="conv2",
            num_features=16,
            sort_by="activation",
            save_path=str(save_path),
        )

        assert fig is not None
        assert save_path.exists()

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_feature_maps_sort_methods(self, simple_cnn, test_image):
        """Test different sorting methods for feature selection."""
        viz = FeatureVisualizer(simple_cnn, use_cuda=False)

        for sort_by in ["activation", "variance", "random"]:
            fig = viz.plot_feature_maps(
                test_image,
                layer_name="conv2",
                num_features=8,
                sort_by=sort_by,
            )
            assert fig is not None

            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_compare_layers(self, simple_cnn, test_image, tmp_path):
        """Test layer comparison."""
        viz = FeatureVisualizer(simple_cnn, use_cuda=False)
        save_path = tmp_path / "layer_comparison.png"

        layer_names = ["conv1", "conv2", "conv3"]
        all_stats = viz.compare_layers(
            test_image,
            layer_names,
            save_path=str(save_path),
        )

        assert len(all_stats) == 3
        for layer_name in layer_names:
            assert layer_name in all_stats
            assert "mean" in all_stats[layer_name]
            assert "std" in all_stats[layer_name]

        assert save_path.exists()

    def test_get_top_activating_features(self, simple_cnn, test_image):
        """Test getting top activating channels."""
        viz = FeatureVisualizer(simple_cnn, use_cuda=False)
        top_features = viz.get_top_activating_features(
            test_image,
            layer_name="conv3",
            top_k=5,
        )

        assert len(top_features) == 5
        for channel_idx, activation in top_features:
            assert isinstance(channel_idx, int)
            assert isinstance(activation, float)
            assert channel_idx >= 0

        # Check that it's sorted in descending order
        activations = [act for _, act in top_features]
        assert activations == sorted(activations, reverse=True)

    def test_visualize_receptive_field(self, simple_cnn, test_image, tmp_path):
        """Test receptive field visualization."""
        viz = FeatureVisualizer(simple_cnn, use_cuda=False)
        save_path = tmp_path / "receptive_field.png"

        # This is slow, so we'll just test the basic functionality
        # Use smaller image for faster test
        small_image = test_image.resize((56, 56))

        sensitivity = viz.visualize_receptive_field(
            small_image,
            layer_name="conv2",
            channel=0,
            position=None,
            save_path=str(save_path),
        )

        assert isinstance(sensitivity, np.ndarray)
        assert sensitivity.ndim == 2
        assert sensitivity.min() >= 0 and sensitivity.max() <= 1
        assert save_path.exists()

    def test_invalid_layer_name(self, simple_cnn, test_image):
        """Test error handling for invalid layer name."""
        viz = FeatureVisualizer(simple_cnn, use_cuda=False)

        with pytest.raises(ValueError, match="Layer .* not found"):
            viz.get_features(test_image, layer_name="invalid_layer")


# Phase 3: Callback Tests
class TestInterpretationCallback:
    def test_callback_initialization_with_pil_images(self, test_image):
        """Test callback initialization with PIL images."""
        sample_images = [test_image] * 4
        callback = InterpretationCallback(
            sample_images=sample_images,
            method="gradcam",
            log_every_n_epochs=5,
        )

        assert len(callback.sample_images) == 4
        assert callback.method == "gradcam"
        assert callback.log_every_n_epochs == 5

    def test_callback_initialization_with_tensors(self):
        """Test callback initialization with tensor images."""
        sample_images = [torch.randn(3, 224, 224) for _ in range(3)]
        callback = InterpretationCallback(
            sample_images=sample_images,
            method="gradcam++",
        )

        assert len(callback.sample_images) == 3
        assert all(isinstance(img, torch.Tensor) for img in callback.sample_images)

    def test_callback_sampling(self, test_image):
        """Test that callback samples images when too many provided."""
        sample_images = [test_image] * 20
        callback = InterpretationCallback(
            sample_images=sample_images,
            num_samples=8,
        )

        # Should sample down to num_samples
        assert len(callback.sample_images) == 8

    def test_callback_on_train_start(self, simple_cnn, test_image):
        """Test callback explainer initialization on training start."""
        callback = InterpretationCallback(
            sample_images=[test_image],
            method="gradcam",
        )

        # Mock trainer and pl_module
        class MockTrainer:
            current_epoch = 0
            global_step = 0
            loggers: ClassVar[list] = []

        trainer = MockTrainer()

        callback.on_train_start(trainer, simple_cnn)

        assert callback.explainer is not None
        assert isinstance(callback.explainer, GradCAM)

    def test_callback_methods(self, test_image):
        """Test different interpretation methods."""
        for method in ["gradcam", "gradcam++", "integrated_gradients"]:
            callback = InterpretationCallback(
                sample_images=[test_image],
                method=method,
            )
            assert callback.method == method


class TestFeatureMonitorCallback:
    def test_callback_initialization(self):
        """Test FeatureMonitorCallback initialization."""
        layer_names = ["conv1", "conv2", "conv3"]
        callback = FeatureMonitorCallback(
            layer_names=layer_names,
            log_every_n_epochs=2,
            num_batches=5,
        )

        assert callback.layer_names == layer_names
        assert callback.log_every_n_epochs == 2
        assert callback.num_batches == 5
        assert len(callback.hooks) == 0
        assert len(callback.activations) == 3

    def test_get_layer_by_name(self, simple_cnn):
        """Test layer resolution by name."""
        callback = FeatureMonitorCallback(
            layer_names=["conv1", "conv2"],
        )

        # Test valid layer
        layer = callback._get_layer_by_name(simple_cnn, "conv1")
        assert layer is simple_cnn.conv1

        # Test invalid layer
        layer = callback._get_layer_by_name(simple_cnn, "invalid")
        assert layer is None

    def test_hook_registration_and_removal(self, simple_cnn):
        """Test hook registration and removal."""
        callback = FeatureMonitorCallback(
            layer_names=["conv1", "conv2"],
        )

        # Register hooks
        callback._register_hooks(simple_cnn)
        assert len(callback.hooks) == 2

        # Remove hooks
        callback._remove_hooks()
        assert len(callback.hooks) == 0
        assert all(len(acts) == 0 for acts in callback.activations.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
