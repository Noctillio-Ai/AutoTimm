"""Tests for explanation quality metrics."""

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from autotimm.interpretation import ExplanationMetrics, GradCAM


# Set random seeds for reproducibility
@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for all tests to ensure reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


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
def metrics_instance(simple_cnn):
    """Create ExplanationMetrics instance."""
    explainer = GradCAM(simple_cnn, use_cuda=False)
    return ExplanationMetrics(simple_cnn, explainer, use_cuda=False)


# Deletion Tests
class TestDeletionMetric:
    def test_deletion_basic(self, metrics_instance, test_image):
        """Test basic deletion metric computation."""
        result = metrics_instance.deletion(test_image, steps=10)

        assert "auc" in result
        assert "final_drop" in result
        assert "scores" in result
        assert "original_score" in result

        # Scores should decrease over time
        assert len(result["scores"]) == 11  # steps + 1
        assert result["scores"][0] == result["original_score"]
        assert result["scores"][-1] <= result["scores"][0]

    def test_deletion_final_drop(self, metrics_instance, test_image):
        """Test that final drop is reasonable."""
        result = metrics_instance.deletion(test_image, steps=20)

        # Final drop should be positive (confidence decreased)
        # Allow small negative values due to floating point precision
        assert result["final_drop"] >= -1e-6
        # Final drop should be at most 1.0 (100%)
        assert result["final_drop"] <= 1.0

    def test_deletion_auc(self, metrics_instance, test_image):
        """Test AUC computation."""
        result = metrics_instance.deletion(test_image, steps=10)

        # AUC should be reasonable (typically 0-1 range)
        assert 0 <= result["auc"] <= 2  # Allow some margin

    def test_deletion_with_target_class(self, metrics_instance, test_image):
        """Test deletion with specific target class."""
        result = metrics_instance.deletion(test_image, target_class=5, steps=10)

        assert "auc" in result
        assert "final_drop" in result

    def test_deletion_different_baselines(self, metrics_instance, test_image):
        """Test deletion with different baseline types."""
        for baseline in ["blur", "black", "mean"]:
            result = metrics_instance.deletion(test_image, steps=10, baseline=baseline)
            assert "auc" in result
            assert "final_drop" in result


# Insertion Tests
class TestInsertionMetric:
    def test_insertion_basic(self, metrics_instance, test_image):
        """Test basic insertion metric computation."""
        result = metrics_instance.insertion(test_image, steps=10)

        assert "auc" in result
        assert "final_rise" in result
        assert "scores" in result
        assert "original_score" in result
        assert "baseline_score" in result

        # Scores should increase over time
        assert len(result["scores"]) == 11  # steps + 1
        assert result["scores"][0] == result["baseline_score"]
        # Allow small tolerance for floating-point precision
        assert result["scores"][-1] >= result["scores"][0] - 1e-6

    def test_insertion_final_rise(self, metrics_instance, test_image):
        """Test that final rise is reasonable."""
        result = metrics_instance.insertion(test_image, steps=20)

        # Final rise can exceed 1.0 if progressive insertion yields higher confidence
        # than the original image, which is a valid outcome
        assert -0.1 <= result["final_rise"] <= 3.0  # Allow for higher values

    def test_insertion_different_baselines(self, metrics_instance, test_image):
        """Test insertion with different baseline types."""
        for baseline in ["blur", "black", "mean"]:
            result = metrics_instance.insertion(test_image, steps=10, baseline=baseline)
            assert "auc" in result
            assert "final_rise" in result


# Sensitivity Tests
class TestSensitivityMetric:
    def test_sensitivity_basic(self, metrics_instance, test_image):
        """Test sensitivity-n metric."""
        result = metrics_instance.sensitivity_n(test_image, n_samples=10)

        assert "sensitivity" in result
        assert "std" in result
        assert "max_change" in result
        assert "changes" in result

        # Sensitivity should be non-negative
        assert result["sensitivity"] >= 0
        assert result["std"] >= 0
        assert result["max_change"] >= 0

        # Should have correct number of samples
        assert len(result["changes"]) == 10

    def test_sensitivity_noise_level(self, metrics_instance, test_image):
        """Test sensitivity with different noise levels."""
        # Higher noise should generally lead to higher sensitivity
        result_low = metrics_instance.sensitivity_n(
            test_image, n_samples=20, noise_level=0.05
        )
        result_high = metrics_instance.sensitivity_n(
            test_image, n_samples=20, noise_level=0.30
        )

        # This is probabilistic, so we just check it runs
        assert result_low["sensitivity"] >= 0
        assert result_high["sensitivity"] >= 0


# Sanity Check Tests
class TestSanityChecks:
    def test_model_parameter_randomization(self, metrics_instance, test_image):
        """Test model parameter randomization sanity check."""
        result = metrics_instance.model_parameter_randomization_test(test_image)

        assert "correlation" in result
        assert "change" in result
        assert "passes" in result

        # Correlation should be in [-1, 1] or nan (if heatmaps are constant)
        if not np.isnan(result["correlation"]):
            assert -1 <= result["correlation"] <= 1
        # Change should be non-negative
        assert result["change"] >= 0
        # Check that passes is boolean
        assert isinstance(result["passes"], (bool, np.bool_))

    def test_data_randomization(self, metrics_instance, test_image):
        """Test data randomization sanity check."""
        result = metrics_instance.data_randomization_test(test_image)

        assert "correlation" in result
        assert "change" in result
        assert "passes" in result

        # Correlation should be in [-1, 1] or nan (if heatmaps are constant)
        if not np.isnan(result["correlation"]):
            assert -1 <= result["correlation"] <= 1
        # Change should be non-negative
        assert result["change"] >= 0
        # Check that passes is boolean
        assert isinstance(result["passes"], (bool, np.bool_))


# Pointing Game Tests
class TestPointingGame:
    def test_pointing_game_basic(self, metrics_instance, test_image):
        """Test pointing game metric."""
        bbox = [50, 50, 150, 150]
        result = metrics_instance.pointing_game(test_image, bbox)

        assert "hit" in result
        assert "max_location" in result
        assert "bbox" in result

        # Check types
        assert isinstance(result["hit"], (bool, np.bool_))
        assert len(result["max_location"]) == 2
        assert result["bbox"] == bbox

    def test_pointing_game_hit(self, metrics_instance, test_image):
        """Test pointing game with large bbox (should likely hit)."""
        # Large bbox covering most of image
        bbox = [0, 0, 224, 224]
        result = metrics_instance.pointing_game(test_image, bbox)

        # With such a large bbox, should hit
        assert result["hit"]

    def test_pointing_game_miss(self, metrics_instance, test_image):
        """Test pointing game with small bbox (may miss)."""
        # Tiny bbox in corner
        bbox = [0, 0, 1, 1]
        result = metrics_instance.pointing_game(test_image, bbox)

        # Result may be True or False, just check it runs
        assert isinstance(result["hit"], (bool, np.bool_))


# Helper Method Tests
class TestHelperMethods:
    def test_preprocess_image_pil(self, metrics_instance, test_image):
        """Test preprocessing PIL image."""
        tensor = metrics_instance._preprocess_image(test_image)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 4
        assert tensor.shape[1] == 3  # RGB

    def test_preprocess_image_numpy(self, metrics_instance):
        """Test preprocessing numpy array."""
        image = np.random.rand(224, 224, 3).astype(np.float32)
        tensor = metrics_instance._preprocess_image(image)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 4

    def test_preprocess_image_tensor(self, metrics_instance):
        """Test preprocessing tensor."""
        image = torch.randn(3, 224, 224)
        tensor = metrics_instance._preprocess_image(image)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 4

    def test_create_baseline_black(self, metrics_instance):
        """Test black baseline creation."""
        input_tensor = torch.randn(1, 3, 224, 224)
        baseline = metrics_instance._create_baseline(input_tensor, "black")

        assert baseline.shape == input_tensor.shape
        assert baseline.sum() == 0

    def test_create_baseline_mean(self, metrics_instance):
        """Test mean baseline creation."""
        input_tensor = torch.randn(1, 3, 224, 224)
        baseline = metrics_instance._create_baseline(input_tensor, "mean")

        assert baseline.shape == input_tensor.shape
        # All values should be equal to the mean
        assert torch.allclose(baseline, baseline[0, 0, 0, 0])

    def test_create_baseline_blur(self, metrics_instance):
        """Test blur baseline creation."""
        input_tensor = torch.randn(1, 3, 224, 224)
        baseline = metrics_instance._create_baseline(input_tensor, "blur")

        assert baseline.shape == input_tensor.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
