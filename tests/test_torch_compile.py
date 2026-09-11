"""
Tests for torch.compile integration.

Note: Forward-pass tests use compile_model=False because torch.compile's
inductor backend can fail due to stale precompiled headers or missing
C++ toolchains in CI environments. The compilation wrapping itself is
verified by checking for OptimizedModule on model components.

On MPS (Apple Silicon), torch.compile is intentionally skipped because
the inductor backend generates invalid Metal shaders. On Windows,
torch.compile is skipped because the inductor backend requires cl.exe
(MSVC) which is typically unavailable. Tests that check for
OptimizedModule are skipped on those platforms accordingly.
"""

import sys

import pytest
import torch

from autotimm import ImageClassifier, ObjectDetector, SemanticSegmentor

# Check PyTorch version
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
HAS_COMPILE = TORCH_VERSION >= (2, 0) and hasattr(torch, "compile")

# Check if MPS is available (Apple Silicon) — torch.compile is skipped on MPS
MPS_AVAILABLE = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

# torch.compile only actually wraps when supported AND not on MPS/Windows
COMPILE_WRAPS = HAS_COMPILE and not MPS_AVAILABLE and sys.platform != "win32"


class TestTorchCompileClassifier:
    """Test torch.compile integration for ImageClassifier."""

    @pytest.mark.skipif(
        not COMPILE_WRAPS,
        reason="torch.compile not effective (MPS, Windows, or PyTorch < 2.0)",
    )
    def test_compile_enabled_by_default(self):
        """Test that compile_model=True wraps components in OptimizedModule."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
        )

        # When compile_model=True (default), backbone and head are wrapped
        assert type(model.backbone).__name__ == "OptimizedModule"
        assert type(model.head).__name__ == "OptimizedModule"

    @pytest.mark.skipif(not HAS_COMPILE, reason="torch.compile requires PyTorch 2.0+")
    def test_compile_disabled_no_wrapping(self):
        """Test that compile_model=False keeps components unwrapped."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            compile_model=False,
        )

        # When compile_model=False, components should NOT be OptimizedModule
        assert type(model.backbone).__name__ != "OptimizedModule"
        assert type(model.head).__name__ != "OptimizedModule"

    @pytest.mark.skipif(
        not COMPILE_WRAPS,
        reason="torch.compile not effective (MPS, Windows, or PyTorch < 2.0)",
    )
    def test_compile_kwargs_accepted(self):
        """Test that custom compile_kwargs are accepted without error."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            compile_kwargs={"mode": "reduce-overhead"},
        )

        # Model should be created successfully with custom kwargs
        assert type(model.backbone).__name__ == "OptimizedModule"

    @pytest.mark.skipif(
        not COMPILE_WRAPS,
        reason="torch.compile not effective (MPS, Windows, or PyTorch < 2.0)",
    )
    def test_compile_kwargs_none_accepted(self):
        """Test that compile_kwargs=None is accepted."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            compile_model=True,
            compile_kwargs=None,
        )

        assert type(model.backbone).__name__ == "OptimizedModule"

    def test_uncompiled_forward_pass(self):
        """Test that uncompiled model performs forward pass correctly."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            compile_model=False,
        )

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 10)

    def test_fallback_on_old_pytorch(self):
        """Test graceful fallback on PyTorch < 2.0."""
        if HAS_COMPILE:
            pytest.skip("Test only for PyTorch < 2.0")

        # Should not crash, just use uncompiled model
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            compile_model=True,
        )

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 10)

    @pytest.mark.skipif(not HAS_COMPILE, reason="torch.compile requires PyTorch 2.0+")
    def test_compile_skipped_on_mps(self):
        """Test that compile_model=True gracefully skips on MPS."""
        if not MPS_AVAILABLE:
            pytest.skip("Test only for MPS devices")

        model = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            compile_model=True,
        )

        # On MPS, torch.compile is skipped, so components remain unwrapped
        assert type(model.backbone).__name__ != "OptimizedModule"
        assert type(model.head).__name__ != "OptimizedModule"

        # But model should still work
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 10)


class TestTorchCompileDetection:
    """Test torch.compile integration for ObjectDetector."""

    @pytest.mark.skipif(
        not COMPILE_WRAPS,
        reason="torch.compile not effective (MPS, Windows, or PyTorch < 2.0)",
    )
    def test_detection_compile_wraps_components(self):
        """Test that compile_model=True wraps detection components."""
        model = ObjectDetector(
            backbone="resnet18",
            num_classes=80,
        )

        # Backbone should be wrapped
        assert type(model.backbone).__name__ == "OptimizedModule"

    @pytest.mark.skipif(not HAS_COMPILE, reason="torch.compile requires PyTorch 2.0+")
    def test_detection_compile_disabled(self):
        """Test that compile_model=False keeps detection components unwrapped."""
        model = ObjectDetector(
            backbone="resnet18",
            num_classes=80,
            compile_model=False,
        )

        assert type(model.backbone).__name__ != "OptimizedModule"

    @pytest.mark.skipif(
        not COMPILE_WRAPS,
        reason="torch.compile not effective (MPS, Windows, or PyTorch < 2.0)",
    )
    def test_detection_custom_compile_kwargs(self):
        """Test custom compile options for detection."""
        model = ObjectDetector(
            backbone="resnet18",
            num_classes=80,
            compile_model=True,
            compile_kwargs={"mode": "default"},
        )

        assert type(model.backbone).__name__ == "OptimizedModule"

    def test_detection_uncompiled_forward_pass(self):
        """Test that uncompiled detection model performs forward pass."""
        model = ObjectDetector(
            backbone="resnet18",
            num_classes=80,
            compile_model=False,
        )

        x = torch.randn(2, 3, 640, 640)
        model.eval()
        with torch.no_grad():
            outputs = model(x)

        assert isinstance(outputs, (dict, list, tuple))


class TestTorchCompileSegmentation:
    """Test torch.compile integration for SemanticSegmentor."""

    @pytest.mark.skipif(
        not COMPILE_WRAPS,
        reason="torch.compile not effective (MPS, Windows, or PyTorch < 2.0)",
    )
    def test_segmentation_compile_wraps_components(self):
        """Test that compile_model=True wraps segmentation components."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=19,
        )

        assert type(model.backbone).__name__ == "OptimizedModule"
        assert type(model.head).__name__ == "OptimizedModule"

    @pytest.mark.skipif(not HAS_COMPILE, reason="torch.compile requires PyTorch 2.0+")
    def test_segmentation_compile_disabled(self):
        """Test that compile_model=False keeps segmentation components unwrapped."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=19,
            compile_model=False,
        )

        assert type(model.backbone).__name__ != "OptimizedModule"
        assert type(model.head).__name__ != "OptimizedModule"

    def test_segmentation_uncompiled_forward_pass(self):
        """Test that uncompiled segmentation model performs forward pass."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=19,
            compile_model=False,
        )

        x = torch.randn(2, 3, 256, 256)
        model.eval()
        with torch.no_grad():
            output = model(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 19

    @pytest.mark.skipif(
        not COMPILE_WRAPS,
        reason="torch.compile not effective (MPS, Windows, or PyTorch < 2.0)",
    )
    def test_segmentation_compile_modes_accepted(self):
        """Test that different compile modes are accepted without error."""
        modes = ["default", "reduce-overhead", "max-autotune"]

        for mode in modes:
            model = SemanticSegmentor(
                backbone="resnet18",
                num_classes=19,
                compile_model=True,
                compile_kwargs={"mode": mode},
            )

            assert type(model.backbone).__name__ == "OptimizedModule"


class TestCompileComponents:
    """Test that specific components are compiled correctly."""

    @pytest.mark.skipif(
        not COMPILE_WRAPS,
        reason="torch.compile not effective (MPS, Windows, or PyTorch < 2.0)",
    )
    def test_classifier_components_compiled(self):
        """Test that backbone and head are compiled for classifier."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            compile_model=True,
        )

        # Classifier should have backbone and head wrapped
        assert hasattr(model, "backbone")
        assert hasattr(model, "head")
        assert type(model.backbone).__name__ == "OptimizedModule"
        assert type(model.head).__name__ == "OptimizedModule"

    @pytest.mark.skipif(
        not COMPILE_WRAPS,
        reason="torch.compile not effective (MPS, Windows, or PyTorch < 2.0)",
    )
    def test_detector_components_compiled(self):
        """Test that detector components are compiled."""
        model = ObjectDetector(
            backbone="resnet18",
            num_classes=80,
            compile_model=True,
        )

        assert hasattr(model, "backbone")
        assert type(model.backbone).__name__ == "OptimizedModule"

    @pytest.mark.skipif(
        not COMPILE_WRAPS,
        reason="torch.compile not effective (MPS, Windows, or PyTorch < 2.0)",
    )
    def test_segmentor_components_compiled(self):
        """Test that segmentation components are compiled."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=19,
            compile_model=True,
        )

        assert hasattr(model, "backbone")
        assert hasattr(model, "head")
        assert type(model.backbone).__name__ == "OptimizedModule"
        assert type(model.head).__name__ == "OptimizedModule"

    @pytest.mark.skipif(
        not COMPILE_WRAPS,
        reason="torch.compile not effective (MPS, Windows, or PyTorch < 2.0)",
    )
    def test_compiled_vs_uncompiled_identical_weights(self):
        """Test that compilation doesn't change model weights."""
        model_compiled = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            seed=42,
            compile_model=True,
        )

        model_uncompiled = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            seed=42,
            compile_model=False,
        )

        # Access original modules from compiled wrappers
        compiled_fc = model_compiled.head._orig_mod.fc.weight
        uncompiled_fc = model_uncompiled.head.fc.weight

        assert torch.allclose(compiled_fc, uncompiled_fc, rtol=1e-5)
