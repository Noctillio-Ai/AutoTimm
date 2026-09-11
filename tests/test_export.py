"""Tests for model export functionality."""

import os

import numpy as np
import pytest
import torch

from autotimm import ImageClassifier
from autotimm.core.metrics import MetricConfig
from autotimm.export import (
    export_to_torchscript,
    load_torchscript,
    validate_torchscript_export,
)

try:
    import onnx
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


@pytest.fixture(autouse=True, scope="function")
def reset_deterministic_mode():
    """Reset deterministic mode before each test.

    This is necessary because torch.use_deterministic_algorithms() is a global
    setting that persists across tests. Other tests may enable it via seed_everything(),
    which can cause TorchScript serialization to fail.
    """
    # Save original state
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_cudnn_deterministic = torch.backends.cudnn.deterministic
    original_cudnn_benchmark = torch.backends.cudnn.benchmark
    original_cublas_allow_tf32 = None
    original_cudnn_allow_tf32 = None

    # Save TF32 settings if available (PyTorch >= 1.7)
    try:
        original_cublas_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        original_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    except AttributeError:
        pass

    # Disable all deterministic algorithms aggressively
    try:
        torch.use_deterministic_algorithms(False, warn_only=False)
    except TypeError:
        # PyTorch < 1.11 doesn't have warn_only parameter
        torch.use_deterministic_algorithms(False)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Enable TF32 if available (helps with performance and compatibility)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except AttributeError:
        pass

    # Clear any CUBLAS workspace configuration that might interfere
    if "CUBLAS_WORKSPACE_CONFIG" in os.environ:
        del os.environ["CUBLAS_WORKSPACE_CONFIG"]

    yield

    # Restore original state after test
    try:
        torch.use_deterministic_algorithms(original_deterministic, warn_only=False)
    except TypeError:
        torch.use_deterministic_algorithms(original_deterministic)

    torch.backends.cudnn.deterministic = original_cudnn_deterministic
    torch.backends.cudnn.benchmark = original_cudnn_benchmark

    # Restore TF32 settings if available
    if original_cublas_allow_tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = original_cublas_allow_tf32
    if original_cudnn_allow_tf32 is not None:
        torch.backends.cudnn.allow_tf32 = original_cudnn_allow_tf32


@pytest.fixture
def simple_classifier():
    """Create a simple classifier for testing."""
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
        compile_model=False,  # Disable torch.compile for TorchScript compatibility
        seed=None,  # Disable seeding for TorchScript compatibility
        deterministic=False,
    )
    model.eval()
    return model


def test_export_to_torchscript_trace(simple_classifier, tmp_path):
    """Test exporting model to TorchScript using trace method."""
    save_path = tmp_path / "model.pt"
    example_input = torch.randn(1, 3, 224, 224)

    export_to_torchscript(
        simple_classifier,
        save_path,
        example_input=example_input,
        method="trace",
    )

    # Check file was created
    assert save_path.exists()

    # Check model can be loaded
    loaded_model = torch.jit.load(str(save_path))
    assert loaded_model is not None

    # Check inference works
    with torch.no_grad():
        output = loaded_model(example_input)
    assert output.shape == (1, 10)


@pytest.mark.skip(reason="torch.jit.script doesn't support Python 3.10+ union types")
def test_export_to_torchscript_script(simple_classifier, tmp_path):
    """Test exporting model to TorchScript using script method.

    Note: Scripting is not well supported for complex models with modern Python
    type annotations. Use tracing (method='trace') instead, which is the recommended approach.
    """
    save_path = tmp_path / "model.pt"

    # Script method doesn't require example input
    export_to_torchscript(
        simple_classifier,
        save_path,
        method="script",
    )

    # Check file was created
    assert save_path.exists()

    # Check model can be loaded
    loaded_model = torch.jit.load(str(save_path))

    # Check inference works
    example_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = loaded_model(example_input)
    assert output.shape == (1, 10)


def test_load_torchscript(simple_classifier, tmp_path):
    """Test loading a TorchScript model."""
    save_path = tmp_path / "model.pt"
    example_input = torch.randn(1, 3, 224, 224)

    # Export model
    export_to_torchscript(
        simple_classifier,
        save_path,
        example_input=example_input,
    )

    # Load model
    loaded_model = load_torchscript(save_path, device="cpu")
    assert loaded_model is not None

    # Test inference
    with torch.no_grad():
        output = loaded_model(example_input)
    assert output.shape == (1, 10)


def test_validate_torchscript_export(simple_classifier, tmp_path):
    """Test validating TorchScript export matches original model."""
    example_input = torch.randn(1, 3, 224, 224)
    save_path = tmp_path / "validate_model.pt"

    scripted_model = export_to_torchscript(
        simple_classifier,
        save_path,
        example_input=example_input,
    )

    # Validate outputs match
    is_valid = validate_torchscript_export(
        simple_classifier,
        scripted_model,
        example_input,
    )
    assert is_valid


def test_model_to_torchscript_method(tmp_path):
    """Test the to_torchscript() convenience method on model."""
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        compile_model=False,
        seed=None,  # Disable seeding for TorchScript compatibility
        deterministic=False,
    )
    model.eval()

    save_path = tmp_path / "model.pt"
    model.to_torchscript(str(save_path))

    # Check file was created
    assert save_path.exists()

    # Check model can be loaded
    loaded = torch.jit.load(str(save_path))

    # Test inference
    example_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = loaded(example_input)
    assert output.shape == (1, 10)


def test_model_to_torchscript_no_save(tmp_path):
    """Test to_torchscript() without saving to file."""
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        compile_model=False,
        seed=None,  # Disable seeding for TorchScript compatibility
        deterministic=False,
    )
    model.eval()

    # Get scripted model — pass a path to avoid NamedTemporaryFile locking
    # issues on Windows; the method internally creates a temp file otherwise.
    save_path = tmp_path / "no_save_model.pt"
    scripted = model.to_torchscript(str(save_path))

    # Test inference
    example_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = scripted(example_input)
    assert output.shape == (1, 10)


def test_export_trace_without_example_input(tmp_path):
    """Test that tracing without example_input raises error."""
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        compile_model=False,
        seed=None,  # Disable seeding for TorchScript compatibility
        deterministic=False,
    )

    save_path = tmp_path / "no_input_model.pt"
    with pytest.raises(RuntimeError, match="example_input is required"):
        export_to_torchscript(
            model,
            save_path,
            example_input=None,
            method="trace",
        )


def test_export_with_invalid_method(tmp_path):
    """Test that invalid export method raises error."""
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        compile_model=False,
        seed=None,  # Disable seeding for TorchScript compatibility
        deterministic=False,
    )

    save_path = tmp_path / "invalid_method_model.pt"
    with pytest.raises(RuntimeError, match="Unknown method"):
        export_to_torchscript(
            model,
            save_path,
            example_input=torch.randn(1, 3, 224, 224),
            method="invalid",
        )


def test_export_preserves_training_mode(tmp_path):
    """Test that export preserves the original training mode."""
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        compile_model=False,
        seed=None,  # Disable seeding for TorchScript compatibility
        deterministic=False,
    )
    model.train()  # Set to training mode

    save_path = tmp_path / "training_mode_model.pt"
    export_to_torchscript(
        model,
        save_path,
        example_input=torch.randn(1, 3, 224, 224),
    )

    # Should still be in training mode
    assert model.training


# ============================================================================
# ONNX Export Tests
# ============================================================================


@pytest.mark.skipif(not HAS_ONNX, reason="onnx/onnxruntime not installed")
def test_export_to_onnx_classification(simple_classifier, tmp_path):
    """Test exporting classification model to ONNX format."""
    from autotimm.export import export_to_onnx

    save_path = tmp_path / "model.onnx"
    example_input = torch.randn(1, 3, 224, 224)

    result_path = export_to_onnx(
        simple_classifier,
        save_path,
        example_input=example_input,
    )

    # Check file was created
    assert save_path.exists()
    assert result_path == str(save_path)

    # Check model can be loaded and validated
    onnx_model = onnx.load(str(save_path))
    onnx.checker.check_model(onnx_model)

    # Check inference works
    session = ort.InferenceSession(str(save_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: example_input.numpy()})
    assert outputs[0].shape == (1, 10)


@pytest.mark.skipif(not HAS_ONNX, reason="onnx/onnxruntime not installed")
def test_load_onnx(simple_classifier, tmp_path):
    """Test loading an ONNX model."""
    from autotimm.export import export_to_onnx, load_onnx

    save_path = tmp_path / "model.onnx"
    example_input = torch.randn(1, 3, 224, 224)

    export_to_onnx(simple_classifier, save_path, example_input)

    # Load model
    session = load_onnx(save_path)
    assert session is not None

    # Test inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: example_input.numpy()})
    assert outputs[0].shape == (1, 10)


@pytest.mark.skipif(not HAS_ONNX, reason="onnx/onnxruntime not installed")
def test_validate_onnx_export(simple_classifier, tmp_path):
    """Test validating ONNX export matches original model."""
    from autotimm.export import export_to_onnx, validate_onnx_export

    save_path = tmp_path / "model.onnx"
    example_input = torch.randn(1, 3, 224, 224)

    export_to_onnx(simple_classifier, save_path, example_input)

    is_valid = validate_onnx_export(
        simple_classifier,
        save_path,
        example_input,
    )
    assert is_valid


@pytest.mark.skipif(not HAS_ONNX, reason="onnx/onnxruntime not installed")
def test_model_to_onnx_method(tmp_path):
    """Test the to_onnx() convenience method on model."""
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        compile_model=False,
        seed=None,
        deterministic=False,
    )
    model.eval()

    save_path = tmp_path / "model.onnx"
    result_path = model.to_onnx(str(save_path))

    # Check file was created
    assert save_path.exists()
    assert result_path == str(save_path)

    # Check inference works
    session = ort.InferenceSession(str(save_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    example_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = session.run(None, {input_name: example_input})
    assert outputs[0].shape == (1, 10)


@pytest.mark.skipif(not HAS_ONNX, reason="onnx/onnxruntime not installed")
def test_export_to_onnx_dynamic_axes(simple_classifier, tmp_path):
    """Test ONNX export with dynamic batch dimension."""
    from autotimm.export import export_to_onnx

    save_path = tmp_path / "model.onnx"
    example_input = torch.randn(1, 3, 224, 224)

    # Export with default dynamic axes (batch dimension)
    export_to_onnx(simple_classifier, save_path, example_input)

    # Test with different batch sizes
    session = ort.InferenceSession(str(save_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    for batch_size in [1, 2, 4]:
        test_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {input_name: test_input})
        assert outputs[0].shape == (batch_size, 10)


@pytest.mark.skipif(not HAS_ONNX, reason="onnx/onnxruntime not installed")
def test_export_onnx_without_example_input(tmp_path):
    """Test that ONNX export without example_input raises error."""
    from autotimm.export import export_to_onnx

    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        compile_model=False,
        seed=None,
        deterministic=False,
    )

    save_path = tmp_path / "model.onnx"
    with pytest.raises(RuntimeError):
        export_to_onnx(model, save_path, example_input=None)


@pytest.mark.skipif(not HAS_ONNX, reason="onnx/onnxruntime not installed")
def test_export_onnx_preserves_training_mode(tmp_path):
    """Test that ONNX export preserves the original training mode."""
    from autotimm.export import export_to_onnx

    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        compile_model=False,
        seed=None,
        deterministic=False,
    )
    model.train()  # Set to training mode

    save_path = tmp_path / "model.onnx"
    export_to_onnx(model, save_path, example_input=torch.randn(1, 3, 224, 224))

    # Should still be in training mode
    assert model.training
