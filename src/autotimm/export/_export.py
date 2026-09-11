"""Model export utilities for TorchScript and ONNX formats.

This module provides utilities to export trained AutoTimm models to various formats
for deployment, including TorchScript (.pt files) and ONNX (.onnx files) for
production inference across multiple runtimes and platforms.
"""

from __future__ import annotations

import contextlib
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from autotimm.core.logging import logger

try:
    import onnx
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import onnxsim

    HAS_ONNXSIM = True
except ImportError:
    HAS_ONNXSIM = False


def _check_onnx_deps() -> None:
    """Check that ONNX dependencies are installed."""
    if not HAS_ONNX:
        raise ImportError(
            "ONNX export requires 'onnx' and 'onnxruntime' packages. "
            "Install them with: pip install onnx onnxruntime"
        )


@contextlib.contextmanager
def _lightning_export_mode(model: nn.Module):
    """Context manager to temporarily make Lightning modules TorchScript-compatible.

    Temporarily replaces problematic properties with None during export.
    """
    # Check if it's a Lightning module
    is_lightning = any("LightningModule" in cls.__name__ for cls in type(model).__mro__)

    if not is_lightning:
        yield
        return

    # Temporarily replace trainer property
    original_trainer = type(model).trainer

    try:
        # Replace with a simple property that returns None
        type(model).trainer = property(lambda self: None)
        yield
    finally:
        # Restore original property
        type(model).trainer = original_trainer


class _ForwardWrapper(nn.Module):
    """Wrapper to export only the forward method, avoiding Lightning module issues.

    This wrapper creates a clean module containing only the essential components
    needed for inference (backbone and head), avoiding problematic attributes like
    metrics, logging configs, and other Lightning-specific state.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


class _DetectionForwardWrapper(nn.Module):
    """Wrapper for FCOS-based detection models that flattens list outputs to named tensors.

    ONNX does not support nested list outputs, so this wrapper converts the
    (cls_outputs, reg_outputs, centerness_outputs) lists into flat named tensors.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.fpn = model.fpn
        self.head = model.head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        cls_outputs, reg_outputs, centerness_outputs = self.head(fpn_features)
        return (*cls_outputs, *reg_outputs, *centerness_outputs)


class _InstanceDetectionForwardWrapper(nn.Module):
    """Wrapper for instance segmentation models that flattens detection outputs.

    Exports the detection head only (mask head excluded), flattening
    (cls_outputs, reg_outputs, centerness_outputs) into named tensors.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.fpn = model.fpn
        self.detection_head = model.detection_head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        cls_outputs, reg_outputs, centerness_outputs = self.detection_head(fpn_features)
        return (*cls_outputs, *reg_outputs, *centerness_outputs)


class _YOLOXForwardWrapper(nn.Module):
    """Wrapper for YOLOX detection models that flattens list outputs to named tensors.

    Converts (cls_outputs, reg_outputs) lists into flat named tensors.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.head = model.head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.backbone(x)
        pafpn_outputs = self.neck(features)
        cls_outputs, reg_outputs = self.head(pafpn_outputs)
        return (*cls_outputs, *reg_outputs)


def _get_detection_output_names(model: nn.Module) -> list[str]:
    """Get flattened output names for detection models."""
    model_class_name = type(model).__name__

    if model_class_name == "YOLOXDetector":
        # 3 FPN levels: cls_l0..cls_l2, reg_l0..reg_l2
        names = []
        for prefix in ("cls", "reg"):
            for i in range(3):
                names.append(f"{prefix}_l{i}")
        return names

    # FCOS-based: ObjectDetector or InstanceSegmentor
    # 5 FPN levels: cls_l0..cls_l4, reg_l0..reg_l4, ctr_l0..ctr_l4
    names = []
    for prefix in ("cls", "reg", "ctr"):
        for i in range(5):
            names.append(f"{prefix}_l{i}")
    return names


def _get_export_wrapper(model: nn.Module) -> nn.Module | None:
    """Get the appropriate forward wrapper for a model, or None for simple models."""
    model_class_name = type(model).__name__

    if model_class_name == "ObjectDetector":
        return _DetectionForwardWrapper(model)
    elif model_class_name == "InstanceSegmentor":
        return _InstanceDetectionForwardWrapper(model)
    elif model_class_name == "YOLOXDetector":
        return _YOLOXForwardWrapper(model)
    return None


def export_to_torchscript(
    model: nn.Module,
    save_path: str | Path,
    example_input: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
    method: str = "trace",
    optimize: bool = True,
    wrap_model: bool = True,
) -> torch.jit.ScriptModule:
    """Export a PyTorch model to TorchScript format.

    TorchScript allows models to be serialized and run without Python dependencies,
    making them suitable for production deployment, mobile devices, and C++ environments.

    Args:
        model: The PyTorch model to export. Should be in eval mode.
        save_path: Path where the TorchScript model will be saved (.pt extension recommended).
        example_input: Example input tensor(s) for tracing. Required if method="trace".
            Should match the expected input shape and type for the model.
        method: Export method. Options:
            - "trace": Uses torch.jit.trace (recommended for most models, requires example_input)
            - "script": Uses torch.jit.script (works without example_input but may fail on dynamic code)
        optimize: If True (default), applies TorchScript optimizations for inference.
        wrap_model: If True (default), wraps PyTorch Lightning modules to avoid compatibility issues.

    Returns:
        The compiled TorchScript module.

    Raises:
        ValueError: If method="trace" but example_input is not provided.
        RuntimeError: If TorchScript compilation fails.

    Example:
        >>> from autotimm import ImageClassifier
        >>> from autotimm.export import export_to_torchscript
        >>> import torch
        >>>
        >>> # Load trained model
        >>> model = ImageClassifier.load_from_checkpoint("model.ckpt")
        >>> model.eval()
        >>>
        >>> # Export with tracing (recommended)
        >>> example_input = torch.randn(1, 3, 224, 224)
        >>> scripted_model = export_to_torchscript(
        ...     model,
        ...     "model.pt",
        ...     example_input=example_input,
        ...     method="trace"
        ... )
        >>>
        >>> # Load and use the exported model
        >>> loaded_model = torch.jit.load("model.pt")
        >>> output = loaded_model(example_input)

    Note:
        - Models with torch.compile enabled should be exported after loading
        - Tracing is generally more reliable than scripting for complex models
        - The exported model is optimized for inference (forward pass only)
        - Training-specific features (optimizer, scheduler) are not included
    """
    save_path = Path(save_path)

    # Ensure model is in eval mode
    was_training = model.training
    model.eval()

    try:
        # Use context manager to make Lightning modules TorchScript-compatible
        with _lightning_export_mode(model):
            # Wrap the model if requested (default for Lightning modules)
            is_lightning = any(
                "LightningModule" in cls.__name__ for cls in type(model).__mro__
            )
            if wrap_model and is_lightning:
                detection_wrapper = _get_export_wrapper(model)
                if detection_wrapper is not None:
                    model_to_export = detection_wrapper
                else:
                    model_to_export = _ForwardWrapper(model)
            else:
                model_to_export = model

            if method == "trace":
                if example_input is None:
                    raise ValueError(
                        "example_input is required when method='trace'. "
                        "Provide a sample input tensor with the expected shape."
                    )

                with torch.inference_mode():
                    scripted_model = torch.jit.trace(model_to_export, example_input)

            elif method == "script":
                # Note: scripting is less reliable for complex models
                scripted_model = torch.jit.script(model_to_export)

            else:
                raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'.")

        # Save the model before optimization (optimize_for_inference can produce
        # frozen modules that fail to deserialize with torch.jit.load)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(scripted_model, str(save_path))

        # Optimize for inference (in-memory only)
        if optimize:
            try:
                scripted_model = torch.jit.optimize_for_inference(scripted_model)
            except RuntimeError as e:
                warnings.warn(
                    f"torch.jit.optimize_for_inference failed: {e}. "
                    "Falling back to non-optimized model. "
                    "Set optimize=False to suppress this warning.",
                    stacklevel=2,
                )

        logger.success(f"Model exported to TorchScript: {save_path}")
        return scripted_model

    except Exception as e:
        raise RuntimeError(
            f"Failed to export model to TorchScript: {e}\n"
            f"Try using method='trace' with a valid example_input, "
            f"or ensure your model is compatible with TorchScript."
        ) from e

    finally:
        # Restore training mode
        if was_training:
            model.train()


def load_torchscript(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> torch.jit.ScriptModule:
    """Load a TorchScript model.

    Args:
        path: Path to the TorchScript model file (.pt).
        device: Device to load the model on ("cpu", "cuda", etc.).

    Returns:
        The loaded TorchScript module ready for inference.

    Example:
        >>> from autotimm.export import load_torchscript
        >>> import torch
        >>>
        >>> # Load model
        >>> model = load_torchscript("model.pt", device="cuda")
        >>>
        >>> # Run inference
        >>> with torch.no_grad():
        ...     output = model(torch.randn(1, 3, 224, 224).cuda())
    """
    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    return model


def export_checkpoint_to_torchscript(
    checkpoint_path: str | Path,
    save_path: str | Path,
    model_class: type[nn.Module],
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    method: str = "trace",
    load_kwargs: dict[str, Any] | None = None,
    **export_kwargs: Any,
) -> torch.jit.ScriptModule:
    """Export a Lightning checkpoint directly to TorchScript.

    Convenience function that loads a checkpoint and exports it in one step.

    Args:
        checkpoint_path: Path to the PyTorch Lightning checkpoint (.ckpt).
        save_path: Path where the TorchScript model will be saved (.pt).
        model_class: The AutoTimm model class (e.g., ImageClassifier, ObjectDetector).
        example_input: Example input tensor(s) for tracing.
        method: Export method ("trace" or "script").
        load_kwargs: Additional kwargs to pass to model_class.load_from_checkpoint().
        **export_kwargs: Additional kwargs to pass to export_to_torchscript().

    Returns:
        The compiled TorchScript module.

    Example:
        >>> from autotimm import ImageClassifier
        >>> from autotimm.export import export_checkpoint_to_torchscript
        >>> import torch
        >>>
        >>> # Export checkpoint to TorchScript
        >>> scripted_model = export_checkpoint_to_torchscript(
        ...     checkpoint_path="model.ckpt",
        ...     save_path="model.pt",
        ...     model_class=ImageClassifier,
        ...     example_input=torch.randn(1, 3, 224, 224),
        ... )
    """
    load_kwargs = load_kwargs or {}

    # Load the checkpoint on CPU to avoid device mismatches during tracing
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model = model_class.load_from_checkpoint(
        str(checkpoint_path), map_location="cpu", **load_kwargs
    )

    # Export to TorchScript
    return export_to_torchscript(
        model=model,
        save_path=save_path,
        example_input=example_input,
        method=method,
        **export_kwargs,
    )


def validate_torchscript_export(
    original_model: nn.Module,
    scripted_model: torch.jit.ScriptModule,
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> bool:
    """Validate that TorchScript export produces identical outputs to the original model.

    Args:
        original_model: The original PyTorch model.
        scripted_model: The exported TorchScript model.
        example_input: Input tensor(s) to test with.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        True if outputs match within tolerance, False otherwise.

    Example:
        >>> from autotimm.export import export_to_torchscript, validate_torchscript_export
        >>> import torch
        >>>
        >>> example_input = torch.randn(1, 3, 224, 224)
        >>> scripted = export_to_torchscript(model, "model.pt", example_input)
        >>>
        >>> # Validate export
        >>> is_valid = validate_torchscript_export(model, scripted, example_input)
        >>> print(f"Export valid: {is_valid}")
    """
    original_model.eval()
    scripted_model.eval()

    with torch.inference_mode():
        original_output = original_model(example_input)
        scripted_output = scripted_model(example_input)

    # Handle different output types
    if isinstance(original_output, torch.Tensor):
        return torch.allclose(original_output, scripted_output, rtol=rtol, atol=atol)
    elif isinstance(original_output, (tuple, list)):
        if len(original_output) != len(scripted_output):
            return False
        return all(
            (
                torch.allclose(o, s, rtol=rtol, atol=atol)
                if isinstance(o, torch.Tensor)
                else o == s
            )
            for o, s in zip(original_output, scripted_output)
        )
    else:
        warnings.warn(
            f"Cannot validate output type: {type(original_output)}. "
            "Manual validation recommended."
        )
        return False


def export_to_onnx(
    model: nn.Module,
    save_path: str | Path,
    example_input: torch.Tensor,
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    wrap_model: bool = True,
    simplify: bool = False,
) -> str:
    """Export a PyTorch model to ONNX format.

    ONNX enables deployment to ONNX Runtime, TensorRT, OpenVINO, CoreML, and
    other inference engines across platforms.

    Args:
        model: The PyTorch model to export.
        save_path: Path where the ONNX model will be saved (.onnx extension recommended).
        example_input: Example input tensor for tracing. Required for ONNX export.
        opset_version: ONNX opset version. Default is 17.
        dynamic_axes: Dynamic axes specification. If None, batch dimension is made dynamic.
        input_names: Names for input tensors. Defaults to ["input"].
        output_names: Names for output tensors. Defaults to ["output"] for simple models,
            or per-level names for detection models.
        wrap_model: If True (default), wraps PyTorch Lightning modules to avoid
            compatibility issues.
        simplify: If True, simplifies the ONNX graph using onnxsim (requires
            onnx-simplifier package). Default is False.

    Returns:
        The save path as a string.

    Raises:
        ImportError: If onnx or onnxruntime packages are not installed.
        ValueError: If example_input is not provided.
        RuntimeError: If ONNX export fails.

    Example:
        >>> from autotimm import ImageClassifier
        >>> from autotimm.export import export_to_onnx
        >>> import torch
        >>>
        >>> model = ImageClassifier(backbone="resnet18", num_classes=10)
        >>> model.eval()
        >>> example_input = torch.randn(1, 3, 224, 224)
        >>> export_to_onnx(model, "model.onnx", example_input)
    """
    _check_onnx_deps()
    save_path = Path(save_path)

    if input_names is None:
        input_names = ["input"]

    # Determine wrapper and output names for detection models
    is_lightning = any("LightningModule" in cls.__name__ for cls in type(model).__mro__)
    detection_wrapper = (
        _get_export_wrapper(model) if wrap_model and is_lightning else None
    )

    if output_names is None:
        if detection_wrapper is not None:
            output_names = _get_detection_output_names(model)
        else:
            output_names = ["output"]

    # Auto-generate dynamic_axes for batch dimension if not provided
    if dynamic_axes is None:
        dynamic_axes = {name: {0: "batch_size"} for name in input_names}
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}

    # Ensure model is in eval mode
    was_training = model.training
    model.eval()

    try:
        with _lightning_export_mode(model):
            if detection_wrapper is not None:
                model_to_export = detection_wrapper
            elif wrap_model and is_lightning:
                model_to_export = _ForwardWrapper(model)
            else:
                model_to_export = model

            model_to_export.eval()

            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.onnx.export(
                model_to_export,
                example_input,
                str(save_path),
                opset_version=opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

        # Optional simplification
        if simplify:
            if not HAS_ONNXSIM:
                warnings.warn(
                    "onnx-simplifier not installed. Skipping simplification. "
                    "Install with: pip install onnx-simplifier",
                    stacklevel=2,
                )
            else:
                onnx_model = onnx.load(str(save_path))
                simplified, check = onnxsim.simplify(onnx_model)
                if check:
                    onnx.save(simplified, str(save_path))
                else:
                    warnings.warn(
                        "ONNX simplification could not be validated. "
                        "Using original model.",
                        stacklevel=2,
                    )

        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        logger.success(f"Model exported to ONNX: {save_path} ({file_size_mb:.1f} MB)")
        return str(save_path)

    except Exception as e:
        raise RuntimeError(
            f"Failed to export model to ONNX: {e}\n"
            f"Ensure your model is compatible with ONNX export "
            f"(opset_version={opset_version})."
        ) from e

    finally:
        if was_training:
            model.train()


def load_onnx(
    path: str | Path,
    providers: list[str] | None = None,
) -> Any:
    """Load an ONNX model and create an inference session.

    Args:
        path: Path to the ONNX model file (.onnx).
        providers: ONNX Runtime execution providers. Defaults to ["CPUExecutionProvider"].

    Returns:
        An onnxruntime.InferenceSession ready for inference.

    Example:
        >>> from autotimm.export import load_onnx
        >>> import numpy as np
        >>>
        >>> session = load_onnx("model.onnx")
        >>> input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        >>> outputs = session.run(None, {"input": input_data})
    """
    _check_onnx_deps()

    if providers is None:
        providers = ["CPUExecutionProvider"]

    # Validate the model
    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)

    return ort.InferenceSession(str(path), providers=providers)


def validate_onnx_export(
    original_model: nn.Module,
    onnx_path: str | Path,
    example_input: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> bool:
    """Validate that ONNX export produces outputs matching the original model.

    Args:
        original_model: The original PyTorch model.
        onnx_path: Path to the exported ONNX model.
        example_input: Input tensor to test with.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        True if outputs match within tolerance, False otherwise.

    Example:
        >>> from autotimm.export import export_to_onnx, validate_onnx_export
        >>> import torch
        >>>
        >>> example_input = torch.randn(1, 3, 224, 224)
        >>> export_to_onnx(model, "model.onnx", example_input)
        >>> is_valid = validate_onnx_export(model, "model.onnx", example_input)
    """
    _check_onnx_deps()

    original_model.eval()

    # Get PyTorch output
    with torch.inference_mode():
        original_output = original_model(example_input)

    # Get ONNX output
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_outputs = session.run(None, {input_name: example_input.numpy()})

    # Compare outputs
    if isinstance(original_output, torch.Tensor):
        return np.allclose(
            original_output.numpy(), onnx_outputs[0], rtol=rtol, atol=atol
        )
    elif isinstance(original_output, (tuple, list)):
        # Flatten nested lists (detection models return list of tensors)
        flat_original = []
        for item in original_output:
            if isinstance(item, (tuple, list)):
                flat_original.extend(item)
            elif isinstance(item, torch.Tensor):
                flat_original.append(item)
            elif item is None:
                continue

        if len(flat_original) != len(onnx_outputs):
            return False
        return all(
            np.allclose(o.numpy(), s, rtol=rtol, atol=atol)
            for o, s in zip(flat_original, onnx_outputs)
        )
    else:
        warnings.warn(
            f"Cannot validate output type: {type(original_output)}. "
            "Manual validation recommended.",
            stacklevel=2,
        )
        return False


def export_checkpoint_to_onnx(
    checkpoint_path: str | Path,
    save_path: str | Path,
    model_class: type[nn.Module],
    example_input: torch.Tensor,
    opset_version: int = 17,
    load_kwargs: dict[str, Any] | None = None,
    **export_kwargs: Any,
) -> str:
    """Export a Lightning checkpoint directly to ONNX.

    Convenience function that loads a checkpoint and exports it in one step.

    Args:
        checkpoint_path: Path to the PyTorch Lightning checkpoint (.ckpt).
        save_path: Path where the ONNX model will be saved (.onnx).
        model_class: The AutoTimm model class (e.g., ImageClassifier, ObjectDetector).
        example_input: Example input tensor for export.
        opset_version: ONNX opset version. Default is 17.
        load_kwargs: Additional kwargs to pass to model_class.load_from_checkpoint().
        **export_kwargs: Additional kwargs to pass to export_to_onnx().

    Returns:
        The save path as a string.

    Example:
        >>> from autotimm import ImageClassifier
        >>> from autotimm.export import export_checkpoint_to_onnx
        >>> import torch
        >>>
        >>> path = export_checkpoint_to_onnx(
        ...     checkpoint_path="model.ckpt",
        ...     save_path="model.onnx",
        ...     model_class=ImageClassifier,
        ...     example_input=torch.randn(1, 3, 224, 224),
        ... )
    """
    _check_onnx_deps()
    load_kwargs = load_kwargs or {}

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model = model_class.load_from_checkpoint(
        str(checkpoint_path), map_location="cpu", **load_kwargs
    )

    return export_to_onnx(
        model=model,
        save_path=save_path,
        example_input=example_input,
        opset_version=opset_version,
        **export_kwargs,
    )


__all__ = [
    "export_checkpoint_to_onnx",
    "export_checkpoint_to_torchscript",
    "export_to_onnx",
    "export_to_torchscript",
    "load_onnx",
    "load_torchscript",
    "validate_onnx_export",
    "validate_torchscript_export",
]
