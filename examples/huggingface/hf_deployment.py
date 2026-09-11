"""Example: Model Deployment & Export for HuggingFace Hub Models.

This example demonstrates production deployment techniques:
- ONNX export for cross-platform deployment
- Model quantization (dynamic and static INT8)
- TorchScript export for production PyTorch
- Inference optimization and benchmarking
- Basic serving with FastAPI (optional)

Usage:
    python examples/hf_deployment.py

Requirements:
    pip install onnx onnxruntime  # For ONNX export/inference
    pip install fastapi uvicorn   # Optional, for serving example
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from autotimm import ImageClassifier

# Optional imports
try:
    import onnx
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠ ONNX not available. Install: pip install onnx onnxruntime")

try:
    import uvicorn
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


def create_sample_input(batch_size: int = 1):
    """Create sample input tensor."""
    return torch.randn(batch_size, 3, 224, 224)


def example_1_onnx_export():
    """Example 1: Export model to ONNX format."""
    if not ONNX_AVAILABLE:
        print("\n⚠ Skipping ONNX export (onnx not installed)")
        return

    print("=" * 80)
    print("Example 1: ONNX Export")
    print("=" * 80)

    # Create model
    model = ImageClassifier(
        backbone="hf-hub:timm/mobilenetv3_small_100.lamb_in1k",
        num_classes=10,
    )
    model.eval()

    print("✓ Created MobileNet-V3 model")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Prepare for export
    dummy_input = create_sample_input(batch_size=1)
    output_path = Path("outputs/mobilenet_v3.onnx")
    output_path.parent.mkdir(exist_ok=True)

    print("\nExporting to ONNX...")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    print(f"✓ Exported to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Verify ONNX model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model validation passed")

    # Test ONNX inference
    print("\nTesting ONNX inference...")
    ort_session = ort.InferenceSession(str(output_path))

    # Prepare input
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}

    # Run inference
    start = time.time()
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_time = time.time() - start

    print("✓ ONNX inference successful")
    print(f"  Output shape: {ort_outputs[0].shape}")
    print(f"  Inference time: {onnx_time * 1000:.2f} ms")

    # Compare with PyTorch
    with torch.inference_mode():
        start = time.time()
        torch_output = model(dummy_input)
        torch_time = time.time() - start

    if isinstance(torch_output, dict):
        torch_output = torch_output.get(
            "logits", torch_output.get("output", next(iter(torch_output.values())))
        )

    print(f"  PyTorch inference time: {torch_time * 1000:.2f} ms")

    # Check numerical difference
    diff = np.abs(ort_outputs[0] - torch_output.numpy()).max()
    print(f"  Max difference: {diff:.6f}")

    print("\nONNX Export Benefits:")
    print("  • Cross-platform deployment (C++, C#, Java)")
    print("  • Hardware acceleration (CUDA, TensorRT, OpenVINO)")
    print("  • Framework-independent")
    print("  • Production-ready format")


def example_2_dynamic_quantization():
    """Example 2: Dynamic quantization for CPU inference."""
    print("\n" + "=" * 80)
    print("Example 2: Dynamic Quantization (INT8)")
    print("=" * 80)

    # Create model
    model = ImageClassifier(
        backbone="hf-hub:timm/resnet18.a1_in1k",
        num_classes=10,
    )
    model.eval()

    print("✓ Created ResNet-18 model")

    # Measure original model size
    original_path = Path("outputs/resnet18_fp32.pt")
    original_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), original_path)
    original_size = original_path.stat().st_size

    print(f"  Original size: {original_size / 1024 / 1024:.2f} MB")

    # Apply dynamic quantization
    print("\nApplying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8,
    )

    print("✓ Quantization complete")

    # Measure quantized model size
    quantized_path = Path("outputs/resnet18_int8.pt")
    torch.save(quantized_model.state_dict(), quantized_path)
    quantized_size = quantized_path.stat().st_size

    print(f"  Quantized size: {quantized_size / 1024 / 1024:.2f} MB")
    print(f"  Compression ratio: {original_size / quantized_size:.2f}x")

    # Benchmark inference speed
    print("\nBenchmarking inference speed...")
    dummy_input = create_sample_input()

    # Warmup
    for _ in range(10):
        with torch.inference_mode():
            _ = model(dummy_input)
            _ = quantized_model(dummy_input)

    # Benchmark original model
    num_runs = 50
    start = time.time()
    for _ in range(num_runs):
        with torch.inference_mode():
            _ = model(dummy_input)
    original_time = (time.time() - start) / num_runs

    # Benchmark quantized model
    start = time.time()
    for _ in range(num_runs):
        with torch.inference_mode():
            _ = quantized_model(dummy_input)
    quantized_time = (time.time() - start) / num_runs

    print(f"  FP32 model: {original_time * 1000:.2f} ms/image")
    print(f"  INT8 model: {quantized_time * 1000:.2f} ms/image")
    print(f"  Speedup: {original_time / quantized_time:.2f}x")

    print("\nDynamic Quantization Benefits:")
    print("  • 2-4x smaller model size")
    print("  • 1.5-2x faster CPU inference")
    print("  • No calibration data needed")
    print("  • Minimal accuracy loss (<1%)")
    print("  • Best for CPU deployment")


def example_3_torchscript_export():
    """Example 3: TorchScript export for production PyTorch."""
    print("\n" + "=" * 80)
    print("Example 3: TorchScript Export")
    print("=" * 80)

    # Create model
    model = ImageClassifier(
        backbone="hf-hub:timm/efficientnet_b0.ra_in1k",
        num_classes=10,
    )
    model.eval()

    print("✓ Created EfficientNet-B0 model")

    # Method 1: Tracing
    print("\nMethod 1: TorchScript Tracing")
    dummy_input = create_sample_input()

    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_path = Path("outputs/efficientnet_traced.pt")
        traced_path.parent.mkdir(exist_ok=True)
        traced_model.save(str(traced_path))

        print(f"✓ Traced model saved to: {traced_path}")
        print(f"  Size: {traced_path.stat().st_size / 1024 / 1024:.2f} MB")

        # Test traced model
        with torch.inference_mode():
            traced_model(dummy_input)
            original_output = model(dummy_input)

        if isinstance(original_output, dict):
            original_output = original_output.get(
                "logits",
                original_output.get("output", next(iter(original_output.values()))),
            )

        print("✓ Traced model inference successful")

    except Exception as e:
        print(f"⚠ Tracing failed: {e}")

    # Method 2: Scripting (more general but may fail for complex models)
    print("\nMethod 2: TorchScript Scripting")
    try:
        scripted_model = torch.jit.script(model)
        scripted_path = Path("outputs/efficientnet_scripted.pt")
        scripted_model.save(str(scripted_path))

        print(f"✓ Scripted model saved to: {scripted_path}")
        print(f"  Size: {scripted_path.stat().st_size / 1024 / 1024:.2f} MB")

    except Exception as e:
        print(f"⚠ Scripting failed (expected for some models): {e}")
        print("  Note: Tracing is more compatible but scripting is more general")

    print("\nTorchScript Benefits:")
    print("  • No Python dependency (C++ deployment)")
    print("  • Optimized execution graph")
    print("  • Mobile deployment (PyTorch Mobile)")
    print("  • Production PyTorch serving")


def example_4_inference_optimization():
    """Example 4: Inference optimization techniques."""
    print("\n" + "=" * 80)
    print("Example 4: Inference Optimization")
    print("=" * 80)

    model = ImageClassifier(
        backbone="hf-hub:timm/resnet18.a1_in1k",
        num_classes=10,
    )
    model.eval()

    print("✓ Created ResNet-18 model")

    # Technique 1: torch.no_grad()
    print("\n1. Disable gradient computation:")
    dummy_input = create_sample_input()

    # With gradients (slower)
    torch.set_grad_enabled(True)
    start = time.time()
    for _ in range(20):
        _ = model(dummy_input)
    time_with_grad = time.time() - start

    # Without gradients (faster)
    torch.set_grad_enabled(False)
    start = time.time()
    for _ in range(20):
        _ = model(dummy_input)
    time_no_grad = time.time() - start

    print(f"  With gradients:    {time_with_grad / 20 * 1000:.2f} ms")
    print(f"  Without gradients: {time_no_grad / 20 * 1000:.2f} ms")
    print(f"  Speedup: {time_with_grad / time_no_grad:.2f}x")

    # Technique 2: torch.compile() (PyTorch 2.0+)
    print("\n2. Model compilation (torch.compile):")
    if hasattr(torch, "compile"):
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")
            print("✓ Model compiled successfully")

            # Warmup compiled model
            for _ in range(5):
                with torch.inference_mode():
                    _ = compiled_model(dummy_input)

            # Benchmark
            start = time.time()
            for _ in range(20):
                with torch.inference_mode():
                    _ = compiled_model(dummy_input)
            time_compiled = time.time() - start

            print(f"  Regular model:  {time_no_grad / 20 * 1000:.2f} ms")
            print(f"  Compiled model: {time_compiled / 20 * 1000:.2f} ms")
            print(f"  Speedup: {time_no_grad / time_compiled:.2f}x")

        except Exception as e:
            print(f"⚠ Compilation failed: {e}")
    else:
        print("  ⚠ torch.compile not available (requires PyTorch 2.0+)")

    # Technique 3: Batch processing
    print("\n3. Batch processing:")
    batch_sizes = [1, 4, 8, 16]
    print(f"\n  {'Batch Size':>12} {'Time/Image (ms)':>18} {'Throughput (img/s)':>20}")
    print("  " + "-" * 52)

    for bs in batch_sizes:
        batch_input = create_sample_input(batch_size=bs)

        start = time.time()
        for _ in range(10):
            with torch.inference_mode():
                _ = model(batch_input)
        total_time = time.time() - start

        time_per_image = total_time / (10 * bs) * 1000
        throughput = (10 * bs) / total_time

        print(f"  {bs:>12} {time_per_image:>18.2f} {throughput:>20.1f}")

    print("\n  Note: Larger batches improve throughput (images/sec)")

    print("\nOptimization Summary:")
    print("  • Always use torch.no_grad() for inference")
    print("  • torch.compile() can give 1.5-2x speedup (PyTorch 2.0+)")
    print("  • Batch processing improves throughput")
    print("  • Quantization gives 2-4x speedup on CPU")
    print("  • ONNX + TensorRT gives 3-5x speedup on GPU")


def example_5_deployment_checklist():
    """Example 5: Production deployment checklist."""
    print("\n" + "=" * 80)
    print("Example 5: Production Deployment Checklist")
    print("=" * 80)

    checklist = {
        "1. Model Selection": [
            "Choose appropriate model size for latency requirements",
            "Consider accuracy vs speed trade-off",
            "Test on target hardware (CPU/GPU/edge device)",
        ],
        "2. Model Optimization": [
            "Export to ONNX for cross-platform deployment",
            "Apply quantization for CPU inference (2-4x speedup)",
            "Use TorchScript for PyTorch production serving",
            "Consider pruning for additional compression",
        ],
        "3. Inference Optimization": [
            "Always use torch.no_grad() or model.eval()",
            "Use torch.compile() if PyTorch 2.0+",
            "Batch requests when possible",
            "Profile and identify bottlenecks",
        ],
        "4. Serving Infrastructure": [
            "Use FastAPI/Flask for REST API",
            "Implement request batching",
            "Add caching for common requests",
            "Set up load balancing for scale",
        ],
        "5. Monitoring & Testing": [
            "Monitor latency (p50, p95, p99)",
            "Track accuracy on production data",
            "Implement A/B testing for model updates",
            "Set up alerts for degradation",
        ],
    }

    for category, items in checklist.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  ☐ {item}")

    print("\n" + "-" * 80)
    print("\nDeployment Patterns by Environment:")

    patterns = {
        "Cloud/Server (GPU)": [
            "Format: ONNX + TensorRT or PyTorch",
            "Optimization: Batch processing, FP16",
            "Serving: FastAPI + Gunicorn + NGINX",
            "Expected: 50-200 img/sec per GPU",
        ],
        "Cloud/Server (CPU)": [
            "Format: ONNX or TorchScript",
            "Optimization: INT8 quantization, batching",
            "Serving: FastAPI + Gunicorn",
            "Expected: 5-20 img/sec per core",
        ],
        "Edge Device (Raspberry Pi)": [
            "Format: TFLite or ONNX",
            "Optimization: INT8 quantization, small models",
            "Models: MobileNet, EfficientNet-Lite",
            "Expected: 1-5 img/sec",
        ],
        "Mobile (iOS/Android)": [
            "Format: CoreML (iOS) or TFLite (Android)",
            "Optimization: INT8/FP16, small models",
            "Models: MobileNetV3, EfficientNet-Lite",
            "Expected: 10-30 img/sec",
        ],
    }

    for env, details in patterns.items():
        print(f"\n{env}:")
        for detail in details:
            print(f"  • {detail}")


def example_6_fastapi_serving():
    """Example 6: Simple FastAPI serving (optional)."""
    if not FASTAPI_AVAILABLE:
        print("\n⚠ Skipping FastAPI example (fastapi not installed)")
        print("  Install with: pip install fastapi uvicorn python-multipart")
        return

    print("\n" + "=" * 80)
    print("Example 6: Model Serving with FastAPI")
    print("=" * 80)

    print("\nExample FastAPI server code:")
    print("-" * 80)

    example_code = '''
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
from autotimm import ImageClassifier
from autotimm.data import get_transforms

# Initialize FastAPI app
app = FastAPI(title="Image Classification API")

# Load model (do this once at startup)
model = ImageClassifier(
    backbone="hf-hub:timm/mobilenetv3_small_100.lamb_in1k",
    num_classes=1000,
)
model.eval()

# Get transforms
transform = get_transforms(
    image_size=224,
    preset="inference",
    backend="torchvision",
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict class for uploaded image."""
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.inference_mode():
        output = model(tensor)
        if isinstance(output, dict):
            output = output.get("logits", output.get("output", list(output.values())[0]))

    # Get top-5 predictions
    probs = torch.softmax(output, dim=1)
    top5_prob, top5_idx = torch.topk(probs, 5)

    results = [
        {"class": int(idx), "probability": float(prob)}
        for prob, idx in zip(top5_prob[0], top5_idx[0])
    ]

    return JSONResponse({"predictions": results})

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

# Run with: uvicorn filename:app --host 0.0.0.0 --port 8000
'''

    print(example_code)

    print("-" * 80)
    print("\nTo deploy this server:")
    print("  1. Save code to 'server.py'")
    print("  2. Run: uvicorn server:app --host 0.0.0.0 --port 8000")
    print("  3. Test: curl -X POST -F 'file=@image.jpg' http://localhost:8000/predict")

    print("\nProduction considerations:")
    print("  • Use Gunicorn for multiple workers")
    print("  • Add request validation and error handling")
    print("  • Implement request batching for throughput")
    print("  • Add authentication and rate limiting")
    print("  • Use NGINX as reverse proxy")


def main():
    """Run all examples."""
    # Create output directory
    Path("outputs").mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("Model Deployment & Export for Production")
    print("=" * 80)
    print("\nThis example demonstrates techniques for deploying models")
    print("to production environments.\n")

    # Run examples
    try:
        example_1_onnx_export()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        example_2_dynamic_quantization()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        example_3_torchscript_export()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    try:
        example_4_inference_optimization()
    except Exception as e:
        print(f"Example 4 failed: {e}")

    try:
        example_5_deployment_checklist()
    except Exception as e:
        print(f"Example 5 failed: {e}")

    try:
        example_6_fastapi_serving()
    except Exception as e:
        print(f"Example 6 failed: {e}")

    print("\n" + "=" * 80)
    print("Key Takeaways")
    print("=" * 80)
    print("\n1. Export formats for different platforms:")
    print("   • ONNX: Cross-platform, hardware-accelerated")
    print("   • TorchScript: PyTorch production, mobile")
    print("   • TFLite: Mobile (Android), edge devices")
    print("   • CoreML: iOS/macOS deployment")

    print("\n2. Optimization techniques:")
    print("   • Quantization: 2-4x smaller, 1.5-2x faster (CPU)")
    print("   • Pruning: Remove unnecessary connections")
    print("   • torch.compile(): 1.5-2x faster (PyTorch 2.0+)")
    print("   • Batching: Better GPU utilization")

    print("\n3. Deployment patterns:")
    print("   • Cloud GPU: ONNX + TensorRT, FP16")
    print("   • Cloud CPU: INT8 quantization, batching")
    print("   • Edge: Small models, INT8, TFLite")
    print("   • Mobile: MobileNet, CoreML/TFLite")

    print("\n4. Serving infrastructure:")
    print("   • FastAPI for REST API")
    print("   • Gunicorn for multiple workers")
    print("   • NGINX as reverse proxy")
    print("   • Docker for containerization")

    print("\nNext steps:")
    print("• Export your model to ONNX")
    print("• Apply quantization and measure speedup")
    print("• Set up FastAPI serving")
    print("• Profile and optimize for your hardware")
    print("• Monitor production performance")


if __name__ == "__main__":
    main()
