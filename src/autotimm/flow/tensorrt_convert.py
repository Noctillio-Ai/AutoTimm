"""Convert an ONNX model to a TensorRT engine.

Usage::

    python -m autotimm.flow.tensorrt_convert --onnx model.onnx --output model.engine
    # or
    autotimm-flow tensorrt-convert --onnx model.onnx --output model.engine
"""

from __future__ import annotations

import argparse
import json
import sys


def convert(onnx_path: str, engine_path: str, workspace_gb: int = 1) -> str:
    """Convert ONNX model to TensorRT engine. Returns the output path."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        errors = []
        for i in range(parser.num_errors):
            errors.append(str(parser.get_error(i)))
        raise RuntimeError("ONNX parsing failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError(
            "TensorRT engine build failed (build_serialized_network returned None). "
            "Check the TensorRT log output for details."
        )

    with open(engine_path, "wb") as f:
        f.write(engine)

    return engine_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT engine")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--output", required=True, help="Output .engine path")
    parser.add_argument(
        "--workspace-gb",
        type=int,
        default=1,
        help="Workspace memory in GB (default: 1)",
    )
    args = parser.parse_args()

    try:
        path = convert(args.onnx, args.output, args.workspace_gb)
        print(path)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
