"""Export a trained checkpoint to TorchScript (JIT) format.

Invoked as: python -m autotimm.export.export_jit --checkpoint <path> --output <path> --task-class <name>

Delegates to :func:`autotimm.export.export_checkpoint_to_torchscript` for the
actual conversion so that Lightning-module wrapping, optimisation and
validation logic is shared with the library API.

Outputs the path to the saved .pt file on stdout.
"""

from __future__ import annotations

import argparse
import sys

import torch

from autotimm.export._export import export_checkpoint_to_torchscript
from autotimm.export._export_cli_common import (
    get_hparams as _get_hparams,
    build_load_overrides as _build_load_overrides,
    resolve_input_size as _resolve_input_size,
    resolve_task_class as _resolve_task_class,
)


def main():
    parser = argparse.ArgumentParser(description="Export checkpoint to TorchScript")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--output", required=True, help="Output .pt file path")
    parser.add_argument(
        "--task-class",
        default="ImageClassifier",
        help="Task class name (default: ImageClassifier)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input image size for tracing (default: 224)",
    )
    parser.add_argument(
        "--method",
        choices=["trace", "script"],
        default="trace",
        help="TorchScript export method (default: trace)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable torch.jit.optimize_for_inference",
    )
    parser.add_argument(
        "--hparams-yaml",
        default=None,
        help="Path to hparams.yaml (logs/<run_id>/hparams.yaml)",
    )
    args = parser.parse_args()

    try:
        cls = _resolve_task_class(args.task_class)

        # Read hparams from YAML file (preferred) or checkpoint fallback
        hp = _get_hparams(args.hparams_yaml, args.checkpoint)
        overrides = _build_load_overrides(hp)

        # Determine input size from model hparams if available
        input_size = _resolve_input_size(
            args.checkpoint, cls, args.input_size, args.hparams_yaml
        )
        example_input = torch.randn(1, 3, input_size, input_size)

        export_checkpoint_to_torchscript(
            checkpoint_path=args.checkpoint,
            save_path=args.output,
            model_class=cls,
            example_input=example_input,
            method=args.method,
            optimize=not args.no_optimize,
            load_kwargs=overrides,
        )

        print(str(args.output))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
