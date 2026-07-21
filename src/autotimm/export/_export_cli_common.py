"""Shared checkpoint/hparams resolution helpers for the export CLIs.

Both :mod:`autotimm.export.export_jit` and :mod:`autotimm.export.export_onnx`
need to resolve a task class, read hparams from a YAML sidecar file (or fall
back to peeking into the checkpoint), and determine the model's input size
before tracing. That logic is identical between the two CLIs, so it lives
here once.
"""

from __future__ import annotations

import os

import yaml  # PyYAML — bundled with PyTorch Lightning

from autotimm.core.utils import safe_torch_load

TASK_CLASS_MAP = {
    "ImageClassifier": "autotimm.tasks.classification.ImageClassifier",
    "ObjectDetector": "autotimm.tasks.object_detection.ObjectDetector",
    "SemanticSegmentor": "autotimm.tasks.semantic_segmentation.SemanticSegmentor",
    "InstanceSegmentor": "autotimm.tasks.instance_segmentation.InstanceSegmentor",
    "YOLOXDetector": "autotimm.tasks.yolox_detector.YOLOXDetector",
}


def resolve_task_class(name: str):
    """Import and return the task class by name."""
    dotted = TASK_CLASS_MAP.get(name)
    if dotted is None:
        raise ValueError(f"Unknown task class: {name}. Valid: {list(TASK_CLASS_MAP)}")
    module_path, cls_name = dotted.rsplit(".", 1)
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def parse_hparams_yaml(path: str) -> dict:
    """Read an hparams.yaml file and return the parsed dict."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data


def build_load_overrides(hp: dict) -> dict:
    """Build ``load_from_checkpoint`` overrides from an hparams dict."""
    override: dict = {}
    backbone = hp.get("backbone") or hp.get("backbone_name")
    if backbone is not None:
        override["backbone"] = backbone
    model_name = hp.get("model_name")
    if model_name is not None:
        override["model_name"] = model_name
    num_classes = hp.get("num_classes")
    if num_classes is not None:
        override["num_classes"] = int(num_classes)
    override["compile_model"] = False
    return override


def get_hparams(hparams_yaml: str | None, checkpoint_path: str) -> dict:
    """Load hparams from YAML file, falling back to checkpoint peek."""
    if hparams_yaml and os.path.isfile(hparams_yaml):
        return parse_hparams_yaml(hparams_yaml)
    # Fallback: peek into checkpoint
    ckpt = safe_torch_load(checkpoint_path, map_location="cpu")
    hp = ckpt.get("hyper_parameters", {})
    if isinstance(hp, dict):
        return hp.get("init_args", hp)
    return {}


def resolve_input_size(
    checkpoint_path: str,
    task_class,
    default_size: int,
    hparams_yaml: str | None = None,
) -> int:
    """Determine image input size from hparams or fall back to *default_size*."""
    hp = get_hparams(hparams_yaml, checkpoint_path)
    overrides = build_load_overrides(hp)
    model = task_class.load_from_checkpoint(
        checkpoint_path, map_location="cpu", **overrides
    )
    if hasattr(model, "hparams"):
        img_size = getattr(model.hparams, "image_size", None)
        if img_size is not None:
            return img_size if isinstance(img_size, int) else img_size[0]
    return default_size
