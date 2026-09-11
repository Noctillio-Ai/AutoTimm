"""AutoTimm: automated deep learning image tasks powered by timm and PyTorch Lightning.

Recommended import alias::

    import autotimm as at

    # Then use:
    model = at.ImageClassifier(backbone="resnet50", num_classes=10)
    trainer = at.AutoTrainer(max_epochs=10)
"""

import sys

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("autotimm")
except PackageNotFoundError:
    __version__ = "unknown"

# Import submodules for convenient access
from autotimm import data, heads, interpretation, losses, tasks
from autotimm.callbacks import JsonProgressCallback
from autotimm.cli import AutoTimmCLI
from autotimm.cli import main as cli_main
from autotimm.core import metrics as metrics_module
from autotimm.core.backbone import (
    BackboneConfig,
    FeatureBackboneConfig,
    ModelSource,
    create_backbone,
    create_feature_backbone,
    get_feature_channels,
    get_feature_info,
    get_feature_strides,
    get_model_source,
    list_backbones,
    list_hf_hub_backbones,
)
from autotimm.core.loggers import LoggerConfig, LoggerManager
from autotimm.core.logging import logger
from autotimm.core.metrics import LoggingConfig, MetricConfig, MetricManager
from autotimm.core.utils import (
    count_parameters,
    list_optimizers,
    list_schedulers,
    seed_everything,
)
from autotimm.data.datamodule import ImageDataModule
from autotimm.data.dataset import CSVImageDataset, MultiLabelImageDataset
from autotimm.data.detection_datamodule import DetectionDataModule
from autotimm.data.detection_dataset import CSVDetectionDataset
from autotimm.data.instance_datamodule import InstanceSegmentationDataModule
from autotimm.data.instance_dataset import CSVInstanceDataset
from autotimm.data.multilabel_datamodule import MultiLabelImageDataModule
from autotimm.data.preset_manager import (
    BackendRecommendation,
    compare_backends,
    recommend_backend,
)
from autotimm.data.segmentation_datamodule import SegmentationDataModule
from autotimm.data.timm_transforms import (
    create_inference_transform,
    get_transforms_from_backbone,
    resolve_backbone_data_config,
)
from autotimm.data.transform_config import TransformConfig, list_transform_presets
from autotimm.export import (
    export_checkpoint_to_onnx,
    export_checkpoint_to_torchscript,
    export_to_onnx,
    export_to_torchscript,
    load_onnx,
    load_torchscript,
    validate_onnx_export,
    validate_torchscript_export,
)
from autotimm.heads import (
    ASPP,
    FPN,
    ClassificationHead,
    DeepLabV3PlusHead,
    DetectionHead,
    FCNHead,
    MaskHead,
    YOLOXHead,
)

# Interpretation
from autotimm.interpretation import (
    AttentionFlow,
    AttentionRollout,
    ExplanationMetrics,
    FeatureMonitorCallback,
    FeatureVisualizer,
    GradCAM,
    GradCAMPlusPlus,
    IntegratedGradients,
    InteractiveVisualizer,
    InterpretationCallback,
    SmoothGrad,
    compare_methods,
    explain_detection,
    explain_prediction,
    explain_segmentation,
    visualize_batch,
)
from autotimm.losses import CenternessLoss, FCOSLoss, FocalLoss, GIoULoss
from autotimm.losses.segmentation import (
    CombinedSegmentationLoss,
    DiceLoss,
    FocalLossPixelwise,
    MaskLoss,
    TverskyLoss,
)
from autotimm.models import (
    get_yolox_model_info,
    list_yolox_backbones,
    list_yolox_heads,
    list_yolox_models,
    list_yolox_necks,
)
from autotimm.tasks.classification import ImageClassifier
from autotimm.tasks.instance_segmentation import InstanceSegmentor
from autotimm.tasks.object_detection import ObjectDetector
from autotimm.tasks.semantic_segmentation import SemanticSegmentor
from autotimm.tasks.yolox_detector import YOLOXDetector
from autotimm.training.trainer import AutoTrainer, TunerConfig

# Create module aliases by registering them in sys.modules
# This allows: import autotimm.loss, from autotimm.loss import DiceLoss, etc.
sys.modules["autotimm.loss"] = losses
sys.modules["autotimm.metric"] = metrics_module
sys.modules["autotimm.head"] = heads
sys.modules["autotimm.task"] = tasks

# Also make them available as attributes for: autotimm.loss.DiceLoss
loss = losses
metric = metrics_module
head = heads
task = tasks

__all__ = [
    # Heads
    "ASPP",
    "FPN",
    "AttentionFlow",
    "AttentionRollout",
    # CLI
    "AutoTimmCLI",
    # Trainer
    "AutoTrainer",
    # Backbone
    "BackboneConfig",
    # Preset manager
    "BackendRecommendation",
    "CSVDetectionDataset",
    # Datasets
    "CSVImageDataset",
    "CSVInstanceDataset",
    # Losses
    "CenternessLoss",
    "ClassificationHead",
    "CombinedSegmentationLoss",
    "DeepLabV3PlusHead",
    # Data
    "DetectionDataModule",
    "DetectionHead",
    "DiceLoss",
    "ExplanationMetrics",
    "FCNHead",
    "FCOSLoss",
    "FeatureBackboneConfig",
    "FeatureMonitorCallback",
    "FeatureVisualizer",
    "FocalLoss",
    "FocalLossPixelwise",
    "GIoULoss",
    # Interpretation
    "GradCAM",
    "GradCAMPlusPlus",
    # Tasks
    "ImageClassifier",
    "ImageDataModule",
    "InstanceSegmentationDataModule",
    "InstanceSegmentor",
    "IntegratedGradients",
    "InteractiveVisualizer",
    "InterpretationCallback",
    # Callbacks
    "JsonProgressCallback",
    # Logging & Metrics
    "LoggerConfig",
    "LoggerManager",
    "LoggingConfig",
    "MaskHead",
    "MaskLoss",
    "MetricConfig",
    "MetricManager",
    "ModelSource",
    "MultiLabelImageDataModule",
    "MultiLabelImageDataset",
    "ObjectDetector",
    "SegmentationDataModule",
    "SemanticSegmentor",
    "SmoothGrad",
    # Transform config
    "TransformConfig",
    "TunerConfig",
    "TverskyLoss",
    "YOLOXDetector",
    "YOLOXHead",
    "__version__",
    "cli_main",
    "compare_backends",
    "compare_methods",
    # Utils
    "count_parameters",
    "create_backbone",
    "create_feature_backbone",
    "create_inference_transform",
    # Submodules
    "data",
    "explain_detection",
    "explain_prediction",
    "explain_segmentation",
    "export_checkpoint_to_onnx",
    "export_checkpoint_to_torchscript",
    "export_to_onnx",
    # Export
    "export_to_torchscript",
    "get_feature_channels",
    "get_feature_info",
    "get_feature_strides",
    "get_model_source",
    "get_transforms_from_backbone",
    "get_yolox_model_info",
    "head",
    "heads",
    "interpretation",
    "list_backbones",
    "list_hf_hub_backbones",
    "list_optimizers",
    "list_schedulers",
    "list_transform_presets",
    "list_yolox_backbones",
    "list_yolox_heads",
    # YOLOX Utils
    "list_yolox_models",
    "list_yolox_necks",
    "load_onnx",
    "load_torchscript",
    # Logging
    "logger",
    # Submodule aliases
    "loss",
    "losses",
    "metric",
    "metrics_module",
    "recommend_backend",
    "resolve_backbone_data_config",
    "seed_everything",
    "task",
    "tasks",
    "validate_onnx_export",
    "validate_torchscript_export",
    "visualize_batch",
]
