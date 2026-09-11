"""Core modules for AutoTimm: backbone, logging, loggers, metrics, and utilities."""

from autotimm.core.backbone import (
    BackboneConfig,
    FeatureBackboneConfig,
    ModelSource,
    create_backbone,
    create_feature_backbone,
    get_backbone_out_features,
    get_feature_channels,
    get_feature_info,
    get_feature_strides,
    get_model_source,
    list_backbones,
    list_hf_hub_backbones,
)
from autotimm.core.loggers import LoggerConfig, LoggerManager
from autotimm.core.logging import log_table, logger
from autotimm.core.metrics import LoggingConfig, MetricConfig, MetricManager
from autotimm.core.utils import (
    count_parameters,
    list_optimizers,
    list_schedulers,
    seed_everything,
)

__all__ = [
    "BackboneConfig",
    "FeatureBackboneConfig",
    "LoggerConfig",
    "LoggerManager",
    "LoggingConfig",
    "MetricConfig",
    "MetricManager",
    "ModelSource",
    "count_parameters",
    "create_backbone",
    "create_feature_backbone",
    "get_backbone_out_features",
    "get_feature_channels",
    "get_feature_info",
    "get_feature_strides",
    "get_model_source",
    "list_backbones",
    "list_hf_hub_backbones",
    "list_optimizers",
    "list_schedulers",
    "log_table",
    "logger",
    "seed_everything",
]
