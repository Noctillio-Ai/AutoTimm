"""Factory for training metrics (torchmetrics)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torchmetrics
import torchmetrics.classification
import torchmetrics.detection


def _instantiate_metric(
    metric_cls: Any,
    params: dict[str, Any],
    auto_injected: set[str] | None = None,
) -> Any:
    """Construct ``metric_cls``, dropping auto-injected kwargs if rejected.

    ``num_classes`` / ``num_labels`` are injected for convenience, but many
    metrics (e.g. ``MeanAveragePrecision``) don't accept them.
    """
    try:
        return metric_cls(**params)
    except (TypeError, ValueError):
        if not auto_injected:
            raise
        filtered = {k: v for k, v in params.items() if k not in auto_injected}
        return metric_cls(**filtered)


@dataclass
class MetricConfig:
    """Configuration for a single metric.

    All parameters are required - no defaults are provided to ensure
    explicit configuration.

    Parameters:
        name: Unique identifier for this metric (used in logging).
        backend: Metric backend type. One of ``"torchmetrics"`` or ``"custom"``.
        metric_class: The metric class name (for torchmetrics/custom).
        params: Parameters passed to the metric constructor/function.
        stages: List of stages where this metric applies: ``"train"``,
            ``"val"``, ``"test"``.
        log_on_step: Whether to log on each step.
        log_on_epoch: Whether to log on epoch end.
        prog_bar: Whether to show in progress bar.

    Example:
        >>> config = MetricConfig(
        ...     name="accuracy",
        ...     backend="torchmetrics",
        ...     metric_class="Accuracy",
        ...     params={"task": "multiclass"},
        ...     stages=["train", "val", "test"],
        ...     prog_bar=True,
        ... )
    """

    name: str
    backend: str
    metric_class: str
    params: dict[str, Any]
    stages: list[str]
    log_on_step: bool = False
    log_on_epoch: bool = True
    prog_bar: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name is required")
        if not self.backend:
            raise ValueError("backend is required")
        if not self.metric_class:
            raise ValueError("metric_class is required")
        if not self.stages:
            raise ValueError("stages is required (e.g., ['train', 'val', 'test'])")

        self.backend = self.backend.lower()
        valid_backends = {"torchmetrics", "custom"}
        if self.backend not in valid_backends:
            raise ValueError(
                f"Unknown backend '{self.backend}'. "
                f"Valid backends: {', '.join(sorted(valid_backends))}"
            )

        valid_stages = {"train", "val", "test"}
        for stage in self.stages:
            if stage.lower() not in valid_stages:
                raise ValueError(
                    f"Unknown stage '{stage}'. "
                    f"Valid stages: {', '.join(sorted(valid_stages))}"
                )
        self.stages = [s.lower() for s in self.stages]


@dataclass
class LoggingConfig:
    """Configuration for enhanced logging during training.

    Parameters:
        log_learning_rate: Whether to log the current learning rate.
        log_gradient_norm: Whether to log gradient norms.
        log_weight_norm: Whether to log weight norms.
        log_confusion_matrix: Whether to log confusion matrix at epoch end.
        log_predictions: Whether to log sample predictions/images.
        predictions_per_epoch: Number of sample predictions to log per epoch.
        verbosity: Logging verbosity level (0=minimal, 1=normal, 2=verbose).

    Example:
        >>> config = LoggingConfig(
        ...     log_learning_rate=True,
        ...     log_gradient_norm=True,
        ...     log_confusion_matrix=True,
        ... )
    """

    log_learning_rate: bool
    log_gradient_norm: bool
    log_weight_norm: bool = False
    log_confusion_matrix: bool = False
    log_predictions: bool = False
    predictions_per_epoch: int = 8
    verbosity: int = 1

    def __post_init__(self) -> None:
        if self.verbosity not in (0, 1, 2):
            raise ValueError("verbosity must be 0, 1, or 2")
        if self.predictions_per_epoch < 0:
            raise ValueError("predictions_per_epoch must be non-negative")


class MetricManager:
    """Manages multiple metrics for training/validation/testing.

    This class creates and manages metric instances from explicit configurations.
    No default values are provided - all configuration must be specified.

    Parameters:
        configs: List of ``MetricConfig`` objects defining each metric.
        num_classes: Number of classes (required for classification metrics).

    Attributes:
        train_metrics: Dict of metrics for training stage.
        val_metrics: Dict of metrics for validation stage.
        test_metrics: Dict of metrics for test stage.

    Example:
        >>> manager = MetricManager(
        ...     configs=[
        ...         MetricConfig(
        ...             name="accuracy",
        ...             backend="torchmetrics",
        ...             metric_class="Accuracy",
        ...             params={"task": "multiclass"},
        ...             stages=["train", "val", "test"],
        ...             prog_bar=True,
        ...         ),
        ...         MetricConfig(
        ...             name="f1",
        ...             backend="torchmetrics",
        ...             metric_class="F1Score",
        ...             params={"task": "multiclass", "average": "macro"},
        ...             stages=["val"],
        ...         ),
        ...     ],
        ...     num_classes=10,
        ... )
        >>> train_metrics = manager.get_train_metrics()
    """

    def __init__(self, configs: list[MetricConfig], num_classes: int) -> None:
        if not configs:
            raise ValueError("At least one MetricConfig is required")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self._configs = configs
        self._num_classes = num_classes

        # Store metrics with their configs by stage
        self._train_metrics: dict[str, tuple[torch.nn.Module, MetricConfig]] = {}
        self._val_metrics: dict[str, tuple[torch.nn.Module, MetricConfig]] = {}
        self._test_metrics: dict[str, tuple[torch.nn.Module, MetricConfig]] = {}

        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize all metric instances from configs."""
        for config in self._configs:
            if "train" in config.stages:
                metric = self._create_metric(config)
                self._train_metrics[config.name] = (metric, config)
            if "val" in config.stages:
                metric = self._create_metric(config)
                self._val_metrics[config.name] = (metric, config)
            if "test" in config.stages:
                metric = self._create_metric(config)
                self._test_metrics[config.name] = (metric, config)

    def _create_metric(self, config: MetricConfig) -> torch.nn.Module:
        """Create a single metric instance from config."""
        backend = config.backend
        params = config.params.copy()

        # Auto-inject num_classes / num_labels if the user didn't provide them.
        # These are tracked separately so _create_torchmetrics_metric can
        # filter them out when the metric constructor doesn't accept them.
        auto_injected: set[str] = set()

        if "num_classes" not in params:
            params["num_classes"] = self._num_classes
            auto_injected.add("num_classes")

        if "num_labels" not in params:
            params["num_labels"] = self._num_classes
            auto_injected.add("num_labels")

        if backend == "torchmetrics":
            return self._create_torchmetrics_metric(
                config.metric_class, params, auto_injected
            )
        elif backend == "custom":
            return self._create_custom_metric(
                config.metric_class, params, auto_injected
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _create_torchmetrics_metric(
        self,
        metric_class: str,
        params: dict[str, Any],
        auto_injected: set[str] | None = None,
    ) -> torchmetrics.Metric:
        """Create a torchmetrics Metric instance."""
        # Try top-level torchmetrics first, then the task submodules that
        # don't re-export everything at the top level.
        metric_cls = getattr(torchmetrics, metric_class, None)
        if metric_cls is None:
            metric_cls = getattr(torchmetrics.classification, metric_class, None)
        if metric_cls is None:
            metric_cls = getattr(torchmetrics.detection, metric_class, None)
        if metric_cls is None:
            raise ValueError(
                f"Unknown torchmetrics metric: {metric_class}. "
                f"Check torchmetrics documentation for available metrics."
            )

        return _instantiate_metric(metric_cls, params, auto_injected)

    def _create_custom_metric(
        self,
        metric_class: str,
        params: dict[str, Any],
        auto_injected: set[str] | None = None,
    ) -> torch.nn.Module:
        """Create a custom metric from a fully qualified class path."""
        import importlib

        if "." not in metric_class:
            raise ValueError(
                f"custom metric_class must be a fully qualified path "
                f"(e.g., 'mypackage.metrics.CustomMetric'), got: {metric_class}"
            )

        module_path, class_name = metric_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        metric_cls = getattr(module, class_name)
        return _instantiate_metric(metric_cls, params, auto_injected)

    def get_train_metrics(self) -> torch.nn.ModuleDict:
        """Return ModuleDict of train metrics for Lightning module."""
        return torch.nn.ModuleDict(
            {name: metric for name, (metric, _) in self._train_metrics.items()}
        )

    def get_val_metrics(self) -> torch.nn.ModuleDict:
        """Return ModuleDict of validation metrics for Lightning module."""
        return torch.nn.ModuleDict(
            {name: metric for name, (metric, _) in self._val_metrics.items()}
        )

    def get_test_metrics(self) -> torch.nn.ModuleDict:
        """Return ModuleDict of test metrics for Lightning module."""
        return torch.nn.ModuleDict(
            {name: metric for name, (metric, _) in self._test_metrics.items()}
        )

    def get_metric_config(self, stage: str, name: str) -> MetricConfig | None:
        """Get the config for a specific metric.

        Parameters:
            stage: One of "train", "val", "test".
            name: The metric name.

        Returns:
            The MetricConfig if found, None otherwise.
        """
        metrics_dict = {
            "train": self._train_metrics,
            "val": self._val_metrics,
            "test": self._test_metrics,
        }.get(stage, {})

        if name in metrics_dict:
            return metrics_dict[name][1]
        return None

    @property
    def configs(self) -> list[MetricConfig]:
        """Return the configurations used to create the metrics."""
        return self._configs

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return self._num_classes

    def __len__(self) -> int:
        """Return total number of unique metric configs."""
        return len(self._configs)

    def __iter__(self):
        """Iterate over the metric configs."""
        return iter(self._configs)

    def __getitem__(self, index: int) -> MetricConfig:
        """Get a metric config by index."""
        return self._configs[index]

    def get_metric_by_name(
        self, name: str, stage: str | None = None
    ) -> torch.nn.Module | None:
        """Get a metric instance by name.

        Parameters:
            name: The metric name to search for.
            stage: Optional stage to search in ("train", "val", "test").
                If None, searches in order: val, train, test.

        Returns:
            The first matching metric instance, or None if not found.
        """
        if stage is not None:
            metrics_dict = {
                "train": self._train_metrics,
                "val": self._val_metrics,
                "test": self._test_metrics,
            }.get(stage, {})
            if name in metrics_dict:
                return metrics_dict[name][0]
            return None

        # Search in order: val, train, test
        for metrics_dict in [
            self._val_metrics,
            self._train_metrics,
            self._test_metrics,
        ]:
            if name in metrics_dict:
                return metrics_dict[name][0]
        return None

    def get_config_by_name(self, name: str) -> MetricConfig | None:
        """Get a metric config by name.

        Parameters:
            name: The metric name to search for.

        Returns:
            The matching MetricConfig, or None if not found.
        """
        for config in self._configs:
            if config.name == name:
                return config
        return None
