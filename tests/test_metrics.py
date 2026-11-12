"""Tests for the metrics module."""

import pytest
import torchmetrics

from autotimm.core.metrics import (
    LoggingConfig,
    MetricConfig,
    MetricManager,
)


class TestMetricConfig:
    """Tests for MetricConfig dataclass."""

    def test_valid_torchmetrics_config(self):
        config = MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
        )
        assert config.name == "accuracy"
        assert config.backend == "torchmetrics"
        assert config.metric_class == "Accuracy"
        assert config.stages == ["train", "val", "test"]

    def test_case_insensitive_backend(self):
        config = MetricConfig(
            name="acc",
            backend="TorchMetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train"],
        )
        assert config.backend == "torchmetrics"

    def test_case_insensitive_stages(self):
        config = MetricConfig(
            name="acc",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["Train", "VAL", "Test"],
        )
        assert config.stages == ["train", "val", "test"]

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="name is required"):
            MetricConfig(
                name="",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={},
                stages=["train"],
            )

    def test_missing_backend_raises(self):
        with pytest.raises(ValueError, match="backend is required"):
            MetricConfig(
                name="acc",
                backend="",
                metric_class="Accuracy",
                params={},
                stages=["train"],
            )

    def test_missing_metric_class_raises(self):
        with pytest.raises(ValueError, match="metric_class is required"):
            MetricConfig(
                name="acc",
                backend="torchmetrics",
                metric_class="",
                params={},
                stages=["train"],
            )

    def test_missing_stages_raises(self):
        with pytest.raises(ValueError, match="stages is required"):
            MetricConfig(
                name="acc",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={},
                stages=[],
            )

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            MetricConfig(
                name="acc",
                backend="invalid",
                metric_class="Accuracy",
                params={},
                stages=["train"],
            )

    def test_invalid_stage_raises(self):
        with pytest.raises(ValueError, match="Unknown stage"):
            MetricConfig(
                name="acc",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={},
                stages=["invalid"],
            )

    def test_logging_options(self):
        config = MetricConfig(
            name="acc",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train"],
            log_on_step=True,
            log_on_epoch=False,
            prog_bar=True,
        )
        assert config.log_on_step is True
        assert config.log_on_epoch is False
        assert config.prog_bar is True


class TestLoggingConfig:
    """Tests for LoggingConfig dataclass."""

    def test_valid_config(self):
        config = LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=True,
            log_weight_norm=False,
            log_confusion_matrix=True,
            verbosity=1,
        )
        assert config.log_learning_rate is True
        assert config.log_gradient_norm is True
        assert config.log_confusion_matrix is True
        assert config.verbosity == 1

    def test_invalid_verbosity_raises(self):
        with pytest.raises(ValueError, match="verbosity must be 0, 1, or 2"):
            LoggingConfig(
                log_learning_rate=True,
                log_gradient_norm=False,
                verbosity=5,
            )

    def test_negative_predictions_per_epoch_raises(self):
        with pytest.raises(
            ValueError, match="predictions_per_epoch must be non-negative"
        ):
            LoggingConfig(
                log_learning_rate=True,
                log_gradient_norm=False,
                predictions_per_epoch=-1,
            )


class TestMetricManager:
    """Tests for MetricManager class."""

    def test_create_torchmetrics_accuracy(self):
        configs = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass"},
                stages=["train", "val", "test"],
            )
        ]
        manager = MetricManager(configs=configs, num_classes=10)

        assert len(manager) == 1
        train_metrics = manager.get_train_metrics()
        val_metrics = manager.get_val_metrics()
        test_metrics = manager.get_test_metrics()

        assert "accuracy" in train_metrics
        assert "accuracy" in val_metrics
        assert "accuracy" in test_metrics
        # torchmetrics.Accuracy with task="multiclass" returns MulticlassAccuracy
        assert isinstance(train_metrics["accuracy"], torchmetrics.Metric)

    def test_create_multiple_metrics(self):
        configs = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass"},
                stages=["train", "val"],
            ),
            MetricConfig(
                name="f1",
                backend="torchmetrics",
                metric_class="F1Score",
                params={"task": "multiclass", "average": "macro"},
                stages=["val"],
            ),
        ]
        manager = MetricManager(configs=configs, num_classes=10)

        train_metrics = manager.get_train_metrics()
        val_metrics = manager.get_val_metrics()

        assert len(train_metrics) == 1
        assert "accuracy" in train_metrics
        assert len(val_metrics) == 2
        assert "accuracy" in val_metrics
        assert "f1" in val_metrics

    def test_empty_configs_raises(self):
        with pytest.raises(ValueError, match="At least one MetricConfig is required"):
            MetricManager(configs=[], num_classes=10)

    def test_invalid_num_classes_raises(self):
        configs = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass"},
                stages=["train"],
            )
        ]
        with pytest.raises(ValueError, match="num_classes must be positive"):
            MetricManager(configs=configs, num_classes=0)

    def test_unknown_torchmetrics_raises(self):
        configs = [
            MetricConfig(
                name="unknown",
                backend="torchmetrics",
                metric_class="UnknownMetric",
                params={},
                stages=["train"],
            )
        ]
        with pytest.raises(ValueError, match="Unknown torchmetrics metric"):
            MetricManager(configs=configs, num_classes=10)

    def test_get_metric_config(self):
        configs = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass"},
                stages=["train", "val"],
                prog_bar=True,
            )
        ]
        manager = MetricManager(configs=configs, num_classes=10)

        config = manager.get_metric_config("train", "accuracy")
        assert config is not None
        assert config.prog_bar is True

        config = manager.get_metric_config("test", "accuracy")
        assert config is None

    def test_num_classes_auto_injected(self):
        configs = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass"},  # num_classes not specified
                stages=["train"],
            )
        ]
        manager = MetricManager(configs=configs, num_classes=10)
        train_metrics = manager.get_train_metrics()

        # Should work without error - num_classes was auto-injected
        assert "accuracy" in train_metrics

    def test_configs_property(self):
        configs = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass"},
                stages=["train"],
            )
        ]
        manager = MetricManager(configs=configs, num_classes=10)
        assert manager.configs == configs
        assert manager.num_classes == 10

    def test_iter(self):
        configs = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass"},
                stages=["train", "val"],
            ),
            MetricConfig(
                name="f1",
                backend="torchmetrics",
                metric_class="F1Score",
                params={"task": "multiclass", "average": "macro"},
                stages=["val"],
            ),
        ]
        manager = MetricManager(configs=configs, num_classes=10)

        iterated = list(manager)
        assert len(iterated) == 2
        assert iterated[0].name == "accuracy"
        assert iterated[1].name == "f1"

    def test_getitem(self):
        configs = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass"},
                stages=["train"],
            ),
            MetricConfig(
                name="f1",
                backend="torchmetrics",
                metric_class="F1Score",
                params={"task": "multiclass", "average": "macro"},
                stages=["val"],
            ),
        ]
        manager = MetricManager(configs=configs, num_classes=10)

        assert manager[0].name == "accuracy"
        assert manager[1].name == "f1"

    def test_get_metric_by_name(self):
        configs = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass"},
                stages=["train", "val"],
            ),
            MetricConfig(
                name="f1",
                backend="torchmetrics",
                metric_class="F1Score",
                params={"task": "multiclass", "average": "macro"},
                stages=["val"],
            ),
        ]
        manager = MetricManager(configs=configs, num_classes=10)

        # Get without stage (searches val first)
        accuracy = manager.get_metric_by_name("accuracy")
        assert accuracy is not None
        assert isinstance(accuracy, torchmetrics.Metric)

        # Get with specific stage
        accuracy_train = manager.get_metric_by_name("accuracy", stage="train")
        assert accuracy_train is not None

        # Get non-existent metric
        unknown = manager.get_metric_by_name("unknown")
        assert unknown is None

        # Get metric from wrong stage
        f1_train = manager.get_metric_by_name("f1", stage="train")
        assert f1_train is None

    def test_get_config_by_name(self):
        configs = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass"},
                stages=["train", "val"],
                prog_bar=True,
            ),
            MetricConfig(
                name="f1",
                backend="torchmetrics",
                metric_class="F1Score",
                params={"task": "multiclass", "average": "macro"},
                stages=["val"],
            ),
        ]
        manager = MetricManager(configs=configs, num_classes=10)

        config = manager.get_config_by_name("accuracy")
        assert config is not None
        assert config.name == "accuracy"
        assert config.prog_bar is True

        config = manager.get_config_by_name("f1")
        assert config is not None
        assert config.name == "f1"

        config = manager.get_config_by_name("unknown")
        assert config is None


class TestDetectionMetrics:
    """MeanAveragePrecision lives in torchmetrics.detection, not top level."""

    def _config(self, backend, metric_class):
        return MetricConfig(
            name="mAP",
            backend=backend,
            metric_class=metric_class,
            params={"box_format": "xyxy", "iou_type": "bbox"},
            stages=["val"],
        )

    def test_torchmetrics_backend_finds_detection_metric(self):
        from torchmetrics.detection import MeanAveragePrecision

        manager = MetricManager(
            configs=[self._config("torchmetrics", "MeanAveragePrecision")],
            num_classes=80,
        )
        assert isinstance(manager.get_val_metrics()["mAP"], MeanAveragePrecision)

    def test_custom_backend_drops_auto_injected_kwargs(self):
        from torchmetrics.detection import MeanAveragePrecision

        manager = MetricManager(
            configs=[
                self._config("custom", "torchmetrics.detection.MeanAveragePrecision")
            ],
            num_classes=80,
        )
        assert isinstance(manager.get_val_metrics()["mAP"], MeanAveragePrecision)
