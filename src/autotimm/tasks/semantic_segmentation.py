"""Semantic segmentation task as a PyTorch Lightning module."""

from __future__ import annotations

import datetime as _dt
import getpass
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from autotimm.core.backbone import (
    FeatureBackboneConfig,
    create_feature_backbone,
    get_feature_channels,
)
from autotimm.data.transform_config import TransformConfig
from autotimm.heads import DeepLabV3PlusHead, FCNHead
from autotimm.losses import get_loss_registry
from autotimm.losses.segmentation import CombinedSegmentationLoss, DiceLoss
from autotimm.core.metrics import LoggingConfig, MetricConfig, MetricManager
from autotimm.tasks.preprocessing_mixin import PreprocessingMixin
from autotimm.core.utils import seed_everything


class SemanticSegmentor(PreprocessingMixin, pl.LightningModule):
    """End-to-end semantic segmentation model backed by a timm backbone.

    Parameters:
        backbone: A timm model name (str) or a :class:`FeatureBackboneConfig`.
        num_classes: Number of segmentation classes.
        head_type: Type of segmentation head ('deeplabv3plus' or 'fcn').
        loss_fn: Loss function to use. Can be:
            - A string from the loss registry (e.g., 'dice', 'focal_pixelwise', 'combined_segmentation')
            - An instance of nn.Module (custom loss)
            - None (uses loss_type parameter for backward compatibility)
        loss_type: [DEPRECATED] Loss function type ('ce', 'dice', 'focal', or 'combined').
            Use loss_fn parameter instead for better flexibility.
        dice_weight: Weight for Dice loss when using 'combined' loss (default: 1.0).
        ce_weight: Weight for cross-entropy loss when using 'combined' loss (default: 1.0).
        ignore_index: Index to ignore in loss computation (default: 255).
        class_weights: Optional per-class weights for CE loss.
        metrics: A :class:`MetricManager` instance or list of :class:`MetricConfig`
            objects. Required - no default metrics are provided.
        logging_config: Optional :class:`LoggingConfig` for enhanced logging.
        transform_config: Optional :class:`TransformConfig` for unified transform
            configuration. When provided, enables the ``preprocess()`` method
            for inference-time preprocessing using model-specific normalization.
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        optimizer: Optimizer name or dict with 'class' and 'params' keys.
        optimizer_kwargs: Additional kwargs for the optimizer.
        scheduler: Scheduler name, dict with 'class' and 'params' keys, or None.
        scheduler_kwargs: Extra kwargs forwarded to the LR scheduler.
        freeze_backbone: If True, backbone parameters are frozen.
        compile_model: If ``True`` (default), apply ``torch.compile()`` to the backbone and head
            for faster inference and training. Requires PyTorch 2.0+.
        compile_kwargs: Optional dict of kwargs to pass to ``torch.compile()``.
            Common options: ``mode`` (``"default"``, ``"reduce-overhead"``, ``"max-autotune"``),
            ``fullgraph`` (``True``/``False``), ``dynamic`` (``True``/``False``).
        seed: Random seed for reproducibility. If ``None``, no seeding is performed.
            Default is ``42`` for reproducible results.
        deterministic: If ``True`` (default), enables deterministic algorithms in PyTorch for full
            reproducibility (may impact performance). Set to ``False`` for faster training.

    Example:
        >>> model = SemanticSegmentor(
        ...     backbone="resnet50",
        ...     num_classes=19,
        ...     head_type="deeplabv3plus",
        ...     metrics=[
        ...         MetricConfig(
        ...             name="iou",
        ...             backend="torchmetrics",
        ...             metric_class="JaccardIndex",
        ...             params={"task": "multiclass", "num_classes": 19, "average": "macro"},
        ...             stages=["val", "test"],
        ...             prog_bar=True,
        ...         ),
        ...     ],
        ...     loss_type="combined",
        ...     dice_weight=1.0,
        ... )
    """

    def __init__(
        self,
        backbone: str | FeatureBackboneConfig,
        num_classes: int,
        head_type: str = "deeplabv3plus",
        loss_fn: str | nn.Module | None = None,
        loss_type: str = "combined",
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        ignore_index: int = 255,
        class_weights: torch.Tensor | None = None,
        metrics: MetricManager | list[MetricConfig] | None = None,
        logging_config: LoggingConfig | None = None,
        transform_config: TransformConfig | None = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer: str | dict[str, Any] = "adamw",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: str | dict[str, Any] | None = "cosine",
        scheduler_kwargs: dict[str, Any] | None = None,
        freeze_backbone: bool = False,
        compile_model: bool = True,
        compile_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
        deterministic: bool = True,
    ):
        # Seed for reproducibility
        if seed is not None:
            seed_everything(seed, deterministic=deterministic)
        elif deterministic:
            import warnings

            warnings.warn(
                "deterministic=True has no effect when seed=None. "
                "Set a seed value to enable deterministic behavior.",
                stacklevel=2,
            )

        # Keep the full config for backbone creation, but normalize the
        # `backbone` local to a plain string so save_hyperparameters() stores
        # a serializable value for checkpoint round-trips.
        if isinstance(backbone, FeatureBackboneConfig):
            backbone_cfg: FeatureBackboneConfig | str = backbone
            backbone = backbone.model_name
        else:
            backbone = str(backbone)
            backbone_cfg = backbone

        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "metrics",
                "logging_config",
                "transform_config",
                "class_weights",
                "loss_fn",
            ]
        )
        self.hparams.update(
            {
                "backbone_name": backbone,
                "username": getpass.getuser(),
                "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            }
        )

        # Create feature backbone
        self.backbone = create_feature_backbone(backbone_cfg)
        in_channels_list = get_feature_channels(self.backbone)

        # Create segmentation head
        if head_type == "deeplabv3plus":
            self.head = DeepLabV3PlusHead(
                in_channels_list=in_channels_list,
                num_classes=num_classes,
            )
        elif head_type == "fcn":
            self.head = FCNHead(
                in_channels=in_channels_list[-1],
                num_classes=num_classes,
            )
        else:
            raise ValueError(
                f"Unknown head_type: {head_type}. Use 'deeplabv3plus' or 'fcn'."
            )

        # Create loss function
        self.loss_type = loss_type
        self.ignore_index = ignore_index

        # Priority: loss_fn > loss_type (for backward compatibility)
        if loss_fn is not None:
            # Use provided loss function
            if isinstance(loss_fn, str):
                # Get from registry
                registry = get_loss_registry()
                # Prepare kwargs for loss instantiation
                loss_kwargs = {"ignore_index": ignore_index}
                if loss_fn in ["dice", "tversky", "combined_segmentation"]:
                    loss_kwargs["num_classes"] = num_classes
                if loss_fn == "combined_segmentation":
                    loss_kwargs["ce_weight"] = ce_weight
                    loss_kwargs["dice_weight"] = dice_weight
                    loss_kwargs["class_weights"] = class_weights
                elif loss_fn == "cross_entropy" and class_weights is not None:
                    loss_kwargs["weight"] = class_weights

                self.criterion = registry.get_loss(loss_fn, **loss_kwargs)
            elif isinstance(loss_fn, nn.Module):
                # Use custom loss instance
                self.criterion = loss_fn
            else:
                raise TypeError(
                    f"loss_fn must be a string or nn.Module instance, got {type(loss_fn)}"
                )
        else:
            # Backward compatibility: use loss_type
            if loss_type == "ce":
                self.criterion = nn.CrossEntropyLoss(
                    weight=class_weights,
                    ignore_index=ignore_index,
                )
            elif loss_type == "dice":
                self.criterion = DiceLoss(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                )
            elif loss_type == "combined":
                self.criterion = CombinedSegmentationLoss(
                    num_classes=num_classes,
                    ce_weight=ce_weight,
                    dice_weight=dice_weight,
                    ignore_index=ignore_index,
                    class_weights=class_weights,
                )
            else:
                from autotimm.losses.segmentation import FocalLossPixelwise

                if loss_type == "focal":
                    self.criterion = FocalLossPixelwise(
                        ignore_index=ignore_index,
                    )
                else:
                    raise ValueError(
                        f"Unknown loss_type: {loss_type}. "
                        f"Use 'ce', 'dice', 'focal', or 'combined'."
                    )

        # Initialize metrics from config
        if metrics is None:
            # Create empty metric dicts if no metrics provided
            from torch.nn import ModuleDict

            self._metric_manager = None
            self.train_metrics = ModuleDict()
            self.val_metrics = ModuleDict()
            self.test_metrics = ModuleDict()
        elif isinstance(metrics, list):
            self._metric_manager = MetricManager(
                configs=metrics, num_classes=num_classes
            )
            # Register metrics as ModuleDicts for proper device handling
            self.train_metrics = self._metric_manager.get_train_metrics()
            self.val_metrics = self._metric_manager.get_val_metrics()
            self.test_metrics = self._metric_manager.get_test_metrics()
        else:
            self._metric_manager = metrics
            # Register metrics as ModuleDicts for proper device handling
            self.train_metrics = self._metric_manager.get_train_metrics()
            self.val_metrics = self._metric_manager.get_val_metrics()
            self.test_metrics = self._metric_manager.get_test_metrics()

        # Logging configuration
        self._logging_config = logging_config or LoggingConfig(
            log_learning_rate=False,
            log_gradient_norm=False,
        )

        # Store hyperparameters
        self._lr = lr
        self._weight_decay = weight_decay
        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._scheduler = scheduler
        self._scheduler_kwargs = scheduler_kwargs or {}
        self._num_classes = num_classes

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Apply torch.compile for optimization (PyTorch 2.0+)
        # Skip on MPS — the inductor backend generates invalid Metal shaders.
        # Skip on Windows — the inductor backend requires cl.exe (MSVC) which
        # is typically unavailable; compilation is deferred so the try/except
        # at init time cannot catch the error.
        import sys as _sys

        _mps_available = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        _skip_compile = _mps_available or _sys.platform == "win32"
        if compile_model and not _skip_compile:
            try:
                compile_opts = compile_kwargs or {}
                self.backbone = torch.compile(self.backbone, **compile_opts)
                self.head = torch.compile(self.head, **compile_opts)
            except Exception as e:
                import warnings

                warnings.warn(
                    f"torch.compile failed: {e}. Continuing without compilation. "
                    f"Ensure you have PyTorch 2.0+ for compile support.",
                    stacklevel=2,
                )
        elif compile_kwargs is not None:
            import warnings

            warnings.warn(
                "compile_kwargs is ignored when compile_model=False.",
                stacklevel=2,
            )

        # Setup transforms from config (PreprocessingMixin)
        self._setup_transforms(transform_config, task="segmentation")

    @property
    def num_classes(self) -> int:
        """Return the number of segmentation classes."""
        return self._num_classes

    def on_fit_start(self) -> None:
        """Capture batch_size from the datamodule when training begins."""
        if (
            self.trainer is not None
            and self.trainer.datamodule is not None
            and hasattr(self.trainer.datamodule, "batch_size")
        ):
            self.hparams["batch_size"] = self.trainer.datamodule.batch_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and head.

        Args:
            images: Input images [B, 3, H, W]

        Returns:
            Segmentation logits [B, num_classes, H', W']
        """
        features = self.backbone(images)
        logits = self.head(features)
        return logits

    def _compute_loss(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute segmentation loss.

        Args:
            logits: Predicted logits [B, C, H, W]
            masks: Ground truth masks [B, H, W]

        Returns:
            Loss value
        """
        # Resize logits to match mask size if needed
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return self.criterion(logits, masks)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Dict with 'image' [B, 3, H, W] and 'mask' [B, H, W]
            batch_idx: Batch index

        Returns:
            Loss value
        """
        images = batch["image"]
        masks = batch["mask"]

        logits = self(images)
        loss = self._compute_loss(logits, masks)

        # Get predictions
        preds = logits.argmax(dim=1)

        # Log loss
        self.log("train/loss", loss, prog_bar=True)

        # Update and log all train metrics
        if self._metric_manager is not None:
            for name, metric in self.train_metrics.items():
                config = self._metric_manager.get_metric_config("train", name)
                metric.update(preds, masks)
                if config:
                    self.log(
                        f"train/{name}",
                        metric,
                        on_step=config.log_on_step,
                        on_epoch=config.log_on_epoch,
                        prog_bar=config.prog_bar,
                    )

        # Enhanced logging: learning rate
        if self._logging_config.log_learning_rate:
            opt = self.optimizers()
            if opt is not None and hasattr(opt, "param_groups"):
                lr = opt.param_groups[0]["lr"]
                self.log("train/lr", lr, on_step=True, on_epoch=False)

        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        """Hook for gradient norm logging."""
        if self._logging_config.log_gradient_norm:
            grad_norm = self._compute_gradient_norm()
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False)

        if self._logging_config.log_weight_norm:
            weight_norm = self._compute_weight_norm()
            self.log("train/weight_norm", weight_norm, on_step=True, on_epoch=False)

    def _compute_gradient_norm(self) -> torch.Tensor:
        """Compute the total gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return torch.tensor(total_norm**0.5, device=self.device)

    def _compute_weight_norm(self) -> torch.Tensor:
        """Compute the total weight norm across all parameters."""
        total_norm = 0.0
        for p in self.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        return torch.tensor(total_norm**0.5, device=self.device)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Validation step.

        Args:
            batch: Dict with 'image' [B, 3, H, W] and 'mask' [B, H, W]
            batch_idx: Batch index
        """
        images = batch["image"]
        masks = batch["mask"]

        logits = self(images)
        loss = self._compute_loss(logits, masks)

        # Get predictions
        preds = logits.argmax(dim=1)

        try:
            is_sanity = getattr(self.trainer, "sanity_checking", False)
        except RuntimeError:
            is_sanity = False
        prefix = "sanity_val" if is_sanity else "val"

        # Log loss
        self.log(f"{prefix}/loss", loss, prog_bar=True)

        # Update and log all val metrics
        if self._metric_manager is not None:
            for name, metric in self.val_metrics.items():
                config = self._metric_manager.get_metric_config("val", name)
                metric.update(preds, masks)
                if config:
                    self.log(
                        f"{prefix}/{name}",
                        metric,
                        on_step=config.log_on_step,
                        on_epoch=config.log_on_epoch,
                        prog_bar=config.prog_bar,
                    )

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Test step.

        Args:
            batch: Dict with 'image' [B, 3, H, W] and 'mask' [B, H, W]
            batch_idx: Batch index
        """
        images = batch["image"]
        masks = batch["mask"]

        logits = self(images)
        loss = self._compute_loss(logits, masks)

        # Get predictions
        preds = logits.argmax(dim=1)

        # Log loss
        self.log("test/loss", loss)

        # Update and log all test metrics
        if self._metric_manager is not None:
            for name, metric in self.test_metrics.items():
                config = self._metric_manager.get_metric_config("test", name)
                metric.update(preds, masks)
                if config:
                    self.log(
                        f"test/{name}",
                        metric,
                        on_step=config.log_on_step,
                        on_epoch=config.log_on_epoch,
                        prog_bar=config.prog_bar,
                    )

    def predict(
        self, images: torch.Tensor, return_logits: bool = False
    ) -> torch.Tensor:
        """Predict segmentation masks for input images.

        Args:
            images: Input images [B, 3, H, W]
            return_logits: If True, return logits instead of class predictions

        Returns:
            Predicted class indices [B, H, W] or logits [B, C, H, W]
        """
        self.eval()
        with torch.inference_mode():
            logits = self(images)
            if return_logits:
                return logits
            return logits.argmax(dim=1)

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Prediction step for Trainer.predict().

        Args:
            batch: Input batch (dict or tensor)
            batch_idx: Batch index

        Returns:
            Predicted class indices [B, H, W]
        """
        if isinstance(batch, dict):
            images = batch["image"]
        elif isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch

        return self.predict(images)

    def to_onnx(
        self,
        save_path: str | None = None,
        example_input: torch.Tensor | None = None,
        opset_version: int = 17,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Export model to ONNX format.

        Args:
            save_path: Path to save the ONNX model. If None, uses a temp file.
            example_input: Example input tensor. If None, uses default shape (1, 3, 224, 224).
            opset_version: ONNX opset version. Default is 17.
            dynamic_axes: Dynamic axes specification. If None, batch dimension is dynamic.
            **kwargs: Additional arguments passed to export_to_onnx.

        Returns:
            Path to the saved ONNX model.

        Example:
            >>> model = SemanticSegmentor(backbone="resnet50", num_classes=21)
            >>> path = model.to_onnx("model.onnx")
        """
        from autotimm.export import export_to_onnx

        if example_input is None:
            example_input = torch.randn(1, 3, 224, 224)

        if save_path is None:
            import tempfile

            save_path = tempfile.mktemp(suffix=".onnx")

        return export_to_onnx(
            self,
            save_path,
            example_input,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            **kwargs,
        )

    def configure_optimizers(self) -> dict:
        """Configure optimizer and learning rate scheduler.

        Supports torch.optim, timm optimizers, and custom optimizers/schedulers.
        """
        params = filter(lambda p: p.requires_grad, self.parameters())

        # Create optimizer
        optimizer = self._create_optimizer(params)

        # Return early if no scheduler
        if self._scheduler is None or self._scheduler == "none":
            return {"optimizer": optimizer}

        # Create scheduler
        scheduler_config = self._create_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def _create_optimizer(self, params) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        # Prepare base kwargs
        opt_kwargs = {"lr": self._lr, "weight_decay": self._weight_decay}
        opt_kwargs.update(self._optimizer_kwargs)

        # Dict config: {"class": "path.to.Optimizer", "params": {...}}
        if isinstance(self._optimizer, dict):
            opt_class_path = self._optimizer["class"]
            opt_params = self._optimizer.get("params", {})
            opt_kwargs.update(opt_params)

            # Import and instantiate
            optimizer_cls = self._import_class(opt_class_path)
            return optimizer_cls(params, **opt_kwargs)

        # String name: try torch.optim first, then timm
        optimizer_name = self._optimizer.lower()

        # Torch optimizers
        torch_optimizers = {
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
            "adagrad": torch.optim.Adagrad,
        }

        if optimizer_name in torch_optimizers:
            return torch_optimizers[optimizer_name](params, **opt_kwargs)

        # Try timm optimizers (looked up lazily so a missing class only
        # affects the optimizer that needs it)
        try:
            import timm.optim as timm_optim

            timm_optimizer_names = {
                "adamp": "AdamP",
                "sgdp": "SGDP",
                "adabelief": "AdaBelief",
                "radam": "RAdam",
                "lamb": "Lamb",
                "madgrad": "MADGRAD",
                "novograd": "NvNovoGrad",
            }

            if optimizer_name in timm_optimizer_names:
                optimizer_cls = getattr(
                    timm_optim, timm_optimizer_names[optimizer_name], None
                )
                if optimizer_cls is not None:
                    return optimizer_cls(params, **opt_kwargs)
        except ImportError:
            pass

        raise ValueError(
            f"Unknown optimizer: {self._optimizer}. "
            f"Use a torch.optim optimizer name, timm optimizer name, "
            f"or provide a dict with 'class' and 'params' keys."
        )

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> dict:
        """Create scheduler config from optimizer."""
        sched_kwargs = self._scheduler_kwargs.copy()
        interval = sched_kwargs.pop("interval", "step")
        frequency = sched_kwargs.pop("frequency", 1)

        # Dict config: {"class": "path.to.Scheduler", "params": {...}}
        if isinstance(self._scheduler, dict):
            sched_class_path = self._scheduler["class"]
            sched_params = self._scheduler.get("params", {})
            sched_kwargs.update(sched_params)

            scheduler_cls = self._import_class(sched_class_path)
            scheduler = scheduler_cls(optimizer, **sched_kwargs)

            return {
                "scheduler": scheduler,
                "interval": interval,
                "frequency": frequency,
            }

        # String name: try torch schedulers first, then timm
        scheduler_name = self._scheduler.lower()

        # Torch schedulers
        if scheduler_name == "cosine":
            # Get T_max from kwargs, or use trainer's estimated_stepping_batches if available
            try:
                default_t_max = self.trainer.estimated_stepping_batches
            except RuntimeError:
                # Trainer not attached yet
                default_t_max = 1000

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_kwargs.pop("T_max", default_t_max),
                **sched_kwargs,
            )
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_kwargs.pop("step_size", 30),
                gamma=sched_kwargs.pop("gamma", 0.1),
                **sched_kwargs,
            )
        elif scheduler_name == "multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=sched_kwargs.pop("milestones", [30, 60, 90]),
                gamma=sched_kwargs.pop("gamma", 0.1),
                **sched_kwargs,
            )
        elif scheduler_name == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=sched_kwargs.pop("gamma", 0.95),
                **sched_kwargs,
            )
        elif scheduler_name == "onecycle":
            try:
                default_total_steps = self.trainer.estimated_stepping_batches
            except RuntimeError:
                # Trainer not attached yet
                default_total_steps = 1000

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=sched_kwargs.pop("max_lr", self._lr * 10),
                total_steps=sched_kwargs.pop("total_steps", default_total_steps),
                **sched_kwargs,
            )
        elif scheduler_name == "plateau":
            monitor = sched_kwargs.pop("monitor", "val/loss")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=sched_kwargs.pop("mode", "min"),
                factor=sched_kwargs.pop("factor", 0.1),
                patience=sched_kwargs.pop("patience", 10),
                **sched_kwargs,
            )
            interval = "epoch"
            return {
                "scheduler": scheduler,
                "monitor": monitor,
                "interval": interval,
                "frequency": frequency,
            }
        else:
            raise ValueError(
                f"Unknown scheduler: {self._scheduler}. "
                f"Use a torch.optim.lr_scheduler name or provide a dict."
            )

        return {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": frequency,
        }

    def _import_class(self, class_path: str):
        """Import a class from a fully qualified path."""
        import importlib

        if "." not in class_path:
            raise ValueError(
                f"Class path must be fully qualified (e.g., 'torch.optim.Adam'), "
                f"got: {class_path}"
            )

        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
