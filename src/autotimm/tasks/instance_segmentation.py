"""Instance segmentation task as a PyTorch Lightning module."""

from __future__ import annotations

from contextlib import contextmanager
import datetime as _dt
import getpass
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

from autotimm.core.backbone import (
    FeatureBackboneConfig,
    create_feature_backbone,
    get_feature_channels,
)
from autotimm.data.transform_config import TransformConfig
from autotimm.heads import FPN, DetectionHead, MaskHead
from autotimm.losses import FocalLoss, GIoULoss, get_loss_registry
from autotimm.losses.segmentation import MaskLoss
from autotimm.core.metrics import LoggingConfig, MetricConfig, MetricManager
from autotimm.tasks.preprocessing_mixin import PreprocessingMixin
from autotimm.core.utils import seed_everything
from autotimm.tasks._fcos_targets import (
    compute_fcos_detection_loss,
    decode_fcos_detections,
)


@contextmanager
def _temporary_eval_mode(module: nn.Module):
    """Run inference without leaking an evaluation-mode state change."""
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


class InstanceSegmentor(PreprocessingMixin, pl.LightningModule):
    """End-to-end instance segmentation model.

    Combines FCOS-style detection with per-instance mask prediction.

    Architecture: timm backbone → FPN → Detection Head + Mask Head → NMS

    Parameters:
        backbone: A timm model name (str) or a :class:`FeatureBackboneConfig`.
        num_classes: Number of object classes (excluding background).
        cls_loss_fn: Classification loss function. Can be:
            - A string from the loss registry (e.g., 'focal')
            - An instance of nn.Module (custom loss)
            - None (uses FocalLoss with focal_alpha and focal_gamma)
        reg_loss_fn: Regression loss function. Can be:
            - A string from the loss registry (e.g., 'giou')
            - An instance of nn.Module (custom loss)
            - None (uses GIoULoss)
        mask_loss_fn: Mask loss function. Can be:
            - A string from the loss registry (e.g., 'mask')
            - An instance of nn.Module (custom loss)
            - None (uses MaskLoss)
        metrics: A :class:`MetricManager` instance or list of :class:`MetricConfig`
            objects. Optional - if not provided, uses MeanAveragePrecision with mask support.
        logging_config: Optional :class:`LoggingConfig` for enhanced logging.
        transform_config: Optional :class:`TransformConfig` for unified transform
            configuration. When provided, enables the ``preprocess()`` method
            for inference-time preprocessing using model-specific normalization.
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        optimizer: Optimizer name or dict with 'class' and 'params' keys.
        optimizer_kwargs: Additional kwargs for the optimizer.
        scheduler: Scheduler name, dict config, or None for no scheduler.
        scheduler_kwargs: Extra kwargs forwarded to the LR scheduler.
        fpn_channels: Number of channels in FPN layers.
        head_num_convs: Number of conv layers in detection head branches.
        mask_size: ROI mask resolution (default: 28).
        mask_loss_weight: Weight for mask loss.
        focal_alpha: Alpha parameter for focal loss.
        focal_gamma: Gamma parameter for focal loss.
        cls_loss_weight: Weight for classification loss.
        reg_loss_weight: Weight for regression loss.
        centerness_loss_weight: Weight for centerness loss.
        score_thresh: Score threshold for detections during inference.
        nms_thresh: IoU threshold for NMS.
        max_detections_per_image: Maximum detections to keep per image.
        freeze_backbone: If True, backbone parameters are frozen.
        roi_pool_size: Size of ROI pooling output (default: 14).
        mask_threshold: Threshold for binarizing predicted masks (default: 0.5).
        strides: FPN output strides. Default (8, 16, 32, 64, 128) for P3-P7.
        regress_ranges: FCOS regression ranges for each FPN level. Defaults to
            the standard FCOS P3-P7 ranges when ``None``.
        compile_model: If ``True`` (default), apply ``torch.compile()`` to the backbone, FPN, and heads
            for faster inference and training. Requires PyTorch 2.0+.
        compile_kwargs: Optional dict of kwargs to pass to ``torch.compile()``.
            Common options: ``mode`` (``"default"``, ``"reduce-overhead"``, ``"max-autotune"``),
            ``fullgraph`` (``True``/``False``), ``dynamic`` (``True``/``False``).
        seed: Random seed for reproducibility. If ``None``, no seeding is performed.
            Default is ``42`` for reproducible results.
        deterministic: If ``True`` (default), enables deterministic algorithms in PyTorch for full
            reproducibility (may impact performance). Set to ``False`` for faster training.

    Example:
        >>> model = InstanceSegmentor(
        ...     backbone="resnet50",
        ...     num_classes=80,
        ...     metrics=[
        ...         MetricConfig(
        ...             name="mask_mAP",
        ...             backend="torchmetrics",
        ...             metric_class="MeanAveragePrecision",
        ...             params={"box_format": "xyxy", "iou_type": "segm"},
        ...             stages=["val", "test"],
        ...         ),
        ...     ],
        ...     lr=1e-4,
        ...     mask_loss_weight=1.0,
        ... )
    """

    def __init__(
        self,
        backbone: str | FeatureBackboneConfig,
        num_classes: int,
        cls_loss_fn: str | nn.Module | None = None,
        reg_loss_fn: str | nn.Module | None = None,
        mask_loss_fn: str | nn.Module | None = None,
        metrics: MetricManager | list[MetricConfig] | None = None,
        logging_config: LoggingConfig | None = None,
        transform_config: TransformConfig | None = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer: str | dict[str, Any] = "adamw",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: str | dict[str, Any] | None = "cosine",
        scheduler_kwargs: dict[str, Any] | None = None,
        fpn_channels: int = 256,
        head_num_convs: int = 4,
        mask_size: int = 28,
        mask_loss_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cls_loss_weight: float = 1.0,
        reg_loss_weight: float = 1.0,
        centerness_loss_weight: float = 1.0,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        max_detections_per_image: int = 100,
        freeze_backbone: bool = False,
        roi_pool_size: int = 14,
        mask_threshold: float = 0.5,
        strides: tuple[int, ...] = (8, 16, 32, 64, 128),
        regress_ranges: tuple[tuple[int, int], ...] | None = None,
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

        # Normalize backbone to a plain string so it survives checkpoint
        # round-trip (FeatureBackboneConfig is not serialised by save_hyperparameters).
        backbone = (
            backbone.model_name if hasattr(backbone, "model_name") else str(backbone)
        )

        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "metrics",
                "logging_config",
                "transform_config",
                "cls_loss_fn",
                "reg_loss_fn",
                "mask_loss_fn",
            ]
        )
        self.hparams.update(
            {
                "backbone_name": backbone,
                "username": getpass.getuser(),
                "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            }
        )

        self.num_classes = num_classes
        self._lr = lr
        self._weight_decay = weight_decay
        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._scheduler = scheduler
        self._scheduler_kwargs = scheduler_kwargs or {}

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.max_detections_per_image = max_detections_per_image
        self.mask_size = mask_size
        self.mask_threshold = mask_threshold
        self.roi_pool_size = roi_pool_size
        self.strides = strides

        # Default regression ranges for FCOS (P3-P7)
        if regress_ranges is None:
            self.regress_ranges = (
                (-1, 64),
                (64, 128),
                (128, 256),
                (256, 512),
                (512, float("inf")),
            )
        else:
            self.regress_ranges = regress_ranges

        # Build model. Use 3 backbone feature levels (C3, C4, C5) which,
        # combined with 2 extra FPN levels, gives P3-P7 (5 levels total) —
        # matching `strides`/`regress_ranges` above and ObjectDetector's FCOS
        # convention. `backbone` here is always a plain model-name string (see
        # the normalization at the top of __init__), so this always applies.
        backbone_cfg = FeatureBackboneConfig(model_name=backbone, out_indices=(2, 3, 4))
        self.backbone = create_feature_backbone(backbone_cfg)
        in_channels = get_feature_channels(self.backbone)

        self.fpn = FPN(
            in_channels_list=in_channels,
            out_channels=fpn_channels,
            num_extra_levels=2,
        )

        # Detection head
        self.detection_head = DetectionHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_convs=head_num_convs,
            prior_prob=0.01,
        )

        # Mask head
        self.mask_head = MaskHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_convs=4,
            mask_size=mask_size,
        )

        # Losses
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.centerness_loss_weight = centerness_loss_weight
        self.mask_loss_weight = mask_loss_weight

        # Setup classification loss
        if cls_loss_fn is not None:
            if isinstance(cls_loss_fn, str):
                registry = get_loss_registry()
                loss_kwargs = {"reduction": "sum"}
                if cls_loss_fn == "focal":
                    loss_kwargs["alpha"] = focal_alpha
                    loss_kwargs["gamma"] = focal_gamma
                self.focal_loss = registry.get_loss(cls_loss_fn, **loss_kwargs)
            elif isinstance(cls_loss_fn, nn.Module):
                self.focal_loss = cls_loss_fn
            else:
                raise TypeError(
                    f"cls_loss_fn must be a string or nn.Module instance, got {type(cls_loss_fn)}"
                )
        else:
            self.focal_loss = FocalLoss(
                alpha=focal_alpha, gamma=focal_gamma, reduction="sum"
            )

        # Setup regression loss
        if reg_loss_fn is not None:
            if isinstance(reg_loss_fn, str):
                registry = get_loss_registry()
                self.giou_loss = registry.get_loss(reg_loss_fn, reduction="sum")
            elif isinstance(reg_loss_fn, nn.Module):
                self.giou_loss = reg_loss_fn
            else:
                raise TypeError(
                    f"reg_loss_fn must be a string or nn.Module instance, got {type(reg_loss_fn)}"
                )
        else:
            self.giou_loss = GIoULoss(reduction="sum")

        # Setup mask loss
        if mask_loss_fn is not None:
            if isinstance(mask_loss_fn, str):
                registry = get_loss_registry()
                self.mask_loss_fn = registry.get_loss(mask_loss_fn, reduction="mean")
            elif isinstance(mask_loss_fn, nn.Module):
                self.mask_loss_fn = mask_loss_fn
            else:
                raise TypeError(
                    f"mask_loss_fn must be a string or nn.Module instance, got {type(mask_loss_fn)}"
                )
        else:
            self.mask_loss_fn = MaskLoss(reduction="mean")

        # Initialize metrics
        if metrics is None:
            # Default: use mask mAP for validation and test
            metrics = [
                MetricConfig(
                    name="mask_mAP",
                    backend="torchmetrics",
                    metric_class="MeanAveragePrecision",
                    params={"box_format": "xyxy", "iou_type": "segm"},
                    stages=["val", "test"],
                    prog_bar=True,
                ),
            ]

        if isinstance(metrics, list):
            self._metric_configs = metrics
            self._use_metric_manager = False
        else:
            self._metric_manager = metrics
            self._use_metric_manager = True

        # Register metrics
        self._register_detection_metrics()

        # Logging configuration
        self._logging_config = logging_config or LoggingConfig(
            log_learning_rate=False,
            log_gradient_norm=False,
        )

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
                self.fpn = torch.compile(self.fpn, **compile_opts)
                self.detection_head = torch.compile(self.detection_head, **compile_opts)
                self.mask_head = torch.compile(self.mask_head, **compile_opts)
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

    def _register_detection_metrics(self):
        """Register detection/instance segmentation metrics."""
        import torchmetrics
        import torchmetrics.detection

        self.val_metrics = nn.ModuleDict()
        self.test_metrics = nn.ModuleDict()

        if self._use_metric_manager:
            self.val_metrics = self._metric_manager.get_val_metrics()
            self.test_metrics = self._metric_manager.get_test_metrics()
        else:
            for config in self._metric_configs:
                if config.backend == "torchmetrics":
                    if hasattr(torchmetrics.detection, config.metric_class):
                        metric_cls = getattr(
                            torchmetrics.detection, config.metric_class
                        )
                    elif hasattr(torchmetrics, config.metric_class):
                        metric_cls = getattr(torchmetrics, config.metric_class)
                    else:
                        raise ValueError(f"Unknown metric: {config.metric_class}")

                    if "val" in config.stages:
                        self.val_metrics[config.name] = metric_cls(**config.params)
                    if "test" in config.stages:
                        self.test_metrics[config.name] = metric_cls(**config.params)

    def on_fit_start(self) -> None:
        """Capture batch_size from the datamodule when training begins."""
        if (
            self.trainer is not None
            and self.trainer.datamodule is not None
            and hasattr(self.trainer.datamodule, "batch_size")
        ):
            self.hparams["batch_size"] = self.trainer.datamodule.batch_size

    def forward(
        self, images: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass through detector (without mask head).

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Tuple of (cls_outputs, reg_outputs, centerness_outputs) per FPN level
        """
        features = self.backbone(images)
        fpn_features = self.fpn(features)
        cls_outputs, reg_outputs, centerness_outputs = self.detection_head(fpn_features)
        return cls_outputs, reg_outputs, centerness_outputs

    def _compute_detection_loss(
        self,
        cls_outputs: list[torch.Tensor],
        reg_outputs: list[torch.Tensor],
        centerness_outputs: list[torch.Tensor],
        target_boxes: list[torch.Tensor],
        target_labels: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute FCOS detection losses via per-level target assignment.

        Returns:
            Tuple of (cls_loss, reg_loss, centerness_loss), each already
            normalized by the number of positive samples.
        """
        losses = compute_fcos_detection_loss(
            cls_outputs,
            reg_outputs,
            centerness_outputs,
            target_boxes,
            target_labels,
            self.strides,
            self.regress_ranges,
            self.focal_loss,
            self.num_classes,
        )
        return losses["cls_loss"], losses["reg_loss"], losses["centerness_loss"]

    def _compute_mask_loss(
        self,
        fpn_features: list[torch.Tensor],
        pred_boxes: list[torch.Tensor],
        target_boxes: list[torch.Tensor],
        target_labels: list[torch.Tensor],
        target_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute mask prediction loss.

        Args:
            fpn_features: FPN feature pyramids
            pred_boxes: Predicted boxes for each image (list of [N, 4])
            target_boxes: Target boxes for each image (list of [M, 4])
            target_labels: Target labels for each image (list of [M])
            target_masks: Target masks for each image (list of [M, H, W])

        Returns:
            Mask loss value
        """
        device = fpn_features[0].device
        batch_size = len(target_boxes)

        # Use FPN level 0 (P3) for ROI pooling
        feature_map = fpn_features[0]

        all_rois = []
        all_target_masks = []
        all_labels = []

        for i in range(batch_size):
            boxes = target_boxes[i]
            labels = target_labels[i]
            masks = target_masks[i]

            if len(boxes) == 0:
                continue

            # Add batch index to boxes for ROI pooling
            batch_indices = torch.full(
                (len(boxes), 1), i, dtype=boxes.dtype, device=device
            )
            rois = torch.cat([batch_indices, boxes], dim=1)  # [N, 5]

            all_rois.append(rois)
            all_target_masks.append(masks)
            all_labels.append(labels)

        if len(all_rois) == 0:
            # Keep the zero loss attached to the graph so an all-empty batch
            # remains safe to backpropagate even when no other loss contributes.
            return feature_map.sum() * 0.0

        # Concatenate all ROIs
        rois = torch.cat(all_rois, dim=0)  # [total_N, 5]
        target_masks_batch = torch.cat(all_target_masks, dim=0)  # [total_N, H, W]
        labels_batch = torch.cat(all_labels, dim=0)  # [total_N]

        # ROI Align
        roi_features = ops.roi_align(
            feature_map,
            rois,
            output_size=(self.roi_pool_size, self.roi_pool_size),
            spatial_scale=1.0 / 8,  # Assuming P3 has stride 8
            aligned=True,
        )

        # Predict masks
        mask_logits = self.mask_head(
            roi_features
        )  # [total_N, num_classes, mask_size, mask_size]

        # Select masks for ground truth classes
        N = len(labels_batch)
        indices = torch.arange(N, device=device)
        mask_logits = mask_logits[
            indices, labels_batch
        ]  # [total_N, mask_size, mask_size]

        # Resize target masks to match prediction size
        target_masks_resized = F.interpolate(
            target_masks_batch.unsqueeze(1).float(),
            size=(self.mask_size, self.mask_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # [total_N, mask_size, mask_size]

        # Compute mask loss
        loss = self.mask_loss_fn(mask_logits, target_masks_resized)

        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Dict with 'image', 'boxes', 'labels', 'masks'
            batch_idx: Batch index

        Returns:
            Total loss value
        """
        images = batch["image"]
        target_boxes = batch["boxes"]
        target_labels = batch["labels"]
        target_masks = batch["masks"]

        # Get features
        features = self.backbone(images)
        fpn_features = self.fpn(features)

        # Detection forward
        cls_outputs, reg_outputs, centerness_outputs = self.detection_head(fpn_features)

        # Compute detection losses
        cls_loss, reg_loss, centerness_loss = self._compute_detection_loss(
            cls_outputs, reg_outputs, centerness_outputs, target_boxes, target_labels
        )

        # Compute mask loss (using ground truth boxes)
        mask_loss = self._compute_mask_loss(
            fpn_features,
            target_boxes,
            target_boxes,
            target_labels,
            target_masks,
        )

        # Total loss
        total_loss = (
            self.cls_loss_weight * cls_loss
            + self.reg_loss_weight * reg_loss
            + self.centerness_loss_weight * centerness_loss
            + self.mask_loss_weight * mask_loss
        )

        # Log losses
        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/cls_loss", cls_loss)
        self.log("train/reg_loss", reg_loss)
        self.log("train/centerness_loss", centerness_loss)
        self.log("train/mask_loss", mask_loss)

        # Enhanced logging
        if self._logging_config.log_learning_rate:
            opt = self.optimizers()
            if opt is not None and hasattr(opt, "param_groups"):
                lr = opt.param_groups[0]["lr"]
                self.log("train/lr", lr, on_step=True, on_epoch=False)

        return total_loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Validation step.

        Args:
            batch: Dict with 'image', 'boxes', 'labels', 'masks'
            batch_idx: Batch index
        """
        images = batch["image"]

        # Get predictions
        predictions = self.predict(images)

        # Prepare targets for metrics. torchmetrics' MeanAveragePrecision
        # (iou_type="segm") requires boolean masks; the dataset stores masks
        # as float.
        targets = []
        for i in range(len(batch["boxes"])):
            target = {
                "boxes": batch["boxes"][i],
                "labels": batch["labels"][i],
                "masks": batch["masks"][i].bool(),
            }
            targets.append(target)

        # Update metrics
        for name, metric in self.val_metrics.items():
            metric.update(predictions, targets)

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Test step.

        Args:
            batch: Dict with 'image', 'boxes', 'labels', 'masks'
            batch_idx: Batch index
        """
        images = batch["image"]

        # Get predictions
        predictions = self.predict(images)

        # Prepare targets for metrics. torchmetrics' MeanAveragePrecision
        # (iou_type="segm") requires boolean masks; the dataset stores masks
        # as float.
        targets = []
        for i in range(len(batch["boxes"])):
            target = {
                "boxes": batch["boxes"][i],
                "labels": batch["labels"][i],
                "masks": batch["masks"][i].bool(),
            }
            targets.append(target)

        # Update metrics
        for name, metric in self.test_metrics.items():
            metric.update(predictions, targets)

    def on_validation_epoch_end(self) -> None:
        """Log metrics at end of validation epoch."""
        is_sanity = getattr(self.trainer, "sanity_checking", False)
        prefix = "sanity_val" if is_sanity else "val"

        for name, metric in self.val_metrics.items():
            try:
                value = metric.compute()
                if isinstance(value, dict):
                    for k, v in value.items():
                        self.log(f"{prefix}/{name}_{k}", v, prog_bar=(k == "map"))
                else:
                    self.log(f"{prefix}/{name}", value, prog_bar=True)
            except Exception:
                pass
            metric.reset()

    def on_test_epoch_end(self) -> None:
        """Log metrics at end of test epoch."""
        for name, metric in self.test_metrics.items():
            try:
                value = metric.compute()
                if isinstance(value, dict):
                    for k, v in value.items():
                        self.log(f"test/{name}_{k}", v)
                else:
                    self.log(f"test/{name}", value)
            except Exception:
                pass
            metric.reset()

    def predict(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Predict instance segmentation for input images.

        Args:
            images: Input images [B, 3, H, W]

        Returns:
            List of dicts with 'boxes', 'labels', 'scores', 'masks' for each image
        """
        with _temporary_eval_mode(self), torch.inference_mode():
            features = self.backbone(images)
            fpn_features = self.fpn(features)
            cls_outputs, reg_outputs, centerness_outputs = self.detection_head(
                fpn_features
            )
            img_h, img_w = images.shape[-2:]

            detections = decode_fcos_detections(
                cls_outputs,
                reg_outputs,
                centerness_outputs,
                self.strides,
                self.num_classes,
                (img_h, img_w),
                self.score_thresh,
                self.nms_thresh,
                self.max_detections_per_image,
            )

            # Predict masks for detected boxes via ROI Align on FPN level 0 (P3),
            # matching the stride/resolution convention used in _compute_mask_loss.
            feature_map = fpn_features[0]
            predictions = []

            for i, det in enumerate(detections):
                boxes = det["boxes"]
                labels = det["labels"]
                scores = det["scores"]

                if len(boxes) == 0:
                    predictions.append(
                        {
                            "boxes": boxes,
                            "labels": labels,
                            "scores": scores,
                            "masks": torch.empty(
                                (0, img_h, img_w),
                                dtype=torch.bool,
                                device=images.device,
                            ),
                        }
                    )
                    continue

                batch_indices = torch.full(
                    (len(boxes), 1), i, dtype=boxes.dtype, device=images.device
                )
                rois = torch.cat([batch_indices, boxes], dim=1)  # [N, 5]

                roi_features = ops.roi_align(
                    feature_map,
                    rois,
                    output_size=(self.roi_pool_size, self.roi_pool_size),
                    spatial_scale=1.0 / 8,  # Assuming P3 has stride 8
                    aligned=True,
                )

                mask_logits = self.mask_head(
                    roi_features
                )  # [N, num_classes, mask_size, mask_size]
                indices = torch.arange(len(labels), device=images.device)
                mask_logits = mask_logits[indices, labels]  # [N, mask_size, mask_size]

                masks = F.interpolate(
                    mask_logits.sigmoid().unsqueeze(1),
                    size=(img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                masks = masks > self.mask_threshold

                predictions.append(
                    {
                        "boxes": boxes,
                        "labels": labels,
                        "scores": scores,
                        "masks": masks,
                    }
                )

            return predictions

    def predict_step(self, batch: Any, batch_idx: int) -> list[dict[str, torch.Tensor]]:
        """Prediction step for Trainer.predict().

        Args:
            batch: Input batch
            batch_idx: Batch index

        Returns:
            List of predictions
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

        Exports the detection head only (mask head excluded). Outputs are flattened
        into named tensors (cls_l0..cls_l4, reg_l0..reg_l4, ctr_l0..ctr_l4).

        Args:
            save_path: Path to save the ONNX model. If None, uses a temp file.
            example_input: Example input tensor. If None, uses default shape (1, 3, 224, 224).
            opset_version: ONNX opset version. Default is 17.
            dynamic_axes: Dynamic axes specification. If None, batch dimension is dynamic.
            **kwargs: Additional arguments passed to export_to_onnx.

        Returns:
            Path to the saved ONNX model.

        Example:
            >>> model = InstanceSegmentor(backbone="resnet50", num_classes=80)
            >>> path = model.to_onnx("instance_seg.onnx")
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
        """Configure optimizer and learning rate scheduler."""
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self._create_optimizer(params)

        if self._scheduler is None or self._scheduler == "none":
            return {"optimizer": optimizer}

        scheduler_config = self._create_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def _create_optimizer(self, params) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        opt_kwargs = {"lr": self._lr, "weight_decay": self._weight_decay}
        opt_kwargs.update(self._optimizer_kwargs)

        if isinstance(self._optimizer, dict):
            opt_class_path = self._optimizer["class"]
            opt_params = self._optimizer.get("params", {})
            opt_kwargs.update(opt_params)
            optimizer_cls = self._import_class(opt_class_path)
            return optimizer_cls(params, **opt_kwargs)

        optimizer_name = self._optimizer.lower()
        torch_optimizers = {
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }

        if optimizer_name in torch_optimizers:
            return torch_optimizers[optimizer_name](params, **opt_kwargs)

        raise ValueError(f"Unknown optimizer: {self._optimizer}")

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> dict:
        """Create scheduler config from optimizer."""
        sched_kwargs = self._scheduler_kwargs.copy()
        interval = sched_kwargs.pop("interval", "step")
        frequency = sched_kwargs.pop("frequency", 1)

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

        scheduler_name = self._scheduler.lower()

        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_kwargs.pop(
                    "T_max", self.trainer.estimated_stepping_batches
                ),
                **sched_kwargs,
            )
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_kwargs.pop("step_size", 30),
                **sched_kwargs,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self._scheduler}")

        return {"scheduler": scheduler, "interval": interval, "frequency": frequency}

    def _import_class(self, class_path: str):
        """Import a class from a fully qualified path."""
        import importlib

        if "." not in class_path:
            raise ValueError(f"Class path must be fully qualified, got: {class_path}")

        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
