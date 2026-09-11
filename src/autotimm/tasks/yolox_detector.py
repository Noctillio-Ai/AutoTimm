"""YOLOX Detector - Complete YOLOX model with official architecture.

Combines CSPDarknet backbone + YOLOXPAFPN neck + YOLOX decoupled head.
"""

from __future__ import annotations

import datetime as _dt
import getpass
from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn
from torchvision import ops

from autotimm.core.metrics import LoggingConfig, MetricConfig, MetricManager
from autotimm.core.utils import seed_everything
from autotimm.data.transform_config import TransformConfig
from autotimm.heads import YOLOXHead
from autotimm.losses import FocalLoss, GIoULoss
from autotimm.models.csp_darknet import build_csp_darknet
from autotimm.models.yolox_pafpn import build_yolox_pafpn
from autotimm.tasks.preprocessing_mixin import PreprocessingMixin


class YOLOXDetector(PreprocessingMixin, pl.LightningModule):
    """Official YOLOX object detector with CSPDarknet backbone.

    Complete YOLOX architecture following the official implementation:
    CSPDarknet → YOLOXPAFPN → YOLOXHead → NMS

    Supports all YOLOX variants:
    - YOLOX-Nano: 0.91M params, 416x416 input
    - YOLOX-Tiny: 5.06M params, 416x416 input
    - YOLOX-s: 9.0M params, 640x640 input
    - YOLOX-m: 25.3M params, 640x640 input
    - YOLOX-l: 54.2M params, 640x640 input
    - YOLOX-x: 99.1M params, 640x640 input

    Parameters:
        model_name: YOLOX variant ("yolox-nano", "yolox-tiny", "yolox-s",
            "yolox-m", "yolox-l", "yolox-x")
        num_classes: Number of object classes (excluding background)
        metrics: MetricManager or list of MetricConfig (optional)
        logging_config: Optional LoggingConfig for enhanced logging
        transform_config: Optional TransformConfig for preprocessing
        lr: Learning rate
        weight_decay: Weight decay
        optimizer: Optimizer name or dict
        optimizer_kwargs: Additional optimizer kwargs
        scheduler: Scheduler name or dict
        scheduler_kwargs: Additional scheduler kwargs
        focal_alpha: Alpha for focal loss
        focal_gamma: Gamma for focal loss
        cls_loss_weight: Classification loss weight
        reg_loss_weight: Regression loss weight (typically 5.0 for YOLOX)
        score_thresh: Score threshold for inference
        nms_thresh: NMS IoU threshold
        max_detections_per_image: Max detections per image
        compile_model: If ``True`` (default), apply ``torch.compile()`` to the backbone, neck, and head
            for faster inference and training. Requires PyTorch 2.0+.
        compile_kwargs: Optional dict of kwargs to pass to ``torch.compile()``.
            Common options: ``mode`` (``"default"``, ``"reduce-overhead"``, ``"max-autotune"``),
            ``fullgraph`` (``True``/``False``), ``dynamic`` (``True``/``False``).
        seed: Random seed for reproducibility. If ``None``, no seeding is performed.
            Default is ``42`` for reproducible results.
        deterministic: If ``True`` (default), enables deterministic algorithms in PyTorch for full
            reproducibility (may impact performance). Set to ``False`` for faster training.

    Example:
        >>> from autotimm import YOLOXDetector, DetectionDataModule, AutoTrainer
        >>>
        >>> # Create YOLOX-s model
        >>> model = YOLOXDetector(
        ...     model_name="yolox-s",
        ...     num_classes=80,
        ...     lr=1e-3,
        ...     reg_loss_weight=5.0,
        ... )
        >>>
        >>> # Data and training
        >>> data = DetectionDataModule(data_dir="./coco", image_size=640, batch_size=16)
        >>> trainer = AutoTrainer(max_epochs=300, precision="16-mixed")
        >>> trainer.fit(model, datamodule=data)
    """

    def __init__(
        self,
        model_name: str = "yolox-s",
        num_classes: int = 80,
        metrics: MetricManager | list[MetricConfig] | None = None,
        logging_config: LoggingConfig | None = None,
        transform_config: TransformConfig | None = None,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        optimizer: str | dict[str, Any] = "sgd",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: str | dict[str, Any] | None = "yolox",
        scheduler_kwargs: dict[str, Any] | None = None,
        total_epochs: int = 300,
        warmup_epochs: int = 5,
        no_aug_epochs: int = 15,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cls_loss_weight: float = 1.0,
        reg_loss_weight: float = 5.0,
        score_thresh: float = 0.01,
        nms_thresh: float = 0.65,
        max_detections_per_image: int = 100,
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

        super().__init__()
        self.save_hyperparameters(
            ignore=["metrics", "logging_config", "transform_config"]
        )
        self.hparams.update(
            {
                "username": getpass.getuser(),
                "timestamp": _dt.datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
            }
        )

        self.model_name = model_name
        self.num_classes = num_classes
        self._lr = lr
        self._weight_decay = weight_decay
        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._scheduler = scheduler
        self._scheduler_kwargs = scheduler_kwargs or {}
        self._total_epochs = total_epochs
        self._warmup_epochs = warmup_epochs
        self._no_aug_epochs = no_aug_epochs

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.max_detections_per_image = max_detections_per_image

        # YOLOX uses 3 strides: 8, 16, 32 (P3, P4, P5)
        self.strides = (8, 16, 32)

        # Build YOLOX components
        self.backbone = build_csp_darknet(model_name)
        self.neck = build_yolox_pafpn(model_name)

        # Get output channels from neck (uniform across all levels)
        in_channels = self.neck.out_channels

        self.head = YOLOXHead(
            in_channels=in_channels,
            num_classes=num_classes,
            num_convs=2,
            prior_prob=0.01,
            activation="silu",
        )

        # Losses
        self.focal_loss = FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma, reduction="sum"
        )
        self.giou_loss = GIoULoss(reduction="sum")
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss_weight = reg_loss_weight

        # Metrics
        if metrics is not None:
            if isinstance(metrics, list):
                self._metric_manager = MetricManager(
                    configs=metrics, num_classes=num_classes
                )
            else:
                self._metric_manager = metrics

            self.val_metrics = self._metric_manager.get_val_metrics()
            self.test_metrics = self._metric_manager.get_test_metrics()
        else:
            self._metric_manager = None
            self.val_metrics = nn.ModuleDict()
            self.test_metrics = nn.ModuleDict()

        # Logging config
        self._logging_config = logging_config or LoggingConfig(
            log_learning_rate=False,
            log_gradient_norm=False,
        )

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
                self.neck = torch.compile(self.neck, **compile_opts)
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

        # Setup transforms
        self._setup_transforms(transform_config, task="detection")

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
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass through YOLOX.

        Args:
            images: Input images [B, 3, H, W]

        Returns:
            Tuple of (cls_outputs, reg_outputs) - each a list of 3 tensors
        """
        # Backbone: extract multi-scale features
        features = self.backbone(images)

        # Neck: PAFPN fusion
        pafpn_outputs = self.neck(features)

        # Head: classification and regression predictions
        cls_outputs, reg_outputs = self.head(pafpn_outputs)

        return cls_outputs, reg_outputs

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step - compute loss."""
        images = batch["image"]
        target_boxes = batch["boxes"]
        target_labels = batch["labels"]

        # Forward
        cls_outputs, reg_outputs = self(images)

        # Compute targets and loss
        device = images.device
        batch_size = images.shape[0]
        img_h, img_w = images.shape[-2:]

        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        num_pos = 0

        for level_idx, (cls_out, reg_out) in enumerate(zip(cls_outputs, reg_outputs)):
            stride = self.strides[level_idx]
            feat_h, feat_w = cls_out.shape[-2:]

            # Generate grid points
            grid_y, grid_x = torch.meshgrid(
                torch.arange(feat_h, device=device, dtype=torch.float32),
                torch.arange(feat_w, device=device, dtype=torch.float32),
                indexing="ij",
            )
            points_x = (grid_x + 0.5) * stride
            points_y = (grid_y + 0.5) * stride
            points = torch.stack([points_x, points_y], dim=-1)

            # Compute targets per image
            level_cls_targets = []
            level_reg_targets = []

            for b in range(batch_size):
                boxes = target_boxes[b]
                labels = target_labels[b]

                if len(boxes) == 0:
                    cls_target = torch.full(
                        (feat_h, feat_w), -1, dtype=torch.long, device=device
                    )
                    reg_target = torch.zeros(feat_h, feat_w, 4, device=device)
                else:
                    cls_target, reg_target, _ = self._compute_targets_per_level(
                        points, boxes, labels, stride, level_idx, (img_h, img_w)
                    )

                level_cls_targets.append(cls_target)
                level_reg_targets.append(reg_target)

            cls_targets = torch.stack(level_cls_targets)
            reg_targets = torch.stack(level_reg_targets)

            # Classification loss. YOLOX uses independent sigmoid classifiers,
            # so unassigned points are background negatives (all-zero targets),
            # not ignored samples — index targets with -1 would drop every
            # background location and the detector would never learn to
            # suppress false positives.
            cls_out_flat = cls_out.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            cls_targets_flat = cls_targets.reshape(-1)
            pos_cls_mask = cls_targets_flat >= 0

            dense_cls_targets = torch.zeros_like(cls_out_flat)
            if pos_cls_mask.any():
                dense_cls_targets[pos_cls_mask, cls_targets_flat[pos_cls_mask]] = 1.0
            total_cls_loss = total_cls_loss + self.focal_loss(
                cls_out_flat, dense_cls_targets
            )

            # Regression loss (positive samples only)
            pos_mask = cls_targets >= 0
            if pos_mask.any():
                pos_reg_pred = reg_out.permute(0, 2, 3, 1)[pos_mask]
                pos_reg_target = reg_targets[pos_mask]
                reg_loss = self._compute_iou_loss(pos_reg_pred, pos_reg_target)
                total_reg_loss = total_reg_loss + reg_loss
                num_pos += pos_mask.sum().item()

        # Normalize losses
        num_pos = max(num_pos, 1)
        cls_loss = self.cls_loss_weight * total_cls_loss / num_pos
        reg_loss = self.reg_loss_weight * total_reg_loss / num_pos
        total_loss = cls_loss + reg_loss

        # Log losses
        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/cls_loss", cls_loss)
        self.log("train/reg_loss", reg_loss)
        self.log("train/num_pos", float(num_pos))

        # Enhanced logging
        if self._logging_config.log_learning_rate:
            opt = self.optimizers()
            if opt is not None and hasattr(opt, "param_groups"):
                lr = opt.param_groups[0]["lr"]
                self.log("train/lr", lr, on_step=True, on_epoch=False)

        return total_loss

    def _compute_targets_per_level(
        self,
        points: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        stride: int,
        level_idx: int,
        img_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute YOLOX targets for a single image at one level."""
        feat_h, feat_w = points.shape[:2]
        device = points.device

        cls_target = torch.full((feat_h, feat_w), -1, dtype=torch.long, device=device)
        reg_target = torch.zeros(feat_h, feat_w, 4, device=device)
        cent_target = torch.zeros(feat_h, feat_w, device=device)

        if len(boxes) == 0:
            return cls_target, reg_target, cent_target

        # Reshape points for computation
        points_flat = points.reshape(-1, 2)

        # Compute distances from points to box edges (LTRB format)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        left = points_flat[:, 0].unsqueeze(1) - x1.unsqueeze(0)
        top = points_flat[:, 1].unsqueeze(1) - y1.unsqueeze(0)
        right = x2.unsqueeze(0) - points_flat[:, 0].unsqueeze(1)
        bottom = y2.unsqueeze(0) - points_flat[:, 1].unsqueeze(1)

        ltrb = torch.stack([left, top, right, bottom], dim=-1)
        is_in_boxes = ltrb.min(dim=-1).values > 0

        # For YOLOX, we use all points inside boxes (no per-level regression ranges)
        is_valid = is_in_boxes

        # For each point, assign to the box with minimum area
        valid_points = is_valid.any(dim=1)
        if valid_points.any():
            areas = (x2 - x1) * (y2 - y1)
            areas_expanded = areas.unsqueeze(0).expand(len(points_flat), -1)
            areas_expanded = areas_expanded.masked_fill(~is_valid, float("inf"))

            min_area_inds = areas_expanded.argmin(dim=1)
            assigned_boxes = ltrb[torch.arange(len(points_flat)), min_area_inds]
            assigned_labels = labels[min_area_inds]

            # Reshape back to spatial dimensions
            assigned_boxes_spatial = assigned_boxes.reshape(feat_h, feat_w, 4)
            assigned_labels_spatial = assigned_labels.reshape(feat_h, feat_w)
            valid_mask_spatial = valid_points.reshape(feat_h, feat_w)

            cls_target[valid_mask_spatial] = assigned_labels_spatial[valid_mask_spatial]
            reg_target[valid_mask_spatial] = assigned_boxes_spatial[valid_mask_spatial]

        return cls_target, reg_target, cent_target

    def _compute_iou_loss(
        self, pred_ltrb: torch.Tensor, target_ltrb: torch.Tensor
    ) -> torch.Tensor:
        """Compute GIoU loss from LTRB predictions.

        GIoULoss expects xyxy boxes; a point with distances (l, t, r, b)
        corresponds to the box (-l, -t, r, b) relative to the point, and
        GIoU is translation-invariant, so relative boxes give the correct
        loss. (Passing raw LTRB tuples as if they were boxes produced a
        nonzero loss even for identical predictions/targets.)
        """
        from autotimm.tasks._fcos_targets import _ltrb_to_relative_boxes

        return self.giou_loss(
            _ltrb_to_relative_boxes(pred_ltrb),
            _ltrb_to_relative_boxes(target_ltrb),
        )

    def predict(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Run inference and return detections with NMS."""
        cls_outputs, reg_outputs = self(images)

        batch_size = images.shape[0]
        img_h, img_w = images.shape[-2:]
        device = images.device

        all_detections = []

        for b in range(batch_size):
            all_boxes = []
            all_scores = []
            all_labels = []

            for level_idx, (cls_out, reg_out) in enumerate(
                zip(cls_outputs, reg_outputs)
            ):
                stride = self.strides[level_idx]
                feat_h, feat_w = cls_out.shape[-2:]

                cls_logits = cls_out[b]
                reg_pred = reg_out[b]

                # Generate grid
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(feat_h, device=device, dtype=torch.float32),
                    torch.arange(feat_w, device=device, dtype=torch.float32),
                    indexing="ij",
                )
                points_x = (grid_x + 0.5) * stride
                points_y = (grid_y + 0.5) * stride

                # Flatten
                cls_logits = cls_logits.permute(1, 2, 0).reshape(-1, self.num_classes)
                reg_pred = reg_pred.permute(1, 2, 0).reshape(-1, 4)
                points_x = points_x.reshape(-1)
                points_y = points_y.reshape(-1)

                # YOLOX uses classification scores only (no centerness)
                scores = cls_logits.sigmoid()
                max_scores, class_ids = scores.max(dim=-1)

                # Filter by threshold
                keep = max_scores > self.score_thresh
                if not keep.any():
                    continue

                max_scores = max_scores[keep]
                class_ids = class_ids[keep]
                reg_pred = reg_pred[keep]
                points_x = points_x[keep]
                points_y = points_y[keep]

                # Convert LTRB to xyxy
                left, top, right, bottom = reg_pred.unbind(dim=-1)
                x1 = (points_x - left).clamp(min=0, max=img_w)
                y1 = (points_y - top).clamp(min=0, max=img_h)
                x2 = (points_x + right).clamp(min=0, max=img_w)
                y2 = (points_y + bottom).clamp(min=0, max=img_h)

                boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                all_boxes.append(boxes)
                all_scores.append(max_scores)
                all_labels.append(class_ids)

            # Concatenate all levels
            if len(all_boxes) > 0:
                boxes = torch.cat(all_boxes, dim=0)
                scores = torch.cat(all_scores, dim=0)
                labels = torch.cat(all_labels, dim=0)

                # NMS per class
                keep_indices = []
                for c in labels.unique():
                    class_mask = labels == c
                    class_boxes = boxes[class_mask]
                    class_scores = scores[class_mask]
                    class_keep = ops.nms(class_boxes, class_scores, self.nms_thresh)
                    keep_indices.append(torch.where(class_mask)[0][class_keep])

                if keep_indices:
                    keep_indices = torch.cat(keep_indices)
                    boxes = boxes[keep_indices]
                    scores = scores[keep_indices]
                    labels = labels[keep_indices]

                    # Limit detections
                    if len(scores) > self.max_detections_per_image:
                        top_k = scores.topk(self.max_detections_per_image)[1]
                        boxes = boxes[top_k]
                        scores = scores[top_k]
                        labels = labels[top_k]

                    all_detections.append(
                        {"boxes": boxes, "scores": scores, "labels": labels}
                    )
                else:
                    all_detections.append(
                        {
                            "boxes": torch.empty(0, 4, device=device),
                            "scores": torch.empty(0, device=device),
                            "labels": torch.empty(0, dtype=torch.long, device=device),
                        }
                    )
            else:
                all_detections.append(
                    {
                        "boxes": torch.empty(0, 4, device=device),
                        "scores": torch.empty(0, device=device),
                        "labels": torch.empty(0, dtype=torch.long, device=device),
                    }
                )

        return all_detections

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Validation step."""
        images = batch["image"]
        predictions = self.predict(images)
        targets = [
            {"boxes": boxes, "labels": labels}
            for boxes, labels in zip(batch["boxes"], batch["labels"])
        ]

        # Update metrics
        for metric in self.val_metrics.values():
            metric.update(predictions, targets)

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics."""
        is_sanity = getattr(self.trainer, "sanity_checking", False)
        prefix = "sanity_val" if is_sanity else "val"

        for name, metric in self.val_metrics.items():
            try:
                value = metric.compute()
                if isinstance(value, dict):
                    for k, v in value.items():
                        self.log(f"{prefix}/{k}", v)
                else:
                    self.log(f"{prefix}/{name}", value)
            except Exception:
                pass
            metric.reset()

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Test step."""
        images = batch["image"]
        predictions = self.predict(images)
        targets = [
            {"boxes": boxes, "labels": labels}
            for boxes, labels in zip(batch["boxes"], batch["labels"])
        ]

        for metric in self.test_metrics.values():
            metric.update(predictions, targets)

    def on_test_epoch_end(self) -> None:
        """Log test metrics."""
        for name, metric in self.test_metrics.items():
            try:
                value = metric.compute()
                if isinstance(value, dict):
                    for k, v in value.items():
                        self.log(f"test/{k}", v)
                else:
                    self.log(f"test/{name}", value)
            except Exception:
                pass
            metric.reset()

    def predict_step(self, batch: Any, batch_idx: int) -> list[dict[str, torch.Tensor]]:
        """Prediction step."""
        images = batch["image"] if isinstance(batch, dict) else batch
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

        YOLOX outputs are flattened into named tensors (cls_l0..cls_l2, reg_l0..reg_l2).

        Args:
            save_path: Path to save the ONNX model. If None, uses a temp file.
            example_input: Example input tensor. If None, uses default shape (1, 3, 640, 640).
            opset_version: ONNX opset version. Default is 17.
            dynamic_axes: Dynamic axes specification. If None, batch dimension is dynamic.
            **kwargs: Additional arguments passed to export_to_onnx.

        Returns:
            Path to the saved ONNX model.

        Example:
            >>> model = YOLOXDetector(num_classes=80)
            >>> path = model.to_onnx("yolox.onnx")
        """
        from autotimm.export import export_to_onnx

        if example_input is None:
            example_input = torch.randn(1, 3, 640, 640)

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
        """Configure optimizer and scheduler.

        Uses official YOLOX optimizer and scheduler settings:
        - Optimizer: SGD with momentum=0.9, nesterov=True
        - Scheduler: Warmup (5 epochs) + Cosine/Linear decay
        """
        from autotimm.models.yolox_scheduler import YOLOXLRScheduler

        params = list(self.parameters())

        # Dict config: {"class": "path.to.Optimizer", "params": {...}}
        if isinstance(self._optimizer, dict):
            opt_kwargs = {"lr": self._lr, "weight_decay": self._weight_decay}
            opt_kwargs.update(self._optimizer_kwargs)
            opt_kwargs.update(self._optimizer.get("params", {}))
            optimizer_cls = self._import_class(self._optimizer["class"])
            optimizer = optimizer_cls(params, **opt_kwargs)
        elif self._optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=self._lr,
                momentum=self._optimizer_kwargs.get("momentum", 0.9),
                weight_decay=self._weight_decay,
                nesterov=self._optimizer_kwargs.get("nesterov", True),
            )
        elif self._optimizer.lower() in ["adam", "adamw"]:
            optimizer = torch.optim.AdamW(
                params,
                lr=self._lr,
                weight_decay=self._weight_decay,
                betas=self._optimizer_kwargs.get("betas", (0.9, 0.999)),
            )
        else:
            # Default to SGD (YOLOX official)
            optimizer = torch.optim.SGD(
                params,
                lr=self._lr,
                momentum=0.9,
                weight_decay=self._weight_decay,
                nesterov=True,
            )

        if self._scheduler is None:
            return {"optimizer": optimizer}

        # Create scheduler
        # Dict config: {"class": "path.to.Scheduler", "params": {...}}
        if isinstance(self._scheduler, dict):
            sched_kwargs = dict(self._scheduler.get("params", {}))
            scheduler_cls = self._import_class(self._scheduler["class"])
            scheduler = scheduler_cls(optimizer, **sched_kwargs)
        elif self._scheduler.lower() == "yolox":
            # Official YOLOX scheduler with warmup
            scheduler = YOLOXLRScheduler(
                optimizer,
                total_epochs=self._scheduler_kwargs.get(
                    "total_epochs", self._total_epochs
                ),
                warmup_epochs=self._scheduler_kwargs.get(
                    "warmup_epochs", self._warmup_epochs
                ),
                no_aug_epochs=self._scheduler_kwargs.get(
                    "no_aug_epochs", self._no_aug_epochs
                ),
                min_lr_ratio=self._scheduler_kwargs.get("min_lr_ratio", 0.05),
                scheduler_type=self._scheduler_kwargs.get("scheduler_type", "cosine"),
            )
        elif self._scheduler.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self._scheduler_kwargs.get("T_max", self._total_epochs),
                eta_min=self._scheduler_kwargs.get("eta_min", 0),
            )
        elif self._scheduler.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self._scheduler_kwargs.get("step_size", 30),
                gamma=self._scheduler_kwargs.get("gamma", 0.1),
            )
        elif self._scheduler.lower() == "multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self._scheduler_kwargs.get("milestones", [200, 250]),
                gamma=self._scheduler_kwargs.get("gamma", 0.1),
            )
        else:
            return {"optimizer": optimizer}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _import_class(self, class_path: str):
        """Import a class from a fully qualified path."""
        import importlib

        if "." not in class_path:
            raise ValueError(f"Class path must be fully qualified, got: {class_path}")

        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
