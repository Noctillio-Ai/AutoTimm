"""JSON progress callback for structured training events on stdout."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch


def _emit(
    event: dict[str, Any],
    log_file: Path | None = None,
) -> None:
    """Write a JSON event line to stdout (and optionally a log file) and flush.

    Stdout writes are wrapped in a try/except so a broken pipe (e.g. Tauri
    crash) does not kill the training process.
    """
    event["timestamp"] = time.time()
    line = json.dumps(event, default=str) + "\n"

    # Always try stdout — swallow BrokenPipeError so training survives
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except BrokenPipeError:
        pass

    # Durable file log — Tauri can tail / replay this on reconnect
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(line)


def _sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert tensor values to Python scalars."""
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.item() if v.numel() == 1 else v.tolist()
        else:
            out[k] = v
    return out


def _build_confusion_data(
    preds: list[torch.Tensor],
    targets: list[torch.Tensor],
    trainer: pl.Trainer,
) -> dict[str, Any] | None:
    """Build confusion matrix + per-class metrics from collected predictions.

    Returns a dict with ``confusion_matrix`` and ``per_class_metrics`` keys,
    or ``None`` if the data is unsuitable (empty, multi-label, etc.).
    """
    if not preds or not targets:
        return None

    all_preds = torch.cat(preds)
    all_targets = torch.cat(targets)

    # Only for single-label classification (1-D integer tensors)
    if all_preds.ndim != 1 or all_targets.ndim != 1:
        return None

    num_classes = int(max(all_preds.max(), all_targets.max()) + 1)

    # Try to get class names from the datamodule
    class_names: list[str] | None = None
    dm = getattr(trainer, "datamodule", None)
    if dm is not None:
        cn = getattr(dm, "class_names", None)
        if cn and len(cn) >= num_classes:
            class_names = list(cn[:num_classes])
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Confusion matrix
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for p, t in zip(all_preds, all_targets):
        cm[int(t), int(p)] += 1

    # Per-class precision / recall / f1
    per_class = []
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class.append(
            {
                "class_index": c,
                "label": class_names[c],
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )

    return {
        "confusion_matrix": {"matrix": cm.tolist(), "labels": class_names},
        "per_class_metrics": per_class,
    }


def _evaluate_on_loader(
    pl_module: pl.LightningModule,
    loader: Any,
    multi_label: bool = False,
    threshold: float = 0.5,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run the model on a dataloader and collect preds + targets."""
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    device = pl_module.device
    was_training = pl_module.training
    pl_module.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            logits = pl_module(x)
            if multi_label:
                pred = (logits.sigmoid() > threshold).int().cpu()
            else:
                pred = logits.argmax(dim=-1).cpu()
            preds.append(pred)
            targets.append(y.cpu() if isinstance(y, torch.Tensor) else torch.tensor(y))
    if was_training:
        pl_module.train()
    return preds, targets


class JsonProgressCallback(pl.Callback):
    """Lightning Callback that emits NDJSON progress events to stdout.

    Confusion matrices are built once at the end of testing (on the best
    model) for all three splits — train, val, and test — rather than
    per-epoch during training.

    Parameters:
        emit_every_n_steps: How often to emit ``batch_end`` events.
            Set to 1 to emit on every batch.  Default ``10``.
        log_file: Optional path to an NDJSON file where every event is
            also appended.  Tauri can read this file on reconnect to
            replay missed events.  If ``None`` (default), events are
            only written to stdout.
    """

    def __init__(
        self,
        emit_every_n_steps: int = 10,
        log_file: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.emit_every_n_steps = emit_every_n_steps
        self._log_file: Path | None = Path(log_file) if log_file is not None else None
        # Accumulate test predictions for confusion matrix
        self._test_preds: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []

    def _emit(self, event: dict[str, Any]) -> None:
        """Emit an event via the module-level helper with our log_file."""
        _emit(event, log_file=self._log_file)

    # ------------------------------------------------------------------
    # Fit lifecycle
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._emit(
            {
                "event": "training_started",
                "max_epochs": trainer.max_epochs,
                "total_steps": trainer.estimated_stepping_batches,
                "fast_dev_run": bool(trainer.fast_dev_run),
            }
        )

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = _sanitize_metrics(trainer.callback_metrics)
        self._emit({"event": "training_complete", "final_metrics": metrics})

    # ------------------------------------------------------------------
    # Training epoch / batch
    # ------------------------------------------------------------------

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._emit(
            {
                "event": "epoch_started",
                "epoch": trainer.current_epoch,
                "max_epochs": trainer.max_epochs,
            }
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if (batch_idx + 1) % self.emit_every_n_steps != 0:
            return

        loss = None
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        elif isinstance(outputs, torch.Tensor):
            loss = outputs

        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        self._emit(
            {
                "event": "batch_end",
                "epoch": trainer.current_epoch,
                "step": trainer.global_step,
                "total_steps": trainer.estimated_stepping_batches,
                "loss": loss,
            }
        )

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        metrics = _sanitize_metrics(trainer.callback_metrics)
        self._emit(
            {
                "event": "epoch_end",
                "epoch": trainer.current_epoch,
                "metrics": metrics,
            }
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        metrics = _sanitize_metrics(trainer.callback_metrics)
        self._emit(
            {
                "event": "validation_end",
                "epoch": trainer.current_epoch,
                "metrics": metrics,
            }
        )

    # ------------------------------------------------------------------
    # Test lifecycle
    # ------------------------------------------------------------------

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._test_preds = []
        self._test_targets = []
        total_batches = trainer.num_test_batches[0] if trainer.num_test_batches else 0
        self._emit({"event": "testing_started", "total_batches": total_batches})

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Collect predictions/targets for confusion matrix
        if isinstance(outputs, dict) and "preds" in outputs and "targets" in outputs:
            self._test_preds.append(outputs["preds"].cpu())
            self._test_targets.append(outputs["targets"].cpu())

        if (batch_idx + 1) % self.emit_every_n_steps != 0:
            return

        total_batches = trainer.num_test_batches[0] if trainer.num_test_batches else 0
        self._emit(
            {
                "event": "test_batch_end",
                "batch": batch_idx + 1,
                "total_batches": total_batches,
            }
        )

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        metrics = _sanitize_metrics(trainer.callback_metrics)

        extra: dict[str, Any] = {}

        # Build test confusion matrix from collected predictions
        cm_data = _build_confusion_data(self._test_preds, self._test_targets, trainer)
        if cm_data:
            extra["confusion_matrix"] = cm_data["confusion_matrix"]
            extra["per_class_metrics"] = cm_data["per_class_metrics"]
        self._test_preds = []
        self._test_targets = []

        # Also evaluate on train and val splits using the current (best) model.
        # At test time the best checkpoint is loaded, so these CMs reflect
        # the best model's performance on all splits.
        dm = getattr(trainer, "datamodule", None)
        if dm is not None:
            multi_label = getattr(pl_module, "_multi_label", False)
            threshold = getattr(pl_module, "_threshold", 0.5)
            for stage, loader_attr in [
                ("train", "train_dataloader"),
                ("val", "val_dataloader"),
            ]:
                try:
                    loader_fn = getattr(dm, loader_attr, None)
                    if loader_fn is None:
                        continue
                    loader = loader_fn()
                    if loader is None:
                        continue
                    preds, targets = _evaluate_on_loader(
                        pl_module,
                        loader,
                        multi_label=multi_label,
                        threshold=threshold,
                    )
                    stage_cm = _build_confusion_data(preds, targets, trainer)
                    if stage_cm:
                        extra[f"{stage}_confusion_matrix"] = stage_cm[
                            "confusion_matrix"
                        ]
                        extra[f"{stage}_per_class_metrics"] = stage_cm[
                            "per_class_metrics"
                        ]
                except Exception:
                    pass  # Non-critical — don't break test reporting

        self._emit({"event": "testing_complete", "metrics": metrics, **extra})

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        self._emit(
            {
                "event": "error",
                "message": str(exception),
                "type": type(exception).__name__,
            }
        )
