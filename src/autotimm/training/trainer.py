"""Convenience wrapper around pl.Trainer for autotimm tasks."""

from __future__ import annotations

import multiprocessing
import platform
import sys
from dataclasses import dataclass
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner

from autotimm.callbacks.json_progress import JsonProgressCallback, _emit
from autotimm.core.loggers import LoggerConfig, LoggerManager
from autotimm.core.logging import logger
from autotimm.core.utils import seed_everything


def _ensure_safe_multiprocessing() -> None:
    """Set multiprocessing start method to 'fork' on macOS if not already set.

    macOS defaults to 'spawn' which requires ``if __name__ == '__main__'``
    guards. Using 'fork' avoids this issue for dataloader workers.

    On Windows, 'spawn' is the only supported start method and is already the
    default, so no action is needed.  On Linux, 'fork' is already the default.
    """
    if platform.system() == "Darwin" and sys.version_info >= (3, 8):
        try:
            multiprocessing.set_start_method("fork", force=False)
        except RuntimeError:
            # Already set — nothing to do.
            pass


# Module-level flag to ensure watermark is printed only once per session
_WATERMARK_PRINTED = False


def _print_watermark() -> None:
    """Print environment watermark with package versions."""
    global _WATERMARK_PRINTED
    if _WATERMARK_PRINTED:
        return
    _WATERMARK_PRINTED = True

    try:
        from watermark import watermark

        logger.info(
            watermark(packages="torch,pytorch_lightning,timm,transformers", python=True)
        )
    except ImportError:
        # Fallback if watermark is not available
        import sys

        import torch

        logger.info(f"Python: {sys.version}")
        logger.info(f"PyTorch: {torch.__version__}")
        try:
            import pytorch_lightning

            logger.info(f"Lightning: {pytorch_lightning.__version__}")
        except Exception:
            pass
        try:
            import timm

            logger.info(f"timm: {timm.__version__}")
        except ImportError:
            pass
        try:
            import transformers

            logger.info(f"transformers: {transformers.__version__}")
        except ImportError:
            pass
    except Exception:
        # Silently ignore any errors in watermark printing
        pass


@dataclass
class TunerConfig:
    """Configuration for automatic hyperparameter tuning.

    Parameters:
        auto_lr: Whether to use the learning rate finder before training.
            If ``True``, the optimal learning rate will be found and applied.
            If ``False``, the user-specified learning rate is used.
            Default: ``True``.
        auto_batch_size: Whether to use the batch size finder before training.
            If ``True``, the optimal batch size will be found and applied.
            If ``False``, the user-specified batch size is used.
            Default: ``True``.
        lr_find_kwargs: Additional kwargs passed to ``Tuner.lr_find()``.
            Common options: ``min_lr``, ``max_lr``, ``num_training``,
            ``mode`` ("exponential" or "linear"), ``early_stop_threshold``.
            Default values are set if not provided.
        batch_size_kwargs: Additional kwargs passed to ``Tuner.scale_batch_size()``.
            Common options: ``mode`` ("power" or "binsearch"), ``steps_per_trial``,
            ``init_val``, ``max_trials``.
            Default values are set if not provided.

    Example:
        >>> # Use defaults (both auto_lr and auto_batch_size enabled)
        >>> config = TunerConfig()

        >>> # Disable auto-tuning
        >>> config = TunerConfig(auto_lr=False, auto_batch_size=False)

        >>> # Custom configuration
        >>> config = TunerConfig(
        ...     auto_lr=True,
        ...     auto_batch_size=True,
        ...     lr_find_kwargs={"min_lr": 1e-6, "max_lr": 1e-1, "num_training": 100},
        ...     batch_size_kwargs={"mode": "power", "init_val": 16},
        ... )
    """

    auto_lr: bool = True
    auto_batch_size: bool = True
    lr_find_kwargs: dict[str, Any] | None = None
    batch_size_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # Set sensible defaults for LR finder if not provided
        if self.lr_find_kwargs is None:
            self.lr_find_kwargs = {
                "min_lr": 1e-7,
                "max_lr": 1.0,
                "num_training": 100,
                "mode": "exponential",
                "early_stop_threshold": 4.0,
            }

        # Set sensible defaults for batch size finder if not provided
        if self.batch_size_kwargs is None:
            self.batch_size_kwargs = {
                "mode": "power",
                "steps_per_trial": 3,
                "init_val": 16,
                "max_trials": 25,
            }


class AutoTrainer(pl.Trainer):
    """A configured ``pl.Trainer`` with sensible defaults for autotimm.

    This is a convenience class that wires up the logger, checkpointing,
    and automatic hyperparameter tuning. All ``**trainer_kwargs`` are
    forwarded to ``pl.Trainer``, so any Lightning Trainer argument works.

    **Auto-tuning is enabled by default** - both learning rate and batch size
    finding are enabled unless explicitly disabled via ``tuner_config``.

    Parameters:
        max_epochs: Number of training epochs.
        accelerator: ``"auto"``, ``"gpu"``, ``"cpu"``, ``"tpu"``, etc.
        devices: Number of devices or ``"auto"``.
        precision: Training precision (``32``, ``16``, ``"bf16-mixed"``, etc.).
        logger: A ``LoggerManager`` instance, a list of ``LoggerConfig``
            objects, a pre-built Logger instance/list, or ``False`` to
            disable logging.
        tuner_config: A ``TunerConfig`` instance to configure automatic learning
            rate and/or batch size finding. If ``None``, a default ``TunerConfig()``
            is created with both auto_lr and auto_batch_size enabled.
            To disable auto-tuning completely, pass ``False``.
        checkpoint_monitor: Metric to monitor for checkpointing (e.g.,
            ``"val/accuracy"``). If ``None``, no automatic checkpoint
            callback is added.
        checkpoint_mode: One of ``"min"`` or ``"max"`` for checkpoint
            monitoring.
        callbacks: List of Lightning callbacks.
        default_root_dir: Root directory for logs and checkpoints.
        gradient_clip_val: Gradient clipping value.
        accumulate_grad_batches: Gradient accumulation steps.
        val_check_interval: How often to run validation (1.0 = every epoch).
        enable_checkpointing: Whether to save model checkpoints.
        fast_dev_run: Runs a single batch through train, val, and test to
            find bugs quickly. Can be ``True`` (1 batch), ``False`` (disabled),
            or an integer (number of batches). Useful for debugging.
        seed: Random seed for reproducibility. If ``None``, no seeding is performed.
            Default is ``42`` for reproducible results.
        deterministic: If ``True`` (default), enables deterministic algorithms in PyTorch
            for full reproducibility (may impact performance). Set to ``False`` for faster training.
        use_autotimm_seeding: If ``True``, uses AutoTimm's custom ``seed_everything()``
            function which provides comprehensive seeding with deterministic mode support.
            If ``False`` (default), uses PyTorch Lightning's built-in seeding.
        json_progress: If ``True``, appends a ``JsonProgressCallback`` that
            emits NDJSON progress events to stdout for consumption by a
            Tauri/Preact frontend.  Default ``False``.
        json_progress_every_n_steps: How often to emit ``batch_end`` events
            when ``json_progress`` is enabled.  Default ``10``.
        json_progress_log_file: Optional path to an NDJSON file where all
            progress events are also appended.  Allows the frontend to
            replay missed events after a disconnect.  Default ``None``.
        **trainer_kwargs: Any other ``pl.Trainer`` argument.

    Example:
        >>> # Default: auto-tuning enabled (both LR and batch size)
        >>> trainer = AutoTrainer(max_epochs=10)
        >>> trainer.fit(model, datamodule=data)

        >>> # Disable all auto-tuning
        >>> trainer = AutoTrainer(max_epochs=10, tuner_config=False)
        >>> trainer.fit(model, datamodule=data)

        >>> # Custom tuning configuration
        >>> trainer = AutoTrainer(
        ...     max_epochs=10,
        ...     tuner_config=TunerConfig(
        ...         auto_lr=True,
        ...         auto_batch_size=False,
        ...         lr_find_kwargs={"min_lr": 1e-6, "max_lr": 1.0},
        ...     ),
        ... )
        >>> trainer.fit(model, datamodule=data)

        >>> # Quick debugging with fast_dev_run (auto-tuning disabled automatically)
        >>> trainer = AutoTrainer(fast_dev_run=True)
        >>> trainer.fit(model, datamodule=data)  # Runs 1 batch only

        >>> # Custom seeding for reproducibility
        >>> trainer = AutoTrainer(max_epochs=10, seed=123, deterministic=True)
        >>> trainer.fit(model, datamodule=data)

        >>> # Use Lightning's built-in seeding instead of AutoTimm's
        >>> trainer = AutoTrainer(max_epochs=10, seed=42, use_autotimm_seeding=False)
        >>> trainer.fit(model, datamodule=data)

        >>> # Disable seeding completely
        >>> trainer = AutoTrainer(max_epochs=10, seed=None)
        >>> trainer.fit(model, datamodule=data)
    """

    def __init__(
        self,
        max_epochs: int = 10,
        accelerator: str = "auto",
        devices: int | str = "auto",
        precision: str | int = 32,
        logger: (
            LoggerManager
            | list[LoggerConfig]
            | pl.loggers.Logger
            | list[pl.loggers.Logger]
            | bool
        ) = False,
        tuner_config: TunerConfig | None | bool = None,
        checkpoint_monitor: str | None = "val/loss",
        checkpoint_mode: str = "min",
        callbacks: list[pl.Callback] | None = None,
        default_root_dir: str = ".",
        gradient_clip_val: float | None = None,
        accumulate_grad_batches: int = 1,
        val_check_interval: float = 1.0,
        enable_checkpointing: bool = True,
        fast_dev_run: bool | int = False,
        seed: int | None = None,
        deterministic: bool = False,
        use_autotimm_seeding: bool = False,
        json_progress: bool = False,
        json_progress_every_n_steps: int = 10,
        json_progress_log_file: str | None = None,
        **trainer_kwargs: Any,
    ) -> None:
        # Print environment watermark on first AutoTrainer instantiation
        _print_watermark()

        # Store seeding params — actual seeding is deferred to fit()
        self._seed = seed
        self._deterministic = deterministic
        self._use_autotimm_seeding = use_autotimm_seeding

        if isinstance(logger, LoggerManager):
            resolved_logger = logger.loggers
        elif (
            isinstance(logger, list) and logger and isinstance(logger[0], LoggerConfig)
        ):
            manager = LoggerManager(configs=logger)
            resolved_logger = manager.loggers
        else:
            resolved_logger = logger

        # Copy rather than mutate the caller's list in place — appending
        # directly to it would silently modify a list the caller may reuse
        # elsewhere (e.g. across multiple AutoTrainer instances).
        callbacks = list(callbacks) if callbacks is not None else []

        if enable_checkpointing and checkpoint_monitor:
            has_checkpoint_cb = any(
                isinstance(cb, pl.callbacks.ModelCheckpoint) for cb in callbacks
            )
            if not has_checkpoint_cb:
                # Replace '/' in monitor name to avoid creating subdirectories
                safe_monitor = checkpoint_monitor.replace("/", "_")
                callbacks.append(
                    pl.callbacks.ModelCheckpoint(
                        monitor=checkpoint_monitor,
                        mode=checkpoint_mode,
                        save_top_k=1,
                        filename=f"best-{{epoch}}-{{{safe_monitor}:.4f}}",
                        auto_insert_metric_name=False,
                    )
                )

        # Store json_progress flag and log file for tuning event emission
        self._json_progress = json_progress
        self._json_progress_log_file = json_progress_log_file

        if json_progress:
            callbacks.append(
                JsonProgressCallback(
                    emit_every_n_steps=json_progress_every_n_steps,
                    log_file=json_progress_log_file,
                )
            )

        # Configure auto-tuning behavior
        # Disable auto-tuning during fast_dev_run or if explicitly set to False
        if fast_dev_run or tuner_config is False:
            self._tuner_config = None
        elif tuner_config is None or tuner_config is True:
            # Enable auto-tuning by default with sensible defaults.
            # Batch size finder is only useful on GPU instances where VRAM
            # probing is meaningful; disable it on CPU / MPS to avoid
            # unnecessary overhead or failures.
            import torch

            has_gpu = torch.cuda.is_available()
            self._tuner_config = TunerConfig(auto_batch_size=has_gpu)
        elif isinstance(tuner_config, TunerConfig):
            self._tuner_config = tuner_config
        else:
            raise TypeError(
                f"tuner_config must be a TunerConfig, None, True, or False, "
                f"got {type(tuner_config).__name__}"
            )

        # Guard flag: Lightning's Tuner internally calls trainer.fit(), which would
        # re-enter our fit() override and trigger a recursive tuning loop. This flag
        # breaks the cycle so that the inner fit() call skips _run_tuning().
        self._is_tuning = False

        super().__init__(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            logger=resolved_logger,
            callbacks=callbacks,
            default_root_dir=default_root_dir,
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            val_check_interval=val_check_interval,
            enable_checkpointing=enable_checkpointing,
            fast_dev_run=fast_dev_run,
            **trainer_kwargs,
        )

    def fit(
        self,
        model: pl.LightningModule,
        train_dataloaders: Any = None,
        val_dataloaders: Any = None,
        datamodule: pl.LightningDataModule | None = None,
        ckpt_path: str | None = None,
    ) -> None:
        """Fit the model, optionally running LR/batch size tuning first.

        If ``tuner_config`` was provided with ``auto_lr=True`` or
        ``auto_batch_size=True``, the respective tuning will run before
        training begins.

        Parameters:
            model: The LightningModule to train.
            train_dataloaders: Train dataloaders (if not using datamodule).
            val_dataloaders: Validation dataloaders (if not using datamodule).
            datamodule: A LightningDataModule instance.
            ckpt_path: Path to checkpoint for resuming training.
        """
        # Ensure fork-based multiprocessing on macOS to avoid spawn guard issues
        _ensure_safe_multiprocessing()

        # Apply seeding at the start of fit() so trainer seed is authoritative
        self._apply_seeding()

        if self._tuner_config is not None and not self._is_tuning:
            self._is_tuning = True
            try:
                if self._json_progress:
                    _emit(
                        {"event": "tuning_started"},
                        log_file=self._json_progress_log_file,
                    )
                self._run_tuning(model, train_dataloaders, val_dataloaders, datamodule)
                if self._json_progress:
                    _emit(
                        {"event": "tuning_complete"},
                        log_file=self._json_progress_log_file,
                    )
            finally:
                self._is_tuning = False

        super().fit(
            model,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )

    def test(
        self,
        model: pl.LightningModule | None = None,
        dataloaders: Any = None,
        datamodule: pl.LightningDataModule | None = None,
        ckpt_path: str | None = "best",
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """Test the model, logging results to configured loggers (CSV, JSONL, etc.).

        Ensures multiprocessing safety and seeding are applied before
        running the test loop. All configured loggers (CSVLogger,
        TensorBoard, etc.) and callbacks (``JsonProgressCallback``)
        remain active during testing.

        Parameters:
            model: The LightningModule to test. If ``None``, uses the
                model from the last ``fit()`` call.
            dataloaders: Test dataloaders (if not using datamodule).
            datamodule: A LightningDataModule instance.
            ckpt_path: Path to checkpoint to load for testing.
                Defaults to ``"best"`` (best checkpoint from training).
            verbose: Whether to print test results.

        Returns:
            List of dictionaries containing test metrics.
        """
        _ensure_safe_multiprocessing()
        self._apply_seeding()

        # Resolve ckpt_path="best" robustly — handles both in-process
        # (fit then test) and separate-process (CLI test after fit) scenarios.
        if ckpt_path == "best":
            ckpt_path = self._resolve_best_ckpt_path()

        return super().test(
            model=model,
            dataloaders=dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            verbose=verbose,
        )

    def _resolve_best_ckpt_path(self) -> str | None:
        """Resolve ``ckpt_path="best"`` to an actual file path.

        Handles three scenarios:
        1. A ``ModelCheckpoint`` callback has ``best_model_path`` populated
           (in-process fit→test) — returns that path directly.
        2. No ``ModelCheckpoint`` with a monitor is configured — falls back
           to ``None`` (current model weights).
        3. A ``ModelCheckpoint`` is configured but ``best_model_path`` is empty
           (separate-process test after fit, e.g. NightFlow CLI) — searches
           the checkpoint directory on disk for ``best-*.ckpt`` files and
           returns the most recent one.
        """
        from pathlib import Path

        ckpt_cb = None
        for cb in self.callbacks or []:
            if isinstance(cb, pl.callbacks.ModelCheckpoint) and cb.monitor is not None:
                ckpt_cb = cb
                break

        if ckpt_cb is None:
            logger.warning(
                "No ModelCheckpoint configured with a monitored metric. "
                "Falling back to ckpt_path=None (current model weights) "
                'instead of ckpt_path="best".'
            )
            return None

        # In-process: fit() already set best_model_path
        if ckpt_cb.best_model_path:
            return "best"

        # Separate-process: search for checkpoint files on disk
        ckpt_dir = Path(ckpt_cb.dirpath) if ckpt_cb.dirpath else None

        # If dirpath is not set, try the default Lightning checkpoint dir
        if ckpt_dir is None:
            root = Path(self.default_root_dir)
            # Check common checkpoint locations
            for candidate in [
                root / "checkpoints",
                root,
            ]:
                if candidate.is_dir():
                    ckpts = sorted(
                        candidate.glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime
                    )
                    if ckpts:
                        ckpt_dir = candidate
                        break

            # Also check logger-specific directories
            if ckpt_dir is None and self.loggers:
                for lg in self.loggers:
                    log_dir = getattr(lg, "log_dir", None)
                    if log_dir:
                        candidate = Path(log_dir) / "checkpoints"
                        if candidate.is_dir():
                            ckpts = sorted(
                                candidate.glob("best-*.ckpt"),
                                key=lambda p: p.stat().st_mtime,
                            )
                            if ckpts:
                                ckpt_dir = candidate
                                break

        if ckpt_dir and ckpt_dir.is_dir():
            ckpts = sorted(
                ckpt_dir.glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime
            )
            if ckpts:
                best = str(ckpts[-1])
                logger.info(f"Found best checkpoint on disk: {best}")
                return best

        logger.warning(
            'ckpt_path="best" requested but no checkpoint file found on disk. '
            "Falling back to ckpt_path=None (current model weights)."
        )
        return None

    def _apply_seeding(self) -> None:
        """Apply seeding based on stored parameters."""
        if self._seed is not None and self._use_autotimm_seeding:
            seed_everything(self._seed, deterministic=self._deterministic)
        elif self._seed is not None:
            import pytorch_lightning as pl_seed

            pl_seed.seed_everything(self._seed, workers=True)
            if self._deterministic:
                import torch

                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        elif self._seed is None and self._deterministic:
            import warnings

            warnings.warn(
                "deterministic=True has no effect when seed=None. "
                "Set a seed value to enable deterministic behavior.",
                stacklevel=2,
            )

    def _remove_tuner_callbacks(self) -> None:
        """Remove BatchSizeFinder/LearningRateFinder callbacks injected by Tuner.

        In Lightning >= 2.x, Tuner methods permanently attach their callbacks
        to the trainer. Calling both scale_batch_size and lr_find sequentially
        (or calling fit() more than once) triggers a conflict. This method
        cleans them up after each tuning step.
        """
        try:
            from pytorch_lightning.callbacks import BatchSizeFinder, LearningRateFinder

            self.callbacks[:] = [
                cb
                for cb in self.callbacks
                if not isinstance(cb, (BatchSizeFinder, LearningRateFinder))
            ]
        except ImportError:
            pass

    def _run_tuning(
        self,
        model: pl.LightningModule,
        train_dataloaders: Any,
        val_dataloaders: Any,
        datamodule: pl.LightningDataModule | None,
    ) -> None:
        """Run automatic hyperparameter tuning."""
        tuner = Tuner(self)
        config = self._tuner_config

        # Run batch size finder first (if enabled)
        # This should run before LR finder since LR depends on batch size
        if config.auto_batch_size:
            logger.info("Running batch size finder...")
            try:
                result = tuner.scale_batch_size(
                    model,
                    train_dataloaders=train_dataloaders,
                    val_dataloaders=val_dataloaders,
                    datamodule=datamodule,
                    **config.batch_size_kwargs,
                )
                logger.info(f"Optimal batch size found: {result}")
            except Exception as e:
                logger.error(f"Batch size finder failed: {e}")
                logger.info("Continuing with user-specified batch size.")
            finally:
                # Remove the BatchSizeFinder callback Tuner injected so the
                # subsequent lr_find call (and any future fit() call) won't
                # see a duplicate and raise an error.
                self._remove_tuner_callbacks()

        # Run LR finder (if enabled)
        if config.auto_lr:
            logger.info("Running learning rate finder...")
            try:
                lr_finder = tuner.lr_find(
                    model,
                    train_dataloaders=train_dataloaders,
                    val_dataloaders=val_dataloaders,
                    datamodule=datamodule,
                    **config.lr_find_kwargs,
                )
                if lr_finder is not None:
                    suggested_lr = lr_finder.suggestion()
                    if suggested_lr is not None:
                        logger.info(f"Suggested learning rate: {suggested_lr:.2e}")
                        # Update model's learning rate
                        if hasattr(model, "_lr"):
                            model._lr = suggested_lr
                        elif hasattr(model, "lr"):
                            model.lr = suggested_lr
                        elif hasattr(model, "hparams") and "lr" in model.hparams:
                            model.hparams.lr = suggested_lr
                    else:
                        logger.warning("LR finder could not suggest a learning rate.")
            except Exception as e:
                logger.error(f"LR finder failed: {e}")
                logger.info("Continuing with user-specified learning rate.")
            finally:
                # Remove the LearningRateFinder callback so fit() starts clean.
                self._remove_tuner_callbacks()

    @property
    def tuner_config(self) -> TunerConfig | None:
        """Return the tuner configuration."""
        return self._tuner_config
