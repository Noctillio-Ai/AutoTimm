"""PyTorch Lightning callbacks for automatic interpretation during training."""

from typing import Literal

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from autotimm.core.logging import logger
from autotimm.interpretation.gradcam import GradCAM
from autotimm.interpretation.integrated_gradients import IntegratedGradients


class InterpretationCallback(Callback):
    """
    Automatically generate and log interpretations during training.

    This callback periodically generates explanations for sample images
    and logs them to TensorBoard, Weights & Biases, or MLflow. Useful
    for monitoring what the model learns during training.

    Args:
        sample_images: Images to explain (tensor, list of tensors, or paths)
        sample_labels: Optional ground truth labels for the images
        method: Interpretation method ('gradcam', 'gradcam++', 'integrated_gradients')
        target_layer: Layer to use (None = auto-detect)
        log_every_n_epochs: Generate explanations every N epochs
        log_every_n_steps: Alternative: log every N steps (overrides epochs)
        num_samples: Number of images to explain (if more provided, random sample)
        colormap: Matplotlib colormap for heatmaps
        alpha: Overlay transparency
        prefix: Prefix for logged images (default: "interpretation")

    Example:
        >>> from autotimm import AutoTrainer, ImageClassifier
        >>> from autotimm.interpretation.callbacks import InterpretationCallback
        >>>
        >>> # Sample images for monitoring
        >>> sample_images = [load_image(f"sample_{i}.jpg") for i in range(8)]
        >>>
        >>> # Create callback
        >>> interp_callback = InterpretationCallback(
        ...     sample_images=sample_images,
        ...     method="gradcam",
        ...     log_every_n_epochs=5,
        ...     num_samples=8,
        ... )
        >>>
        >>> # Add to trainer
        >>> trainer = AutoTrainer(
        ...     max_epochs=100,
        ...     callbacks=[interp_callback],
        ... )
        >>> trainer.fit(model, datamodule=data)
    """

    def __init__(
        self,
        sample_images: torch.Tensor | list[torch.Tensor] | list[str],
        sample_labels: list[int] | None = None,
        method: Literal["gradcam", "gradcam++", "integrated_gradients"] = "gradcam",
        target_layer: str | torch.nn.Module | None = None,
        log_every_n_epochs: int = 5,
        log_every_n_steps: int | None = None,
        num_samples: int = 8,
        colormap: str = "viridis",
        alpha: float = 0.4,
        prefix: str = "interpretation",
    ):
        super().__init__()
        self.sample_images = self._prepare_images(sample_images, num_samples)
        self.sample_labels = sample_labels
        self.method = method
        self.target_layer = target_layer
        self.log_every_n_epochs = log_every_n_epochs
        self.log_every_n_steps = log_every_n_steps
        self.num_samples = num_samples
        self.colormap = colormap
        self.alpha = alpha
        self.prefix = prefix
        self.explainer = None

    def _prepare_images(
        self,
        images: torch.Tensor | list[torch.Tensor] | list[str],
        num_samples: int,
    ) -> list[torch.Tensor]:
        """Prepare and sample images."""
        # Load images if paths
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], str):
            import torchvision.transforms as T
            from PIL import Image

            transform = T.Compose([T.ToTensor()])
            images = [transform(Image.open(img).convert("RGB")) for img in images]

        # Convert single tensor to list
        if isinstance(images, torch.Tensor):
            if images.ndim == 4:
                images = [images[i] for i in range(images.shape[0])]
            else:
                images = [images]

        # Sample if too many
        if len(images) > num_samples:
            indices = np.random.choice(len(images), num_samples, replace=False)
            images = [images[i] for i in indices]

        return images

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Initialize explainer when training starts."""
        # Create explainer
        if self.method == "gradcam":
            self.explainer = GradCAM(
                pl_module, target_layer=self.target_layer, use_cuda=True
            )
        elif self.method == "gradcam++":
            from autotimm.interpretation import GradCAMPlusPlus

            self.explainer = GradCAMPlusPlus(
                pl_module, target_layer=self.target_layer, use_cuda=True
            )
        elif self.method == "integrated_gradients":
            self.explainer = IntegratedGradients(pl_module, baseline="black", steps=30)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Generate and log interpretations at epoch end."""
        # Check if should log this epoch
        if self.log_every_n_steps is None:
            if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
                return
        else:
            # Skip if using step-based logging
            return

        self._generate_and_log(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        """Generate and log interpretations at batch end (if step-based)."""
        if self.log_every_n_steps is not None:
            if (trainer.global_step + 1) % self.log_every_n_steps == 0:
                self._generate_and_log(trainer, pl_module)

    def _generate_and_log(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Generate interpretations and log to available loggers."""
        # Set model to eval mode
        was_training = pl_module.training
        pl_module.eval()

        try:
            # Generate explanations
            visualizations = []
            for idx, image in enumerate(self.sample_images):
                # Move to device
                image_tensor = image.unsqueeze(0).to(pl_module.device)

                # Get prediction
                with torch.inference_mode():
                    output = pl_module(image_tensor)
                    if isinstance(output, dict):
                        output = output.get("logits", output.get("output", output))
                    pred_class = output.argmax(dim=1).item()

                # Generate heatmap
                heatmap = self.explainer.explain(image_tensor, target_class=pred_class)

                # Create visualization
                viz = self._create_visualization(image, heatmap, pred_class)
                visualizations.append(viz)

            # Log to available loggers
            self._log_visualizations(trainer, visualizations)

        finally:
            # Restore training mode
            if was_training:
                pl_module.train()

    def _create_visualization(
        self,
        image: torch.Tensor,
        heatmap: np.ndarray,
        pred_class: int,
    ) -> np.ndarray:
        """Create overlay visualization."""
        from autotimm.interpretation.visualization.heatmap import overlay_heatmap

        # Convert image to numpy
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        # Create overlay
        overlayed = overlay_heatmap(
            image_np,
            heatmap,
            alpha=self.alpha,
            colormap=self.colormap,
        )

        return overlayed

    def _log_visualizations(
        self,
        trainer: pl.Trainer,
        visualizations: list[np.ndarray],
    ):
        """Log visualizations to available loggers."""
        epoch = trainer.current_epoch
        step = trainer.global_step

        for log in trainer.loggers:
            logger_name = log.__class__.__name__.lower()

            if "tensorboard" in logger_name:
                self._log_to_tensorboard(log, visualizations, step)
            elif "wandb" in logger_name:
                self._log_to_wandb(log, visualizations, epoch, step)
            elif "mlflow" in logger_name:
                self._log_to_mlflow(log, visualizations, epoch, step)

    def _log_to_tensorboard(self, logger, visualizations: list[np.ndarray], step: int):
        """Log to TensorBoard."""
        try:
            writer = logger.experiment

            for idx, viz in enumerate(visualizations):
                # Convert to CHW format
                viz_tensor = torch.from_numpy(viz).permute(2, 0, 1).float() / 255.0
                writer.add_image(
                    f"{self.prefix}/sample_{idx}",
                    viz_tensor,
                    global_step=step,
                )
        except Exception as e:
            logger.warning(f"Failed to log to TensorBoard: {e}")

    def _log_to_wandb(
        self, logger, visualizations: list[np.ndarray], epoch: int, step: int
    ):
        """Log to Weights & Biases."""
        try:
            import wandb

            images = [
                wandb.Image(viz, caption=f"Sample {idx}")
                for idx, viz in enumerate(visualizations)
            ]
            logger.experiment.log(
                {f"{self.prefix}": images},
                step=step,
            )
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")

    def _log_to_mlflow(
        self, logger, visualizations: list[np.ndarray], epoch: int, step: int
    ):
        """Log to MLflow."""
        try:
            import tempfile

            import mlflow
            from PIL import Image

            for idx, viz in enumerate(visualizations):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    Image.fromarray(viz).save(tmp.name)
                    mlflow.log_artifact(
                        tmp.name, artifact_path=f"{self.prefix}/epoch_{epoch}"
                    )
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")


class FeatureMonitorCallback(Callback):
    """
    Monitor feature statistics during training.

    Tracks mean activation, sparsity, and other statistics for specified
    layers to understand how features evolve during training.

    Args:
        layer_names: Names of layers to monitor
        log_every_n_epochs: Log statistics every N epochs
        num_batches: Number of batches to accumulate statistics over

    Example:
        >>> callback = FeatureMonitorCallback(
        ...     layer_names=["backbone.layer2", "backbone.layer3", "backbone.layer4"],
        ...     log_every_n_epochs=1,
        ... )
        >>> trainer = AutoTrainer(callbacks=[callback])
    """

    def __init__(
        self,
        layer_names: list[str],
        log_every_n_epochs: int = 1,
        num_batches: int = 10,
    ):
        super().__init__()
        self.layer_names = layer_names
        self.log_every_n_epochs = log_every_n_epochs
        self.num_batches = num_batches
        self.hooks = []
        self.activations = {name: [] for name in layer_names}

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Register hooks at epoch start if logging this epoch."""
        if (trainer.current_epoch + 1) % self.log_every_n_epochs == 0:
            self._register_hooks(pl_module)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        """Accumulate activations."""
        if len(self.hooks) > 0 and batch_idx < self.num_batches:
            # Activations are being collected via hooks
            pass

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Compute and log statistics."""
        if len(self.hooks) > 0:
            self._compute_and_log_statistics(trainer)
            self._remove_hooks()

    def _register_hooks(self, pl_module: pl.LightningModule):
        """Register hooks to capture activations."""
        for name in self.layer_names:
            layer = self._get_layer_by_name(pl_module, name)
            if layer is not None:
                hook = layer.register_forward_hook(
                    lambda module, input, output, name=name: self.activations[
                        name
                    ].append(output.detach().cpu())
                )
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {name: [] for name in self.layer_names}

    def _get_layer_by_name(self, module: torch.nn.Module, name: str):
        """Get layer by name."""
        try:
            parts = name.split(".")
            layer = module
            for part in parts:
                layer = getattr(layer, part)
            return layer
        except AttributeError:
            logger.warning(f"Layer {name} not found")
            return None

    def _compute_and_log_statistics(self, trainer: pl.Trainer):
        """Compute and log feature statistics."""
        stats = {}

        for name, acts in self.activations.items():
            if len(acts) == 0:
                continue

            # Concatenate activations
            all_acts = torch.cat(acts, dim=0)

            # Compute statistics
            mean_act = all_acts.mean().item()
            std_act = all_acts.std().item()
            sparsity = (all_acts == 0).float().mean().item()
            max_act = all_acts.max().item()

            stats[f"features/{name}/mean"] = mean_act
            stats[f"features/{name}/std"] = std_act
            stats[f"features/{name}/sparsity"] = sparsity
            stats[f"features/{name}/max"] = max_act

        # Log to all loggers
        for log in trainer.loggers:
            try:
                log.log_metrics(stats, step=trainer.global_step)
            except Exception as e:
                logger.warning(f"Failed to log feature stats: {e}")


__all__ = [
    "FeatureMonitorCallback",
    "InterpretationCallback",
]
