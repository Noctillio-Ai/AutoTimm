"""Loss function registry for centralized loss management."""

from __future__ import annotations

from typing import Any, ClassVar

from torch import nn

from autotimm.losses.detection import (
    CenternessLoss,
    FCOSLoss,
    FocalLoss,
    GIoULoss,
)
from autotimm.losses.segmentation import (
    CombinedSegmentationLoss,
    DiceLoss,
    FocalLossPixelwise,
    MaskLoss,
    TverskyLoss,
)


class LossRegistry:
    """Central registry for all loss functions available in AutoTimm.

    This registry provides a unified interface for accessing and instantiating
    loss functions across different tasks (classification, detection, segmentation).

    Usage:
        >>> # Get available losses
        >>> registry = LossRegistry()
        >>> print(registry.list_losses())

        >>> # Create a loss function
        >>> loss_fn = registry.get_loss("dice", num_classes=10)

        >>> # Register a custom loss
        >>> registry.register_loss("custom_dice", MyCustomDiceLoss)
    """

    # Classification losses
    CLASSIFICATION_LOSSES: ClassVar[dict[str, type[nn.Module]]] = {
        "cross_entropy": nn.CrossEntropyLoss,
        "bce": nn.BCEWithLogitsLoss,
        "bce_with_logits": nn.BCEWithLogitsLoss,
        "nll": nn.NLLLoss,
        "mse": nn.MSELoss,
    }

    # Detection losses
    DETECTION_LOSSES: ClassVar[dict[str, type[nn.Module]]] = {
        "focal": FocalLoss,
        "giou": GIoULoss,
        "centerness": CenternessLoss,
        "fcos": FCOSLoss,
    }

    # Segmentation losses
    SEGMENTATION_LOSSES: ClassVar[dict[str, type[nn.Module]]] = {
        "dice": DiceLoss,
        "focal_pixelwise": FocalLossPixelwise,
        "tversky": TverskyLoss,
        "mask": MaskLoss,
        "combined_segmentation": CombinedSegmentationLoss,
    }

    def __init__(self):
        """Initialize the loss registry with all available losses."""
        self._registry = {}
        self._registry.update(self.CLASSIFICATION_LOSSES)
        self._registry.update(self.DETECTION_LOSSES)
        self._registry.update(self.SEGMENTATION_LOSSES)

        # Add aliases for common usage
        self._aliases = {
            "ce": "cross_entropy",
            "bce": "bce_with_logits",
            "focal": "focal",
            "combined": "combined_segmentation",
        }

    def register_loss(
        self,
        name: str,
        loss_class: type[nn.Module],
        alias: str | None = None,
    ) -> None:
        """Register a custom loss function.

        Args:
            name: Name of the loss function.
            loss_class: The loss class (must be a subclass of nn.Module).
            alias: Optional alias for the loss.

        Raises:
            ValueError: If the loss class is not a subclass of nn.Module.

        Example:
            >>> registry = LossRegistry()
            >>> registry.register_loss("my_loss", MyLossClass, alias="ml")
        """
        if not issubclass(loss_class, nn.Module):
            raise ValueError(
                f"loss_class must be a subclass of nn.Module, got {loss_class}"
            )

        self._registry[name] = loss_class
        if alias:
            self._aliases[alias] = name

    def get_loss(
        self,
        name: str,
        **kwargs: Any,
    ) -> nn.Module:
        """Get a loss function instance by name.

        Args:
            name: Name or alias of the loss function.
            **kwargs: Keyword arguments to pass to the loss constructor.

        Returns:
            Instantiated loss function.

        Raises:
            ValueError: If the loss name is not found in the registry.

        Example:
            >>> registry = LossRegistry()
            >>> dice_loss = registry.get_loss("dice", num_classes=10)
            >>> focal_loss = registry.get_loss("focal", alpha=0.25, gamma=2.0)
        """
        # Resolve alias
        resolved_name = self._aliases.get(name, name)

        if resolved_name not in self._registry:
            available = self.list_losses()
            raise ValueError(
                f"Loss '{name}' not found in registry. Available losses: {available}"
            )

        loss_class = self._registry[resolved_name]
        return loss_class(**kwargs)

    def list_losses(self, task: str | None = None) -> list[str]:
        """List all available loss functions.

        Args:
            task: Optional task filter. One of: 'classification', 'detection', 'segmentation'.
                If None, returns all losses.

        Returns:
            List of available loss names.

        Example:
            >>> registry = LossRegistry()
            >>> registry.list_losses()  # All losses
            >>> registry.list_losses(task="segmentation")  # Only segmentation losses
        """
        if task is None:
            return sorted(self._registry.keys())

        task = task.lower()
        if task == "classification":
            return sorted(self.CLASSIFICATION_LOSSES.keys())
        elif task == "detection":
            return sorted(self.DETECTION_LOSSES.keys())
        elif task == "segmentation":
            return sorted(self.SEGMENTATION_LOSSES.keys())
        else:
            raise ValueError(
                f"Unknown task: {task}. "
                f"Use 'classification', 'detection', or 'segmentation'."
            )

    def has_loss(self, name: str) -> bool:
        """Check if a loss function is registered.

        Args:
            name: Name or alias of the loss function.

        Returns:
            True if the loss is registered, False otherwise.

        Example:
            >>> registry = LossRegistry()
            >>> registry.has_loss("dice")  # True
            >>> registry.has_loss("unknown")  # False
        """
        resolved_name = self._aliases.get(name, name)
        return resolved_name in self._registry

    def get_loss_info(self) -> dict[str, dict[str, list[str]]]:
        """Get information about all registered losses organized by task.

        Returns:
            Dictionary with task categories and their loss functions.

        Example:
            >>> registry = LossRegistry()
            >>> info = registry.get_loss_info()
            >>> print(info["segmentation"])  # List of segmentation losses
        """
        return {
            "classification": self.list_losses(task="classification"),
            "detection": self.list_losses(task="detection"),
            "segmentation": self.list_losses(task="segmentation"),
        }


# Global instance for convenience
_global_registry = LossRegistry()


def get_loss_registry() -> LossRegistry:
    """Get the global loss registry instance.

    Returns:
        The global LossRegistry instance.

    Example:
        >>> from autotimm.losses import get_loss_registry
        >>> registry = get_loss_registry()
        >>> loss = registry.get_loss("dice", num_classes=10)
    """
    return _global_registry


def register_custom_loss(
    name: str,
    loss_class: type[nn.Module],
    alias: str | None = None,
) -> None:
    """Register a custom loss function in the global registry.

    Args:
        name: Name of the loss function.
        loss_class: The loss class (must be a subclass of nn.Module).
        alias: Optional alias for the loss.

    Example:
        >>> from autotimm.losses import register_custom_loss
        >>> register_custom_loss("my_loss", MyLossClass, alias="ml")
    """
    _global_registry.register_loss(name, loss_class, alias)


def list_available_losses(task: str | None = None) -> list[str]:
    """List all available loss functions.

    Args:
        task: Optional task filter. One of: 'classification', 'detection', 'segmentation'.

    Returns:
        List of available loss names.

    Example:
        >>> from autotimm.losses import list_available_losses
        >>> print(list_available_losses())  # All losses
        >>> print(list_available_losses(task="segmentation"))  # Only segmentation losses
    """
    return _global_registry.list_losses(task=task)
