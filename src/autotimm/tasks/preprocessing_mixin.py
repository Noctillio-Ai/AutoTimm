"""Mixin class that adds preprocessing capabilities to model classes."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from autotimm.data.transform_config import TransformConfig


class PreprocessingMixin:
    """Mixin that adds preprocessing capabilities to model classes.

    This mixin provides:
    1. ``_setup_transforms()``: Initialize transforms from TransformConfig
    2. ``preprocess()``: Preprocess raw images for model inference
    3. ``get_data_config()``: Get the model's data configuration

    When added to a model class, it enables easy inference-time preprocessing
    using the model's native normalization statistics from timm.

    Example:
        >>> class ImageClassifier(PreprocessingMixin, pl.LightningModule):
        ...     def __init__(self, ..., transform_config=None):
        ...         super().__init__()
        ...         self._setup_transforms(transform_config)
        ...
        >>> model = ImageClassifier(backbone="resnet50", ...)
        >>> image = Image.open("test.jpg")
        >>> tensor = model.preprocess(image)
        >>> output = model(tensor)
    """

    # These will be set by _setup_transforms
    _transform_config: TransformConfig | None
    _eval_transform: Any
    _train_transform: Any
    _data_config: dict[str, Any]

    def _setup_transforms(
        self,
        transform_config: TransformConfig | None,
        task: str = "classification",
    ) -> None:
        """Initialize transforms from TransformConfig.

        This should be called in the model's ``__init__()`` after creating
        the backbone. It sets up both training and evaluation transforms
        using model-specific normalization from timm.

        Args:
            transform_config: TransformConfig instance. If None, a default
                config with ``use_timm_config=True`` is used.
            task: Task type for transform creation. One of "classification",
                "detection", or "segmentation".
        """
        from autotimm.data.timm_transforms import (
            get_transforms_from_backbone,
            resolve_backbone_data_config,
        )

        # Store config
        self._transform_config = transform_config

        if transform_config is None:
            # No config provided - create minimal setup for backward compatibility
            self._eval_transform = None
            self._train_transform = None
            self._data_config = {}
            return

        # Get the backbone - it should be set on self by the model class
        backbone = getattr(self, "backbone", None)
        if backbone is None:
            # Try to get backbone name from hparams if available
            if hasattr(self, "hparams") and hasattr(self.hparams, "backbone"):
                backbone = self.hparams.backbone
            else:
                # Fall back to ImageNet defaults
                backbone = "resnet50"

        # Resolve data config from backbone
        self._data_config = resolve_backbone_data_config(
            backbone,
            override_mean=transform_config.mean,
            override_std=transform_config.std,
            override_interpolation=transform_config.interpolation,
            override_crop_pct=transform_config.crop_pct,
        )

        # Create eval transform (always needed for preprocess())
        self._eval_transform = get_transforms_from_backbone(
            backbone=backbone,
            transform_config=transform_config,
            is_train=False,
            task=task,
        )

        # Create train transform (may be used by some workflows)
        self._train_transform = get_transforms_from_backbone(
            backbone=backbone,
            transform_config=transform_config,
            is_train=True,
            task=task,
        )

    def preprocess(
        self,
        images: Image.Image | list[Image.Image] | torch.Tensor,
        is_train: bool = False,
    ) -> torch.Tensor:
        """Preprocess raw images for model inference.

        This method handles various input formats and returns a properly
        normalized tensor ready for model forward pass.

        Args:
            images: Input images in one of these formats:
                - Single PIL Image
                - List of PIL Images
                - Numpy array (H, W, 3) or (B, H, W, 3)
                - Torch tensor (already preprocessed, will be returned as-is)
            is_train: If True, use training transforms with augmentation.
                If False (default), use evaluation transforms.

        Returns:
            Tensor of shape (B, C, H, W) ready for model input.
            For a single image, B=1.

        Raises:
            RuntimeError: If transforms were not set up (no TransformConfig
                was provided to the model).

        Example:
            >>> model = ImageClassifier(
            ...     backbone="resnet50",
            ...     num_classes=10,
            ...     transform_config=TransformConfig(),
            ... )
            >>> image = Image.open("test.jpg")
            >>> tensor = model.preprocess(image)
            >>> assert tensor.shape == (1, 3, 224, 224)
            >>> output = model(tensor)
        """
        import numpy as np

        # Select transform
        transform = self._train_transform if is_train else self._eval_transform

        if transform is None:
            raise RuntimeError(
                "Transforms not set up. Provide a TransformConfig to the model "
                "constructor to enable preprocessing."
            )

        # Handle torch tensor - assume already preprocessed
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = images.unsqueeze(0)
            return images

        # Handle numpy array
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                # Single image (H, W, C)
                images = [Image.fromarray(images)]
            else:
                # Batch (B, H, W, C)
                images = [Image.fromarray(img) for img in images]

        # Handle single PIL Image
        if isinstance(images, Image.Image):
            images = [images]

        # Convert to RGB if needed and apply transform
        tensors = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Check if we're using albumentations (has a processors attribute)
            # Note: both torchvision and albumentations have 'transforms' attribute
            is_albumentations = hasattr(transform, "processors") or (
                hasattr(transform, "__class__")
                and transform.__class__.__module__.startswith("albumentations")
            )

            if is_albumentations:
                # Albumentations - needs numpy array
                import numpy as np

                img_np = np.array(img)
                result = transform(image=img_np)
                tensor = result["image"]
            else:
                # Torchvision
                tensor = transform(img)

            tensors.append(tensor)

        # Stack into batch
        return torch.stack(tensors, dim=0)

    def get_data_config(self) -> dict[str, Any]:
        """Get the model's data configuration.

        Returns the normalization and input specifications that the model
        expects. This is useful for creating compatible data pipelines.

        Returns:
            Dictionary with keys:
                - ``mean``: Tuple of 3 floats for normalization mean.
                - ``std``: Tuple of 3 floats for normalization std.
                - ``input_size``: Tuple (C, H, W) for expected input size.
                - ``interpolation``: String interpolation mode.
                - ``crop_pct``: Float center crop percentage.

        Raises:
            RuntimeError: If transforms were not set up.

        Example:
            >>> model = ImageClassifier(
            ...     backbone="vit_base_patch16_224",
            ...     transform_config=TransformConfig(),
            ... )
            >>> config = model.get_data_config()
            >>> print(config["mean"])  # (0.5, 0.5, 0.5) for ViT
        """
        if not hasattr(self, "_data_config") or not self._data_config:
            raise RuntimeError(
                "Data config not available. Provide a TransformConfig to the "
                "model constructor to enable this feature."
            )
        return self._data_config.copy()

    def get_transform(self, is_train: bool = False) -> Any:
        """Get the transform pipeline.

        Args:
            is_train: If True, return training transforms.
                If False, return evaluation transforms.

        Returns:
            The transform pipeline (Compose object).

        Raises:
            RuntimeError: If transforms were not set up.
        """
        transform = self._train_transform if is_train else self._eval_transform
        if transform is None:
            raise RuntimeError(
                "Transforms not set up. Provide a TransformConfig to the model "
                "constructor to enable this feature."
            )
        return transform
