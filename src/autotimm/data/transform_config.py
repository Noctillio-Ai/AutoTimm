"""Unified transform configuration for AutoTimm models and data modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class TransformConfig:
    """Unified configuration for image transforms.

    This dataclass provides a consistent interface for configuring transforms
    across models and data modules. When used with models, it enables
    automatic preprocessing using model-specific normalization from timm.

    Attributes:
        preset: Transform preset name. Common presets:
            - ``"default"``: Standard augmentation (random crop, flip, color jitter)
            - ``"autoaugment"``: AutoAugment policy
            - ``"randaugment"``: RandAugment with configurable ops/magnitude
            - ``"trivialaugment"``: TrivialAugmentWide
            - ``"strong"``: Heavy augmentation (albumentations only)
            - ``"light"``: Light augmentation (minimal transforms)
        backend: Transform backend to use. Either ``"torchvision"`` (PIL-based)
            or ``"albumentations"`` (OpenCV-based).
        image_size: Target image size (square). This is used for both training
            (random resized crop) and evaluation (resize + center crop).
        use_timm_config: If True, get mean/std/input_size from the timm model's
            data config. This ensures the model receives inputs normalized
            with the same statistics it was pretrained with.
        mean: Override normalization mean. If None and use_timm_config is True,
            uses the model's pretrained mean. Otherwise defaults to ImageNet.
        std: Override normalization std. If None and use_timm_config is True,
            uses the model's pretrained std. Otherwise defaults to ImageNet.
        interpolation: Interpolation mode for resizing. Common values:
            ``"bilinear"``, ``"bicubic"``, ``"lanczos"``. If ``None``
            (default) and ``use_timm_config`` is True, uses the model's
            pretrained interpolation; otherwise defaults to ``"bicubic"``.
        crop_pct: Center crop percentage for evaluation transforms.
            For a 224x224 image with crop_pct=0.875, the image is first
            resized to 256x256 (224/0.875) then center cropped. If ``None``
            (default) and ``use_timm_config`` is True, uses the model's
            pretrained crop_pct; otherwise defaults to ``0.875``.

        Detection-specific options:
        min_bbox_area: Minimum bounding box area to keep after transforms.
        min_visibility: Minimum visibility ratio for bboxes (0.0-1.0).
        bbox_format: Bounding box format (``"coco"``, ``"pascal_voc"``, ``"yolo"``).

        Segmentation-specific options:
        ignore_index: Label index to ignore in segmentation masks.

    Example:
        >>> from autotimm import ImageClassifier, TransformConfig
        >>> config = TransformConfig(
        ...     preset="randaugment",
        ...     image_size=384,
        ...     use_timm_config=True,
        ... )
        >>> model = ImageClassifier(
        ...     backbone="efficientnet_b4",
        ...     num_classes=10,
        ...     metrics=[...],
        ...     transform_config=config,
        ... )
        >>> # Now model.preprocess() uses the correct normalization
        >>> tensor = model.preprocess(pil_image)
    """

    # General transform options
    preset: str = "default"
    backend: Literal["torchvision", "albumentations"] = "torchvision"
    image_size: int = 224
    use_timm_config: bool = True
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    interpolation: str | None = None
    crop_pct: float | None = None

    # Detection-specific options
    min_bbox_area: float = 0.0
    min_visibility: float = 0.0
    bbox_format: str = "coco"

    # Segmentation-specific options
    ignore_index: int = 255

    # RandAugment-specific options (when preset="randaugment")
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 9

    def __post_init__(self):
        """Validate configuration values."""
        valid_backends = ("torchvision", "albumentations")
        if self.backend not in valid_backends:
            raise ValueError(
                f"Invalid backend '{self.backend}'. Choose from: {valid_backends}"
            )

        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")

        if self.crop_pct is not None and not 0.0 < self.crop_pct <= 1.0:
            raise ValueError(f"crop_pct must be in (0, 1], got {self.crop_pct}")

        if self.mean is not None and len(self.mean) != 3:
            raise ValueError(f"mean must have 3 values, got {len(self.mean)}")

        if self.std is not None and len(self.std) != 3:
            raise ValueError(f"std must have 3 values, got {len(self.std)}")

    def with_overrides(self, **kwargs: object) -> TransformConfig:
        """Create a new TransformConfig with specified overrides.

        Args:
            **kwargs: Fields to override in the new config.

        Returns:
            New TransformConfig with the overrides applied.

        Example:
            >>> base_config = TransformConfig(image_size=224)
            >>> large_config = base_config.with_overrides(image_size=384)
        """
        from dataclasses import asdict

        current = asdict(self)
        current.update(kwargs)
        return TransformConfig(**current)

    def to_dict(self) -> dict:
        """Convert config to a dictionary.

        Returns:
            Dictionary representation of the config.
        """
        from dataclasses import asdict

        return asdict(self)


# Preset definitions for each backend
TORCHVISION_PRESETS = {
    "default": "RandomResizedCrop, HorizontalFlip, ColorJitter",
    "autoaugment": "AutoAugment (ImageNet policy)",
    "randaugment": "RandAugment with configurable ops/magnitude",
    "trivialaugment": "TrivialAugmentWide",
    "light": "RandomResizedCrop, HorizontalFlip only",
}

ALBUMENTATIONS_PRESETS = {
    "default": "RandomResizedCrop, HorizontalFlip, ColorJitter",
    "strong": "Affine, blur/noise, ColorJitter, CoarseDropout",
    "light": "RandomResizedCrop, HorizontalFlip only",
}


def list_transform_presets(
    backend: str = "torchvision",
    verbose: bool = False,
) -> list[str] | list[tuple[str, str]]:
    """List available transform presets for a given backend.

    Args:
        backend: Transform backend ("torchvision" or "albumentations").
        verbose: If True, return tuples of (name, description).
            If False, return just preset names.

    Returns:
        List of preset names, or list of (name, description) tuples if verbose.

    Example:
        >>> list_transform_presets()
        ['default', 'autoaugment', 'randaugment', 'trivialaugment', 'light']
        >>> list_transform_presets(backend="albumentations")
        ['default', 'strong', 'light']
        >>> list_transform_presets(verbose=True)
        [('default', 'RandomResizedCrop, HorizontalFlip, ColorJitter'), ...]
    """
    if backend == "torchvision":
        presets = TORCHVISION_PRESETS
    elif backend == "albumentations":
        presets = ALBUMENTATIONS_PRESETS
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Use 'torchvision' or 'albumentations'."
        )

    if verbose:
        return list(presets.items())
    return list(presets.keys())
