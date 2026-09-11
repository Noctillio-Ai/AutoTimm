"""Preset manager for recommending the best transform backend and preset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from autotimm.core.logging import logger
from autotimm.data.transform_config import (
    ALBUMENTATIONS_PRESETS,
    TORCHVISION_PRESETS,
    TransformConfig,
)


@dataclass
class BackendRecommendation:
    """Recommendation for transform backend and preset.

    Attributes:
        backend: Recommended backend ("torchvision" or "albumentations").
        preset: Recommended preset for the backend.
        reasoning: Explanation for the recommendation.
        available_presets: List of all available presets for the backend.
        alternative: Alternative backend option with reasoning.
    """

    backend: Literal["torchvision", "albumentations"]
    preset: str
    reasoning: str
    available_presets: list[str]
    alternative: str | None = None

    def to_config(self, **kwargs) -> TransformConfig:
        """Convert recommendation to TransformConfig.

        Args:
            **kwargs: Additional arguments to pass to TransformConfig.

        Returns:
            TransformConfig with recommended backend and preset.

        Example:
            >>> recommendation = recommend_backend(task="classification")
            >>> config = recommendation.to_config(image_size=384)
        """
        return TransformConfig(backend=self.backend, preset=self.preset, **kwargs)

    def __str__(self) -> str:
        """Pretty print the recommendation."""
        lines = [
            f"Recommended Backend: {self.backend}",
            f"Recommended Preset: {self.preset}",
            f"\nReasoning: {self.reasoning}",
            f"\nAvailable presets for {self.backend}:",
        ]
        for preset in self.available_presets:
            lines.append(f"  - {preset}")

        if self.alternative:
            lines.append(f"\nAlternative: {self.alternative}")

        return "\n".join(lines)


def recommend_backend(
    task: (
        Literal["classification", "detection", "segmentation", "instance_segmentation"]
        | None
    ) = None,
    needs_advanced_augmentation: bool = False,
    needs_spatial_transforms: bool = False,
    prioritize_speed: bool = False,
    has_bbox_or_masks: bool = False,
) -> BackendRecommendation:
    """Recommend the best transform backend based on requirements.

    This function analyzes your use case and recommends either torchvision or
    albumentations as the transform backend, along with an appropriate preset.

    Args:
        task: Type of vision task. Options:
            - ``"classification"``: Image classification
            - ``"detection"``: Object detection
            - ``"segmentation"``: Semantic segmentation
            - ``"instance_segmentation"``: Instance segmentation
        needs_advanced_augmentation: Whether you need advanced augmentations
            like geometric transforms, blur, noise, etc.
        needs_spatial_transforms: Whether you need spatial transforms like
            rotation, affine, elastic transforms.
        prioritize_speed: Whether to prioritize transform speed over flexibility.
        has_bbox_or_masks: Whether your data includes bounding boxes or masks
            that need to be transformed together with images.

    Returns:
        BackendRecommendation with suggested backend, preset, and reasoning.

    Examples:
        >>> # Simple classification
        >>> rec = recommend_backend(task="classification")
        >>> print(rec)
        Recommended Backend: torchvision
        Recommended Preset: default
        ...

        >>> # Object detection with advanced augmentation
        >>> rec = recommend_backend(
        ...     task="detection",
        ...     needs_advanced_augmentation=True
        ... )
        >>> config = rec.to_config(image_size=640)

        >>> # Segmentation with spatial transforms
        >>> rec = recommend_backend(
        ...     task="segmentation",
        ...     needs_spatial_transforms=True
        ... )

        >>> # Quick prototyping (speed priority)
        >>> rec = recommend_backend(prioritize_speed=True)
    """
    # Default recommendation
    backend: Literal["torchvision", "albumentations"] = "torchvision"
    preset = "default"
    reasoning_parts = []

    # Task-based recommendations
    if task == "classification":
        if needs_advanced_augmentation:
            backend = "albumentations"
            preset = "strong"
            reasoning_parts.append(
                "Albumentations provides stronger augmentation options for classification."
            )
        else:
            backend = "torchvision"
            preset = "randaugment"
            reasoning_parts.append(
                "Torchvision's RandAugment is efficient and well-suited for classification."
            )

    elif task in ("detection", "instance_segmentation"):
        backend = "albumentations"
        preset = "default"
        reasoning_parts.append(
            f"{task.replace('_', ' ').title()} requires bbox/mask-aware transforms. "
            "Albumentations provides built-in support for synchronized transforms."
        )
        has_bbox_or_masks = True

    elif task == "segmentation":
        backend = "albumentations"
        preset = "default"
        reasoning_parts.append(
            "Segmentation requires mask-aware transforms. "
            "Albumentations handles image-mask synchronization automatically."
        )
        has_bbox_or_masks = True

    # Override based on specific requirements
    if needs_spatial_transforms and backend == "torchvision":
        backend = "albumentations"
        preset = "strong"
        reasoning_parts.append(
            "Albumentations excels at spatial transforms (rotation, affine, elastic)."
        )

    if has_bbox_or_masks and backend == "torchvision":
        backend = "albumentations"
        if preset not in ALBUMENTATIONS_PRESETS:
            preset = "default"
        reasoning_parts.append(
            "Albumentations is recommended for bbox/mask transforms to ensure proper synchronization."
        )

    if prioritize_speed and not has_bbox_or_masks:
        backend = "torchvision"
        preset = "light"
        reasoning_parts.append(
            "Torchvision's PIL-based transforms are fast for simple augmentations."
        )

    # Get available presets
    presets = (
        list(ALBUMENTATIONS_PRESETS.keys())
        if backend == "albumentations"
        else list(TORCHVISION_PRESETS.keys())
    )

    # Build reasoning
    if not reasoning_parts:
        reasoning_parts.append(
            "Torchvision is a good default choice for general-purpose transforms."
        )

    reasoning = " ".join(reasoning_parts)

    # Alternative recommendation
    alternative = None
    if backend == "torchvision" and (
        task in ("detection", "segmentation", "instance_segmentation")
    ):
        alternative = (
            "Consider albumentations for more advanced augmentations specific to "
            "detection/segmentation tasks."
        )
    elif (
        backend == "albumentations"
        and task == "classification"
        and not needs_advanced_augmentation
    ):
        alternative = "Torchvision with RandAugment/AutoAugment is simpler and often sufficient for classification."

    return BackendRecommendation(
        backend=backend,
        preset=preset,
        reasoning=reasoning,
        available_presets=presets,
        alternative=alternative,
    )


def compare_backends(verbose: bool = True) -> dict[str, dict]:
    """Compare torchvision and albumentations backends.

    Args:
        verbose: If True, print a formatted comparison table.

    Returns:
        Dictionary with comparison data for both backends.

    Example:
        >>> compare_backends()
        ╔═══════════════════════════════════════════════════════════╗
        ║           Transform Backend Comparison                    ║
        ╚═══════════════════════════════════════════════════════════╝
        ...
    """
    comparison = {
        "torchvision": {
            "backend": "PIL (CPU-based)",
            "speed": "Fast for simple transforms",
            "augmentations": "Standard (AutoAugment, RandAugment, TrivialAugment)",
            "bbox_mask_support": "Manual implementation required",
            "best_for": ["Classification", "Quick prototyping", "Simple pipelines"],
            "presets": list(TORCHVISION_PRESETS.keys()),
            "pros": [
                "Built into PyTorch ecosystem",
                "Fast for basic transforms",
                "Well-documented",
                "No extra dependencies (included in AutoTimm)",
            ],
            "cons": [
                "Limited spatial transforms",
                "No built-in bbox/mask handling",
                "Fewer augmentation options",
            ],
        },
        "albumentations": {
            "backend": "OpenCV (optimized)",
            "speed": "Fast for complex transforms",
            "augmentations": "Extensive (80+ transforms including spatial, blur, noise)",
            "bbox_mask_support": "Built-in synchronized transforms",
            "best_for": [
                "Object detection",
                "Segmentation",
                "Instance segmentation",
                "Advanced augmentation",
            ],
            "presets": list(ALBUMENTATIONS_PRESETS.keys()),
            "pros": [
                "Rich augmentation library",
                "Built-in bbox/mask support",
                "Spatial transforms (affine, elastic, etc.)",
                "Highly optimized (included in AutoTimm)",
            ],
            "cons": [
                "Slightly more complex API",
                "OpenCV-based (different from PIL)",
            ],
        },
    }

    if verbose:
        _print_comparison(comparison)

    return comparison


def _print_comparison(comparison: dict) -> None:
    """Pretty print the backend comparison using loguru."""
    from autotimm.core.logging import log_table

    # Build comparison table
    features = [
        "Backend",
        "Speed",
        "Augmentations",
        "BBox/Mask Support",
        "Best For",
        "Presets",
    ]
    feature_keys = [
        "backend",
        "speed",
        "augmentations",
        "bbox_mask_support",
        "best_for",
        "presets",
    ]

    rows = []
    for feature, key in zip(features, feature_keys):
        tv = comparison["torchvision"][key]
        albu = comparison["albumentations"][key]
        if isinstance(tv, list):
            tv = ", ".join(tv)
        if isinstance(albu, list):
            albu = ", ".join(albu)
        rows.append([feature, tv, albu])

    log_table(
        "Transform Backend Comparison",
        ["Feature", "Torchvision", "Albumentations"],
        rows,
    )

    # Print pros and cons
    lines = ["\nTorchvision - Pros:"]
    for pro in comparison["torchvision"]["pros"]:
        lines.append(f"  + {pro}")
    lines.append("\nTorchvision - Cons:")
    for con in comparison["torchvision"]["cons"]:
        lines.append(f"  - {con}")
    lines.append("\nAlbumentations - Pros:")
    for pro in comparison["albumentations"]["pros"]:
        lines.append(f"  + {pro}")
    lines.append("\nAlbumentations - Cons:")
    for con in comparison["albumentations"]["cons"]:
        lines.append(f"  - {con}")

    logger.info("\n".join(lines))


__all__ = [
    "BackendRecommendation",
    "compare_backends",
    "recommend_backend",
]
