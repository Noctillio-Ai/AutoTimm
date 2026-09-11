"""Detection-specific transforms with bounding box support."""

from __future__ import annotations

from autotimm.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def _require_albumentations():
    try:
        import albumentations  # noqa: F401

        return albumentations
    except ImportError:
        raise ImportError(
            "Albumentations is required for detection transforms. "
            "Install with: pip install autotimm[albumentations]"
        ) from None


def detection_train_transforms(
    image_size: int = 640,
    min_area: float = 0.0,
    min_visibility: float = 0.0,
):
    """Training transforms for object detection with bbox augmentation.

    Uses albumentations with bbox_params to ensure bounding boxes are
    transformed consistently with images.

    Parameters:
        image_size: Target image size (square).
        min_area: Minimum area ratio for bounding boxes to be kept.
        min_visibility: Minimum visibility ratio for bboxes after transform.

    Returns:
        Albumentations Compose with bbox support.
    """
    A = _require_albumentations()
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=(114, 114, 114),
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",  # [x_min, y_min, width, height]
            label_fields=["labels"],
            min_area=min_area,
            min_visibility=min_visibility,
        ),
    )


def detection_strong_train_transforms(
    image_size: int = 640,
    min_area: float = 0.0,
    min_visibility: float = 0.0,
):
    """Strong training transforms for object detection.

    Includes additional augmentations like affine, blur, and mosaic-style
    random crop for improved generalization.

    Parameters:
        image_size: Target image size (square).
        min_area: Minimum area ratio for bounding boxes to be kept.
        min_visibility: Minimum visibility ratio for bboxes after transform.

    Returns:
        Albumentations Compose with bbox support.
    """
    A = _require_albumentations()
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.5, 1.0),
                ratio=(0.8, 1.2),
                p=0.5,
            ),
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=(114, 114, 114),
            ),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                shear=(-5, 5),
                p=0.3,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=(3, 5)),
                ],
                p=0.2,
            ),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.8,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
            min_area=min_area,
            min_visibility=min_visibility,
        ),
    )


def detection_eval_transforms(image_size: int = 640):
    """Evaluation transforms for object detection.

    Minimal transforms: resize to fit within image_size, pad, normalize.

    Parameters:
        image_size: Target image size (square).

    Returns:
        Albumentations Compose with bbox support.
    """
    A = _require_albumentations()
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=(114, 114, 114),
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
        ),
    )


DETECTION_PRESETS: dict[str, callable] = {
    "default": detection_train_transforms,
    "strong": detection_strong_train_transforms,
}


def get_detection_transforms(
    preset: str = "default",
    image_size: int = 640,
    is_train: bool = True,
    **kwargs,
):
    """Get detection transforms by preset name.

    Parameters:
        preset: Preset name (``"default"``, ``"strong"``).
        image_size: Target image size.
        is_train: Whether to get training or evaluation transforms.
        **kwargs: Additional arguments passed to the transform function.

    Returns:
        Albumentations Compose with bbox support.

    Raises:
        ValueError: If the preset is unrecognized.
    """
    _require_albumentations()

    if not is_train:
        return detection_eval_transforms(image_size=image_size)

    if preset not in DETECTION_PRESETS:
        raise ValueError(
            f"Unknown detection preset '{preset}'. "
            f"Choose from: {', '.join(DETECTION_PRESETS)}."
        )

    return DETECTION_PRESETS[preset](image_size=image_size, **kwargs)
