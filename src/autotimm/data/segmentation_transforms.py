"""Transform presets for segmentation tasks."""

from __future__ import annotations


def _require_albumentations():
    """Lazy import albumentations to avoid hard dependency."""
    try:
        import albumentations

        return albumentations
    except ImportError:
        raise ImportError(
            "Albumentations is required for segmentation transforms. "
            "Install with: pip install autotimm[segmentation]"
        ) from None


def segmentation_train_transforms(
    image_size: int = 512,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
):
    """Training transforms for semantic segmentation.

    Applies data augmentation including random scaling, cropping, flipping,
    and color jittering. Masks are automatically transformed with nearest
    interpolation to preserve class indices.

    Args:
        image_size: Target image size (square)
        mean: Normalization mean for RGB channels
        std: Normalization standard deviation for RGB channels

    Returns:
        Albumentations composition of transforms
    """
    A = _require_albumentations()
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.RandomScale(scale_limit=0.5, p=0.5),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
                fill_mask=255,
            ),
            A.RandomCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def segmentation_eval_transforms(
    image_size: int = 512,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
):
    """Evaluation transforms for semantic segmentation.

    No augmentation, only resizing and normalization.

    Args:
        image_size: Target image size (square)
        mean: Normalization mean for RGB channels
        std: Normalization standard deviation for RGB channels

    Returns:
        Albumentations composition of transforms
    """
    A = _require_albumentations()
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def instance_segmentation_transforms(
    image_size: int = 640,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    train: bool = True,
):
    """Transforms for instance segmentation with bounding boxes and masks.

    Args:
        image_size: Target image size (square)
        mean: Normalization mean for RGB channels
        std: Normalization standard deviation for RGB channels
        train: Whether to apply training augmentation

    Returns:
        Albumentations composition of transforms
    """
    A = _require_albumentations()
    from albumentations.pytorch import ToTensorV2

    if train:
        transforms = [
            A.RandomScale(scale_limit=0.3, p=0.5),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
                fill_mask=0,
            ),
            A.RandomCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05,
                p=0.5,
            ),
        ]
    else:
        transforms = [
            A.Resize(height=image_size, width=image_size),
        ]

    transforms.extend(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.3,
        ),
    )


def get_segmentation_preset(
    preset: str = "default",
    image_size: int = 512,
    train: bool = True,
):
    """Get segmentation transform preset by name.

    Args:
        preset: Preset name ('default', 'strong', 'light')
        image_size: Target image size
        train: Whether to get training or eval transforms

    Returns:
        Albumentations composition of transforms
    """
    if not train:
        return segmentation_eval_transforms(image_size=image_size)

    A = _require_albumentations()
    from albumentations.pytorch import ToTensorV2

    if preset == "strong":
        return A.Compose(
            [
                A.RandomScale(scale_limit=0.75, p=0.7),
                A.PadIfNeeded(
                    min_height=image_size,
                    min_width=image_size,
                    border_mode=0,
                    fill=0,
                    fill_mask=255,
                ),
                A.RandomCrop(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.5,
                    saturation=0.5,
                    hue=0.15,
                    p=0.7,
                ),
                A.GaussianBlur(blur_limit=(3, 9), p=0.5),
                A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    elif preset == "light":
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:  # default
        return segmentation_train_transforms(image_size=image_size)
