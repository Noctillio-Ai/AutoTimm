"""Default image transform presets for training and evaluation."""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Standard training transforms: random crop, flip, color jitter, normalize."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def autoaugment_train_transforms(
    image_size: int = 224,
    policy: transforms.AutoAugmentPolicy = transforms.AutoAugmentPolicy.IMAGENET,
) -> transforms.Compose:
    """Training transforms with AutoAugment.

    Parameters:
        image_size: Target image size (square).
        policy: AutoAugment policy (IMAGENET, CIFAR10, or SVHN).
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy=policy),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def randaugment_train_transforms(
    image_size: int = 224,
    num_ops: int = 2,
    magnitude: int = 9,
) -> transforms.Compose:
    """Training transforms with RandAugment.

    Parameters:
        image_size: Target image size (square).
        num_ops: Number of augmentation operations per image.
        magnitude: Magnitude of augmentation operations (0-30).
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def trivialaugment_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training transforms with TrivialAugmentWide."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def light_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Light training transforms: random crop and flip only."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def default_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Standard evaluation transforms: resize, center crop, normalize."""
    resize_size = int(image_size * 256 / 224)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


TORCHVISION_PRESETS: dict[str, callable] = {
    "default": default_train_transforms,
    "autoaugment": autoaugment_train_transforms,
    "randaugment": randaugment_train_transforms,
    "trivialaugment": trivialaugment_train_transforms,
    "light": light_train_transforms,
}


# ---------------------------------------------------------------------------
# Albumentations transforms (lazy-imported, requires `pip install autotimm[albumentations]`)
# ---------------------------------------------------------------------------


def _require_albumentations():
    try:
        import albumentations

        return albumentations
    except ImportError:
        raise ImportError(
            "Albumentations is required for this transform backend. "
            "Install with: pip install autotimm[albumentations]"
        ) from None


def albu_default_train_transforms(image_size: int = 224):
    """Default albumentations training transforms with OpenCV backend."""
    A = _require_albumentations()
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.RandomResizedCrop(size=(image_size, image_size)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.8),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def albu_strong_train_transforms(image_size: int = 224):
    """Strong albumentations training transforms (heavy augmentation)."""
    A = _require_albumentations()
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.RandomResizedCrop(size=(image_size, image_size)),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=(-0.1, 0.1),
                scale=(0.8, 1.2),
                rotate=(-15, 15),
                p=0.5,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.GaussNoise(std_range=(0.02, 0.05)),
                ],
                p=0.3,
            ),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                p=0.3,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def albu_light_train_transforms(image_size: int = 224):
    """Light albumentations training transforms: random crop and flip only."""
    A = _require_albumentations()
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.RandomResizedCrop(size=(image_size, image_size)),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def albu_default_eval_transforms(image_size: int = 224):
    """Default albumentations evaluation transforms with OpenCV backend."""
    A = _require_albumentations()
    from albumentations.pytorch import ToTensorV2

    resize_size = int(image_size * 256 / 224)
    return A.Compose(
        [
            A.Resize(height=resize_size, width=resize_size),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


ALBUMENTATIONS_PRESETS: dict[str, callable] = {
    "default": albu_default_train_transforms,
    "strong": albu_strong_train_transforms,
    "light": albu_light_train_transforms,
}


def get_train_transforms(
    preset: str = "default",
    backend: str = "torchvision",
    **kwargs,
) -> transforms.Compose:
    """Get training transforms by preset name and backend.

    Parameters:
        preset: Preset name. For ``torchvision``: ``"default"``,
            ``"autoaugment"``, ``"randaugment"``, ``"trivialaugment"``,
            ``"light"``. For ``albumentations``: ``"default"``,
            ``"strong"``, ``"light"``.
        backend: ``"torchvision"`` or ``"albumentations"``.
        **kwargs: Forwarded to the preset function (e.g. ``image_size``).

    Raises:
        ValueError: If the preset or backend is unrecognized.
    """
    if backend == "torchvision":
        registry = TORCHVISION_PRESETS
    elif backend == "albumentations":
        _require_albumentations()
        registry = ALBUMENTATIONS_PRESETS
    else:
        raise ValueError(
            f"Unknown transform backend '{backend}'. "
            f"Choose from: torchvision, albumentations."
        )

    if preset not in registry:
        raise ValueError(
            f"Unknown augmentation preset '{preset}' for backend '{backend}'. "
            f"Choose from: {', '.join(registry)}."
        )
    return registry[preset](**kwargs)
