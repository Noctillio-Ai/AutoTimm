"""Preview augmentation transforms on an image.

Applies a given augmentation preset 6 times and returns base64-encoded PNGs.

Usage::

    python -m autotimm.flow.augmentation_preview --image photo.jpg --preset light
    # or
    autotimm-flow augmentation-preview --image photo.jpg --preset light
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys


def preview(
    image_path: str, preset: str, image_size: int = 224, count: int = 6
) -> list[str]:
    """Apply augmentation transforms and return base64-encoded PNG strings."""
    from PIL import Image as PILImage

    from autotimm.data.transforms import get_train_transforms

    img = PILImage.open(image_path).convert("RGB")
    transform = get_train_transforms(preset, image_size=image_size)
    results: list[str] = []
    import numpy as np

    from autotimm.data.transforms import IMAGENET_MEAN, IMAGENET_STD

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    for _ in range(count):
        augmented = transform(img)
        if hasattr(augmented, "numpy"):
            # The preset pipelines end with Normalize, so undo it for a
            # human-viewable preview.
            if augmented.shape[0] == 3:
                arr = augmented.permute(1, 2, 0).numpy() * std + mean
            else:
                arr = augmented.numpy() * std + mean
            arr = (arr * 255).clip(0, 255).astype("uint8")
            pil = PILImage.fromarray(arr)
        else:
            pil = augmented
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        results.append(b64)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview augmentation transforms")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--preset", required=True, help="Augmentation preset name")
    parser.add_argument(
        "--image-size", type=int, default=224, help="Image size (default: 224)"
    )
    parser.add_argument(
        "--count", type=int, default=6, help="Number of augmented samples (default: 6)"
    )
    args = parser.parse_args()

    try:
        results = preview(args.image, args.preset, args.image_size, args.count)
        print(json.dumps(results))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
