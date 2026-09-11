"""Demonstrate semantic segmentation model inference and prediction.

This example demonstrates:
- Loading a trained segmentation model from checkpoint
- Running inference on single images using preprocess()
- Running batch predictions
- Visualizing segmentation masks with color overlays
- Exporting predictions to PNG/JSON

Usage:
    python examples/segmentation_inference.py
"""

import json

import numpy as np
import torch
from PIL import Image

from autotimm import (
    MetricConfig,
    SemanticSegmentor,
    TransformConfig,
)

# Cityscapes color palette (19 classes + background)
CITYSCAPES_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

CITYSCAPES_COLORS = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
]

# VOC color palette (21 classes)
VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]

VOC_COLORS = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128),
]


def load_model(
    checkpoint_path: str | None = None,
    backbone: str = "resnet50",
    num_classes: int = 19,
    head_type: str = "deeplabv3plus",
    image_size: int = 512,
) -> SemanticSegmentor:
    """Load a SemanticSegmentor model.

    Args:
        checkpoint_path: Path to checkpoint file. If None, creates untrained model.
        backbone: Backbone architecture name.
        num_classes: Number of segmentation classes.
        head_type: Segmentation head type ("deeplabv3plus" or "fcn").
        image_size: Input image size for preprocessing.

    Returns:
        Loaded SemanticSegmentor model.
    """
    metrics = [
        MetricConfig(
            name="mIoU",
            backend="torchmetrics",
            metric_class="JaccardIndex",
            params={
                "task": "multiclass",
                "num_classes": num_classes,
                "average": "macro",
                "ignore_index": 255,
            },
            stages=["val", "test"],
        ),
    ]

    # TransformConfig for preprocessing
    transform_config = TransformConfig(
        image_size=image_size,
        use_timm_config=True,
    )

    if checkpoint_path:
        model = SemanticSegmentor.load_from_checkpoint(
            checkpoint_path,
            backbone=backbone,
            num_classes=num_classes,
            head_type=head_type,
            metrics=metrics,
            transform_config=transform_config,
        )
    else:
        model = SemanticSegmentor(
            backbone=backbone,
            num_classes=num_classes,
            head_type=head_type,
            metrics=metrics,
            transform_config=transform_config,
        )

    model.eval()
    return model


def predict_single_image(
    model: SemanticSegmentor,
    image_path: str,
) -> dict:
    """Predict segmentation mask for a single image.

    Args:
        model: Trained SemanticSegmentor model.
        image_path: Path to input image.

    Returns:
        Dictionary with 'mask' (H, W numpy array) and 'probabilities' (C, H, W).
    """
    # Load and preprocess image using model's preprocessing
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)

    # Use model's preprocess method (requires TransformConfig)
    input_tensor = model.preprocess(image)

    # Move to same device as model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Get segmentation predictions
    with torch.inference_mode():
        logits = model.predict(input_tensor)

    # Get probabilities and predicted classes
    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (C, H, W)
    mask = logits.argmax(dim=1)[0].cpu().numpy()  # (H, W)

    # Resize mask back to original image size
    mask_pil = Image.fromarray(mask.astype(np.uint8), mode="L")
    mask_pil = mask_pil.resize(original_size, Image.NEAREST)
    mask = np.array(mask_pil)

    return {
        "mask": mask,
        "probabilities": probabilities,
        "original_size": original_size,
    }


def predict_batch(
    model: SemanticSegmentor,
    image_paths: list[str],
    batch_size: int = 4,
) -> list[dict]:
    """Run batch prediction on multiple images.

    Args:
        model: Trained SemanticSegmentor model.
        image_paths: List of image paths.
        batch_size: Batch size for inference.

    Returns:
        List of prediction results per image.
    """
    device = next(model.parameters()).device
    all_results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
        original_sizes = [img.size for img in batch_images]

        # Preprocess batch
        input_tensor = model.preprocess(batch_images).to(device)

        # Get predictions
        with torch.inference_mode():
            logits = model.predict(input_tensor)

        # Process each image's prediction
        for j, (logit, orig_size) in enumerate(zip(logits, original_sizes)):
            probabilities = torch.softmax(logit, dim=0).cpu().numpy()
            mask = logit.argmax(dim=0).cpu().numpy()

            # Resize mask back to original size
            mask_pil = Image.fromarray(mask.astype(np.uint8), mode="L")
            mask_pil = mask_pil.resize(orig_size, Image.NEAREST)
            mask = np.array(mask_pil)

            all_results.append(
                {
                    "mask": mask,
                    "probabilities": probabilities,
                    "original_size": orig_size,
                }
            )

    return all_results


def mask_to_color(
    mask: np.ndarray,
    color_palette: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Convert class mask to RGB color image.

    Args:
        mask: (H, W) array of class indices.
        color_palette: List of RGB tuples for each class.

    Returns:
        (H, W, 3) RGB image.
    """
    if color_palette is None:
        # Generate random colors if not provided
        num_classes = int(mask.max()) + 1
        np.random.seed(42)
        color_palette = [
            tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_classes)
        ]

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, color in enumerate(color_palette):
        if class_idx < mask.max() + 1:
            color_mask[mask == class_idx] = color

    return color_mask


def visualize_segmentation(
    image_path: str,
    mask: np.ndarray,
    output_path: str | None = None,
    color_palette: list[tuple[int, int, int]] | None = None,
    alpha: float = 0.5,
    show: bool = False,
) -> Image.Image:
    """Overlay segmentation mask on original image.

    Args:
        image_path: Path to input image.
        mask: (H, W) array of class indices.
        output_path: Optional path to save annotated image.
        color_palette: List of RGB tuples for each class.
        alpha: Transparency of overlay (0=transparent, 1=opaque).
        show: Whether to display the image.

    Returns:
        Annotated PIL Image.
    """
    # Load original image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Convert mask to color
    color_mask = mask_to_color(mask, color_palette)

    # Blend with original image
    blended = (alpha * color_mask + (1 - alpha) * image_np).astype(np.uint8)
    blended_image = Image.fromarray(blended)

    if output_path:
        blended_image.save(output_path)
        print(f"Saved annotated image to: {output_path}")

    if show:
        blended_image.show()

    return blended_image


def export_mask_to_png(
    mask: np.ndarray,
    output_path: str,
    color_palette: list[tuple[int, int, int]] | None = None,
) -> None:
    """Save segmentation mask as PNG.

    Args:
        mask: (H, W) array of class indices.
        output_path: Path to output PNG file.
        color_palette: Optional color palette for visualization.
    """
    if color_palette:
        # Save colored mask
        color_mask = mask_to_color(mask, color_palette)
        Image.fromarray(color_mask).save(output_path)
    else:
        # Save grayscale mask (class indices)
        Image.fromarray(mask.astype(np.uint8), mode="L").save(output_path)

    print(f"Saved mask to: {output_path}")


def export_to_json(
    mask: np.ndarray | list[np.ndarray],
    output_path: str,
    image_paths: list[str] | None = None,
    class_names: list[str] | None = None,
) -> None:
    """Export segmentation statistics to JSON file.

    Args:
        mask: Segmentation mask(s).
        output_path: Path to output JSON file.
        image_paths: Optional list of image paths for batch results.
        class_names: Optional list of class names.
    """

    def compute_stats(m):
        """Compute per-class pixel counts."""
        unique, counts = np.unique(m, return_counts=True)
        total_pixels = m.size
        stats = {}
        for cls_idx, count in zip(unique, counts):
            class_name = (
                class_names[cls_idx]
                if class_names and cls_idx < len(class_names)
                else f"class_{cls_idx}"
            )
            stats[class_name] = {
                "class_idx": int(cls_idx),
                "pixel_count": int(count),
                "percentage": float(count / total_pixels * 100),
            }
        return stats

    if image_paths:
        # Batch format
        output = {
            "images": [
                {
                    "path": str(path),
                    "statistics": compute_stats(m),
                }
                for path, m in zip(image_paths, mask)
            ]
        }
    else:
        # Single image format
        output = {"statistics": compute_stats(mask)}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved statistics to: {output_path}")


def create_legend(
    class_names: list[str],
    color_palette: list[tuple[int, int, int]],
    output_path: str | None = None,
) -> Image.Image:
    """Create a legend image for the segmentation classes.

    Args:
        class_names: List of class names.
        color_palette: List of RGB tuples for each class.
        output_path: Optional path to save legend image.

    Returns:
        PIL Image of the legend.
    """
    from PIL import ImageDraw, ImageFont

    # Image dimensions
    box_size = 30
    padding = 10
    text_offset = box_size + padding
    line_height = box_size + padding

    width = 400
    height = len(class_names) * line_height + padding

    # Create image
    legend = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(legend)

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except OSError:
            font = ImageFont.load_default()

    # Draw legend items
    for i, (name, color) in enumerate(zip(class_names, color_palette)):
        y = padding + i * line_height

        # Draw color box
        draw.rectangle(
            [padding, y, padding + box_size, y + box_size],
            fill=color,
            outline=(0, 0, 0),
        )

        # Draw text
        draw.text((padding + text_offset, y + 5), name, fill=(0, 0, 0), font=font)

    if output_path:
        legend.save(output_path)
        print(f"Saved legend to: {output_path}")

    return legend


def demo_with_trained_model():
    """Demo using a trained model checkpoint.

    This is the recommended usage pattern for real inference.
    Uncomment and modify paths as needed.
    """
    # Load trained model
    model = load_model(
        checkpoint_path="path/to/your/checkpoint.ckpt",
        backbone="resnet50",
        num_classes=19,
        head_type="deeplabv3plus",
        image_size=512,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Single image inference
    result = predict_single_image(
        model=model,
        image_path="path/to/image.jpg",
    )

    print(f"Segmentation mask shape: {result['mask'].shape}")

    # Visualize
    visualize_segmentation(
        image_path="path/to/image.jpg",
        mask=result["mask"],
        output_path="output_segmentation.jpg",
        color_palette=CITYSCAPES_COLORS,
        alpha=0.5,
    )

    # Export
    export_mask_to_png(result["mask"], "mask.png", color_palette=CITYSCAPES_COLORS)
    export_to_json(result["mask"], "statistics.json", class_names=CITYSCAPES_CLASSES)

    # Create legend
    create_legend(CITYSCAPES_CLASSES, CITYSCAPES_COLORS, "legend.png")


def main():
    # ========================================================================
    # Configuration
    # ========================================================================
    print("=" * 60)
    print("Semantic Segmentation Inference Demo")
    print("=" * 60)

    # Model configuration
    backbone = "resnet50"
    num_classes = 19  # Cityscapes
    head_type = "deeplabv3plus"
    image_size = 512

    # ========================================================================
    # Part 1: Load model
    # ========================================================================
    print("\n1. Loading model...")

    # Option A: Load from checkpoint (recommended for real use)
    # model = load_model(
    #     checkpoint_path="path/to/checkpoint.ckpt",
    #     backbone=backbone,
    #     num_classes=num_classes,
    #     head_type=head_type,
    #     image_size=image_size,
    # )

    # Option B: Create untrained model (for demo purposes)
    model = load_model(
        checkpoint_path=None,
        backbone=backbone,
        num_classes=num_classes,
        head_type=head_type,
        image_size=image_size,
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"   Model loaded on: {device}")

    # ========================================================================
    # Part 2: Demonstrate preprocessing
    # ========================================================================
    print("\n2. Demonstrating preprocessing...")

    demo_image = Image.fromarray(
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    )
    demo_image_path = "/tmp/demo_segmentation_image.jpg"
    demo_image.save(demo_image_path)

    # Show preprocessing capabilities
    image = Image.open(demo_image_path).convert("RGB")
    input_tensor = model.preprocess(image)
    print(f"   Input image size: {image.size}")
    print(f"   Preprocessed tensor shape: {input_tensor.shape}")

    # ========================================================================
    # Part 3: Get model's data config
    # ========================================================================
    print("\n3. Model data configuration:")

    data_config = model.get_data_config()
    print(f"   Mean: {data_config['mean']}")
    print(f"   Std: {data_config['std']}")
    print(f"   Input size: {data_config['input_size']}")

    # ========================================================================
    # Part 4: Run inference
    # ========================================================================
    print("\n4. Running inference...")

    try:
        result = predict_single_image(
            model=model,
            image_path=demo_image_path,
        )
        print(f"   Mask shape: {result['mask'].shape}")
        print(f"   Unique classes: {np.unique(result['mask']).tolist()}")

        # Visualize
        print("\n5. Visualizing segmentation...")
        output_image_path = "/tmp/demo_segmentation_output.jpg"
        visualize_segmentation(
            image_path=demo_image_path,
            mask=result["mask"],
            output_path=output_image_path,
            color_palette=CITYSCAPES_COLORS[:num_classes],
            alpha=0.6,
        )

        # Export mask
        print("\n6. Exporting results...")
        export_mask_to_png(
            result["mask"],
            "/tmp/mask.png",
            color_palette=CITYSCAPES_COLORS[:num_classes],
        )
        export_to_json(
            result["mask"],
            "/tmp/statistics.json",
            class_names=CITYSCAPES_CLASSES[:num_classes],
        )

        # Create legend
        print("\n7. Creating legend...")
        create_legend(
            CITYSCAPES_CLASSES[:num_classes],
            CITYSCAPES_COLORS[:num_classes],
            "/tmp/legend.png",
        )

    except Exception as e:
        print(f"   Note: Inference skipped (untrained model or error: {e})")
        print("   For real inference, load a trained checkpoint.")

    # ========================================================================
    # Part 5: Show example usage code
    # ========================================================================
    print("\n" + "=" * 60)
    print("Example Usage with Trained Model:")
    print("=" * 60)
    print("""
    from examples.segmentation_inference import (
        load_model,
        predict_single_image,
        predict_batch,
        visualize_segmentation,
        export_mask_to_png,
        export_to_json,
        create_legend,
        CITYSCAPES_CLASSES,
        CITYSCAPES_COLORS,
    )

    # Load trained model
    model = load_model(
        checkpoint_path="best-segmentor.ckpt",
        backbone="resnet50",
        num_classes=19,
        image_size=512,
    )
    model = model.cuda()

    # Single image
    result = predict_single_image(model, "image.jpg")

    # Visualize with overlay
    visualize_segmentation(
        "image.jpg",
        result["mask"],
        "output.jpg",
        color_palette=CITYSCAPES_COLORS,
        alpha=0.5,
    )

    # Batch inference
    results = predict_batch(model, ["img1.jpg", "img2.jpg"])

    # Export
    export_mask_to_png(result["mask"], "mask.png", CITYSCAPES_COLORS)
    export_to_json(result["mask"], "stats.json", CITYSCAPES_CLASSES)

    # Create legend
    create_legend(CITYSCAPES_CLASSES, CITYSCAPES_COLORS, "legend.png")
    """)

    # ========================================================================
    # Cleanup
    # ========================================================================
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)

    import os

    for path in [
        demo_image_path,
        "/tmp/demo_segmentation_output.jpg",
        "/tmp/mask.png",
        "/tmp/statistics.json",
        "/tmp/legend.png",
    ]:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    main()
