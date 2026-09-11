"""Demonstrate object detection model inference and prediction.

This example demonstrates:
- Loading a trained detection model from checkpoint
- Running inference on single images using preprocess()
- Running batch predictions
- Visualizing detections with bounding boxes
- Exporting predictions to JSON/CSV

Usage:
    python examples/detection_inference.py
"""

import json

import torch
from PIL import Image, ImageDraw, ImageFont

from autotimm import (
    MetricConfig,
    ObjectDetector,
    TransformConfig,
)

# COCO class names (80 classes)
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def load_model(
    checkpoint_path: str | None = None,
    backbone: str = "resnet50",
    num_classes: int = 80,
    image_size: int = 640,
    score_thresh: float = 0.3,
) -> ObjectDetector:
    """Load an ObjectDetector model.

    Args:
        checkpoint_path: Path to checkpoint file. If None, creates untrained model.
        backbone: Backbone architecture name.
        num_classes: Number of detection classes.
        image_size: Input image size for preprocessing.
        score_thresh: Score threshold for detections.

    Returns:
        Loaded ObjectDetector model.
    """
    metrics = [
        MetricConfig(
            name="mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "bbox"},
            stages=["val", "test"],
        ),
    ]

    # TransformConfig for preprocessing
    transform_config = TransformConfig(
        image_size=image_size,
        use_timm_config=True,
    )

    if checkpoint_path:
        model = ObjectDetector.load_from_checkpoint(
            checkpoint_path,
            backbone=backbone,
            num_classes=num_classes,
            metrics=metrics,
            transform_config=transform_config,
            score_thresh=score_thresh,
        )
    else:
        model = ObjectDetector(
            backbone=backbone,
            num_classes=num_classes,
            metrics=metrics,
            transform_config=transform_config,
            score_thresh=score_thresh,
        )

    model.eval()
    return model


def predict_single_image(
    model: ObjectDetector,
    image_path: str,
    class_names: list[str] | None = None,
) -> list[dict]:
    """Predict objects in a single image.

    Args:
        model: Trained ObjectDetector model.
        image_path: Path to input image.
        class_names: Optional list of class names.

    Returns:
        List of detections, each with 'class', 'class_idx', 'confidence', 'bbox'.
    """
    # Load and preprocess image using model's preprocessing
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)

    # Use model's preprocess method (requires TransformConfig)
    input_tensor = model.preprocess(image)

    # Move to same device as model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Get detections
    with torch.inference_mode():
        detections = model.predict(input_tensor)

    # Process results for first (only) image
    det = detections[0]
    boxes = det["boxes"].cpu()
    scores = det["scores"].cpu()
    labels = det["labels"].cpu()

    # Scale boxes back to original image size
    img_h, img_w = input_tensor.shape[-2:]
    scale_x = original_size[0] / img_w
    scale_y = original_size[1] / img_h

    results = []
    for box, score, label in zip(boxes, scores, labels):
        # Scale box coordinates
        x1 = box[0].item() * scale_x
        y1 = box[1].item() * scale_y
        x2 = box[2].item() * scale_x
        y2 = box[3].item() * scale_y

        class_idx = label.item()
        class_name = class_names[class_idx] if class_names else str(class_idx)

        results.append(
            {
                "class": class_name,
                "class_idx": class_idx,
                "confidence": score.item(),
                "bbox": [x1, y1, x2, y2],
            }
        )

    return results


def predict_batch(
    model: ObjectDetector,
    image_paths: list[str],
    class_names: list[str] | None = None,
    batch_size: int = 4,
) -> list[list[dict]]:
    """Run batch prediction on multiple images.

    Args:
        model: Trained ObjectDetector model.
        image_paths: List of image paths.
        class_names: Optional list of class names.
        batch_size: Batch size for inference.

    Returns:
        List of detection results per image.
    """
    device = next(model.parameters()).device
    all_results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
        original_sizes = [img.size for img in batch_images]

        # Preprocess batch
        input_tensor = model.preprocess(batch_images).to(device)

        # Get detections
        with torch.inference_mode():
            detections = model.predict(input_tensor)

        # Process each image's detections
        img_h, img_w = input_tensor.shape[-2:]

        for j, (det, orig_size) in enumerate(zip(detections, original_sizes)):
            boxes = det["boxes"].cpu()
            scores = det["scores"].cpu()
            labels = det["labels"].cpu()

            scale_x = orig_size[0] / img_w
            scale_y = orig_size[1] / img_h

            image_results = []
            for box, score, label in zip(boxes, scores, labels):
                x1 = box[0].item() * scale_x
                y1 = box[1].item() * scale_y
                x2 = box[2].item() * scale_x
                y2 = box[3].item() * scale_y

                class_idx = label.item()
                class_name = class_names[class_idx] if class_names else str(class_idx)

                image_results.append(
                    {
                        "class": class_name,
                        "class_idx": class_idx,
                        "confidence": score.item(),
                        "bbox": [x1, y1, x2, y2],
                    }
                )

            all_results.append(image_results)

    return all_results


def visualize_detections(
    image_path: str,
    detections: list[dict],
    output_path: str | None = None,
    threshold: float = 0.0,
    show: bool = False,
) -> Image.Image:
    """Draw bounding boxes on image.

    Args:
        image_path: Path to input image.
        detections: List of detection dicts from predict_single_image.
        output_path: Optional path to save annotated image.
        threshold: Confidence threshold for visualization.
        show: Whether to display the image.

    Returns:
        Annotated PIL Image.
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except OSError:
            font = ImageFont.load_default()

    # Color palette for different classes
    colors = [
        "#FF0000",
        "#00FF00",
        "#0000FF",
        "#FFFF00",
        "#FF00FF",
        "#00FFFF",
        "#FFA500",
        "#800080",
        "#008000",
        "#000080",
        "#808000",
        "#800000",
    ]

    for det in detections:
        if det["confidence"] < threshold:
            continue

        bbox = det["bbox"]
        class_name = det["class"]
        confidence = det["confidence"]
        class_idx = det["class_idx"]

        # Get color for this class
        color = colors[class_idx % len(colors)]

        # Draw bounding box
        draw.rectangle(bbox, outline=color, width=3)

        # Draw label background
        label = f"{class_name}: {confidence:.2f}"
        text_bbox = draw.textbbox((bbox[0], bbox[1] - 20), label, font=font)
        draw.rectangle(text_bbox, fill=color)

        # Draw label text
        draw.text((bbox[0], bbox[1] - 20), label, fill="white", font=font)

    if output_path:
        image.save(output_path)
        print(f"Saved annotated image to: {output_path}")

    if show:
        image.show()

    return image


def export_to_json(
    detections: list[dict] | list[list[dict]],
    output_path: str,
    image_paths: list[str] | None = None,
) -> None:
    """Export detections to JSON file.

    Args:
        detections: Detection results (single image or batch).
        output_path: Path to output JSON file.
        image_paths: Optional list of image paths for batch results.
    """
    if image_paths:
        # Batch format
        output = {
            "images": [
                {
                    "path": str(path),
                    "detections": dets,
                }
                for path, dets in zip(image_paths, detections)
            ]
        }
    else:
        # Single image format
        output = {"detections": detections}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved detections to: {output_path}")


def export_to_csv(
    detections: list[dict] | list[list[dict]],
    output_path: str,
    image_paths: list[str] | None = None,
) -> None:
    """Export detections to CSV file.

    Args:
        detections: Detection results (single image or batch).
        output_path: Path to output CSV file.
        image_paths: Optional list of image paths for batch results.
    """
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_path", "class", "class_idx", "confidence", "x1", "y1", "x2", "y2"]
        )

        if image_paths:
            # Batch format
            for path, image_dets in zip(image_paths, detections):
                for det in image_dets:
                    bbox = det["bbox"]
                    writer.writerow(
                        [
                            str(path),
                            det["class"],
                            det["class_idx"],
                            f"{det['confidence']:.4f}",
                            f"{bbox[0]:.2f}",
                            f"{bbox[1]:.2f}",
                            f"{bbox[2]:.2f}",
                            f"{bbox[3]:.2f}",
                        ]
                    )
        else:
            # Single image format
            for det in detections:
                bbox = det["bbox"]
                writer.writerow(
                    [
                        "",
                        det["class"],
                        det["class_idx"],
                        f"{det['confidence']:.4f}",
                        f"{bbox[0]:.2f}",
                        f"{bbox[1]:.2f}",
                        f"{bbox[2]:.2f}",
                        f"{bbox[3]:.2f}",
                    ]
                )

    print(f"Saved detections to: {output_path}")


def demo_with_trained_model():
    """Demo using a trained model checkpoint.

    This is the recommended usage pattern for real inference.
    Uncomment and modify paths as needed.
    """
    # Load trained model
    model = load_model(
        checkpoint_path="path/to/your/checkpoint.ckpt",
        backbone="resnet50",
        num_classes=80,
        image_size=640,
        score_thresh=0.3,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Single image inference
    detections = predict_single_image(
        model=model,
        image_path="path/to/image.jpg",
        class_names=COCO_CLASSES,
    )

    print(f"Found {len(detections)} objects:")
    for det in detections:
        print(f"  {det['class']}: {det['confidence']:.2%}")

    # Visualize
    visualize_detections(
        image_path="path/to/image.jpg",
        detections=detections,
        output_path="output_with_boxes.jpg",
        threshold=0.3,
    )

    # Export
    export_to_json(detections, "detections.json")


def main():
    # ========================================================================
    # Configuration
    # ========================================================================
    print("=" * 60)
    print("Object Detection Inference Demo")
    print("=" * 60)

    # Model configuration
    backbone = "resnet50"
    num_classes = 80  # COCO
    image_size = 640
    score_thresh = 0.3

    # ========================================================================
    # Part 1: Load model
    # ========================================================================
    print("\n1. Loading model...")

    # Option A: Load from checkpoint (recommended for real use)
    # model = load_model(
    #     checkpoint_path="path/to/checkpoint.ckpt",
    #     backbone=backbone,
    #     num_classes=num_classes,
    #     image_size=image_size,
    #     score_thresh=score_thresh,
    # )

    # Option B: Create untrained model (for demo purposes)
    model = load_model(
        checkpoint_path=None,
        backbone=backbone,
        num_classes=num_classes,
        image_size=image_size,
        score_thresh=score_thresh,
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"   Model loaded on: {device}")

    # ========================================================================
    # Part 2: Demonstrate preprocessing
    # ========================================================================
    print("\n2. Demonstrating preprocessing...")

    import numpy as np

    demo_image = Image.fromarray(
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    )
    demo_image_path = "/tmp/demo_detection_image.jpg"
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
    # Part 4: Run inference (with trained model)
    # ========================================================================
    print("\n4. Running inference...")

    try:
        # This requires a trained model for meaningful results
        detections = predict_single_image(
            model=model,
            image_path=demo_image_path,
            class_names=COCO_CLASSES,
        )
        print(f"   Found {len(detections)} detections")
        for det in detections[:5]:
            print(f"   - {det['class']}: {det['confidence']:.2%}")

        # Visualize
        print("\n5. Visualizing detections...")
        output_image_path = "/tmp/demo_detection_output.jpg"
        visualize_detections(
            image_path=demo_image_path,
            detections=detections,
            output_path=output_image_path,
            threshold=0.1,
        )

        # Export
        print("\n6. Exporting results...")
        export_to_json(detections, "/tmp/detections.json")
        export_to_csv(detections, "/tmp/detections.csv")

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
    from examples.detection_inference import (
        load_model,
        predict_single_image,
        predict_batch,
        visualize_detections,
        export_to_json,
        COCO_CLASSES,
    )

    # Load trained model
    model = load_model(
        checkpoint_path="best-detector.ckpt",
        backbone="resnet50",
        num_classes=80,
        score_thresh=0.3,
    )
    model = model.cuda()

    # Single image
    detections = predict_single_image(
        model, "image.jpg", class_names=COCO_CLASSES
    )

    # Visualize
    visualize_detections(
        "image.jpg", detections, "output.jpg", threshold=0.3
    )

    # Batch inference
    results = predict_batch(
        model, ["img1.jpg", "img2.jpg"], class_names=COCO_CLASSES
    )

    # Export
    export_to_json(results, "detections.json", image_paths=["img1.jpg", "img2.jpg"])
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
        "/tmp/demo_detection_output.jpg",
        "/tmp/detections.json",
        "/tmp/detections.csv",
    ]:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    main()
