"""Detection-specific interpretation adapters."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from PIL import Image


def explain_detection(
    model: torch.nn.Module,
    image: str | Image.Image | np.ndarray | torch.Tensor,
    method: str = "gradcam",
    detection_threshold: float = 0.5,
    bbox: tuple[int, int, int, int] | None = None,
    target_layer: str | torch.nn.Module | None = None,
    colormap: str = "viridis",
    alpha: float = 0.4,
    save_path: str | None = None,
) -> dict:
    """
    Explain object detection predictions.

    This function generates explanations for object detection models by
    creating heatmaps that show which image regions contributed to each
    detected object.

    Args:
        model: Object detection model
        image: Input image
        method: Interpretation method ('gradcam', 'gradcam++')
        detection_threshold: Confidence threshold for detections
        bbox: Specific bbox to explain (x1, y1, x2, y2). If None, explains all detections
        target_layer: Layer to use for GradCAM (None = auto-detect)
        colormap: Matplotlib colormap name
        alpha: Overlay transparency
        save_path: Optional path to save visualization

    Returns:
        Dictionary with:
            - 'detections': List of detection dicts
            - 'heatmaps': List of heatmaps for each detection
            - 'visualization': Combined visualization if save_path provided

    Example:
        >>> from autotimm import ObjectDetector
        >>> from autotimm.interpretation.adapters import explain_detection
        >>> model = ObjectDetector(backbone="resnet50", num_classes=80)
        >>> results = explain_detection(
        ...     model, "image.jpg", detection_threshold=0.5, save_path="detections.png"
        ... )
        >>> print(f"Found {len(results['detections'])} detections")
    """
    from pathlib import Path

    # Load image if path
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
    elif isinstance(image, Image.Image):
        image_pil = image
        image_np = np.array(image)
    elif isinstance(image, np.ndarray):
        image_np = image
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image_np = (image * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
    else:
        # Tensor
        image_tensor = image
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.squeeze(0)
        image_np = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(
            np.uint8
        )
        image_pil = Image.fromarray(image_np)

    # Create explainer
    if method.lower() == "gradcam":
        from autotimm.interpretation import GradCAM

        explainer = GradCAM(model, target_layer=target_layer)
    elif method.lower() in ["gradcam++", "gradcampp"]:
        from autotimm.interpretation import GradCAMPlusPlus

        explainer = GradCAMPlusPlus(model, target_layer=target_layer)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Get detections
    model.eval()
    with torch.inference_mode():
        input_tensor = explainer._preprocess_image(image_pil)
        detections_output = model(input_tensor)

    # Parse detections (format depends on model type)
    detections = _parse_detections(detections_output, detection_threshold)

    # If specific bbox provided, filter to closest detection
    if bbox is not None:
        detections = [_find_closest_detection(detections, bbox)]

    # Generate heatmaps for each detection
    results = {
        "detections": [],
        "heatmaps": [],
    }

    for detection in detections:
        # Get class ID
        class_id = int(detection["class_id"])

        # Generate heatmap for this class
        heatmap = explainer.explain(image_pil, target_class=class_id)

        # Optionally mask heatmap to bbox region
        heatmap_masked = _mask_heatmap_to_bbox(
            heatmap, detection["bbox"], image_np.shape[:2]
        )

        results["detections"].append(
            {
                "bbox": detection["bbox"],
                "class_id": class_id,
                "confidence": detection["confidence"],
                "class_name": detection.get("class_name", f"class_{class_id}"),
            }
        )
        results["heatmaps"].append(heatmap_masked)

    # Create visualization if save path provided
    if save_path:
        viz = _visualize_detection_explanations(
            image_np,
            results["detections"],
            results["heatmaps"],
            colormap=colormap,
            alpha=alpha,
        )
        Image.fromarray(viz).save(save_path)
        results["visualization"] = viz

    return results


def _parse_detections(
    detections_output: torch.Tensor | dict | list,
    threshold: float,
) -> list[dict]:
    """Parse detection output to standard format."""
    parsed = []

    # Handle different output formats
    if isinstance(detections_output, dict):
        # Dict format: {'boxes': tensor, 'scores': tensor, 'labels': tensor}
        boxes = detections_output.get("boxes", detections_output.get("bbox", None))
        scores = detections_output.get(
            "scores", detections_output.get("confidence", None)
        )
        labels = detections_output.get(
            "labels", detections_output.get("class_id", None)
        )

        if boxes is not None and scores is not None and labels is not None:
            boxes = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
            scores = (
                scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
            )
            labels = (
                labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            )

            for box, score, label in zip(boxes, scores, labels):
                if score >= threshold:
                    parsed.append(
                        {
                            "bbox": box,
                            "confidence": float(score),
                            "class_id": int(label),
                        }
                    )

    elif isinstance(detections_output, torch.Tensor):
        # Tensor format: (N, 6) with [x1, y1, x2, y2, conf, class]
        dets = detections_output.cpu().numpy()
        if dets.ndim == 3:
            dets = dets[0]  # Remove batch dimension

        for det in dets:
            if len(det) >= 6 and det[4] >= threshold:
                parsed.append(
                    {
                        "bbox": det[:4],
                        "confidence": float(det[4]),
                        "class_id": int(det[5]),
                    }
                )

    return parsed


def _find_closest_detection(
    detections: list[dict],
    target_bbox: tuple[int, int, int, int],
) -> dict:
    """Find detection closest to target bbox."""
    min_dist = float("inf")
    closest = detections[0] if detections else None

    target_center = [
        (target_bbox[0] + target_bbox[2]) / 2,
        (target_bbox[1] + target_bbox[3]) / 2,
    ]

    for det in detections:
        bbox = det["bbox"]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        dist = np.sqrt(
            (center[0] - target_center[0]) ** 2 + (center[1] - target_center[1]) ** 2
        )

        if dist < min_dist:
            min_dist = dist
            closest = det

    return closest


def _mask_heatmap_to_bbox(
    heatmap: np.ndarray,
    bbox: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Mask heatmap to bbox region (optional soft masking)."""
    # Resize heatmap to image size
    import cv2

    heatmap_resized = cv2.resize(heatmap, (image_shape[1], image_shape[0]))

    # Apply Gaussian weighting centered on bbox
    y1, x1, y2, x2 = [int(coord) for coord in bbox]

    # Create distance-based mask
    y_coords, x_coords = np.ogrid[: image_shape[0], : image_shape[1]]
    center_y = (y1 + y2) / 2
    center_x = (x1 + x2) / 2

    # Gaussian falloff
    sigma = max(x2 - x1, y2 - y1) / 2
    distance = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    mask = np.exp(-(distance**2) / (2 * sigma**2))

    # Apply mask
    heatmap_masked = heatmap_resized * mask

    # Resize back to original heatmap size
    heatmap_masked = cv2.resize(heatmap_masked, (heatmap.shape[1], heatmap.shape[0]))

    return heatmap_masked


def _visualize_detection_explanations(
    image: np.ndarray,
    detections: list[dict],
    heatmaps: list[np.ndarray],
    colormap: str = "viridis",
    alpha: float = 0.4,
) -> np.ndarray:
    """Create visualization of detection explanations."""
    from autotimm.interpretation.visualization.heatmap import overlay_heatmap

    # Create figure
    fig, axes = plt.subplots(
        1, min(4, len(detections) + 1), figsize=(4 * min(4, len(detections) + 1), 4)
    )
    if len(detections) == 0 or not isinstance(axes, np.ndarray):
        axes = [axes]

    # Original image with all bboxes
    axes[0].imshow(image)
    axes[0].set_title("Detections", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    for det in detections:
        bbox = det["bbox"]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        axes[0].add_patch(rect)
        axes[0].text(
            bbox[0],
            bbox[1] - 5,
            f"{det.get('class_name', det['class_id'])}: {det['confidence']:.2f}",
            color="red",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.7},
        )

    # Individual detection explanations
    for idx, (det, heatmap) in enumerate(zip(detections[:3], heatmaps[:3])):
        overlayed = overlay_heatmap(image, heatmap, alpha=alpha, colormap=colormap)
        axes[idx + 1].imshow(overlayed)
        axes[idx + 1].set_title(
            f"Class {det.get('class_name', det['class_id'])} ({det['confidence']:.2f})",
            fontsize=10,
        )
        axes[idx + 1].axis("off")

        # Draw bbox
        bbox = det["bbox"]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
        )
        axes[idx + 1].add_patch(rect)

    plt.tight_layout()

    # Convert to numpy array
    fig.canvas.draw()
    viz = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    viz = viz.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)

    return viz


__all__ = [
    "explain_detection",
]
