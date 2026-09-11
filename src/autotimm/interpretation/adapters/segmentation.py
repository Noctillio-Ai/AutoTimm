"""Segmentation-specific interpretation adapters."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def explain_segmentation(
    model: torch.nn.Module,
    image: str | Image.Image | np.ndarray | torch.Tensor,
    target_class: int | None = None,
    method: str = "gradcam",
    target_layer: str | torch.nn.Module | None = None,
    show_mask_overlay: bool = True,
    show_uncertainty: bool = False,
    uncertainty_method: str = "entropy",
    colormap: str = "viridis",
    alpha: float = 0.4,
    save_path: str | None = None,
) -> dict:
    """
    Explain semantic segmentation predictions.

    This function generates explanations for segmentation models by creating
    heatmaps that show which image regions contributed to predictions for
    specific classes, optionally with uncertainty visualization.

    Args:
        model: Semantic segmentation model
        image: Input image
        target_class: Class to explain (None = dominant class)
        method: Interpretation method ('gradcam', 'gradcam++')
        target_layer: Layer to use for GradCAM (None = auto-detect)
        show_mask_overlay: Whether to show predicted segmentation mask
        show_uncertainty: Whether to visualize prediction uncertainty
        uncertainty_method: Uncertainty metric ('entropy', 'margin')
        colormap: Matplotlib colormap name
        alpha: Overlay transparency
        save_path: Optional path to save visualization

    Returns:
        Dictionary with:
            - 'heatmap': Attribution heatmap
            - 'prediction': Predicted segmentation mask
            - 'uncertainty': Uncertainty map (if show_uncertainty=True)
            - 'target_class': Class that was explained

    Example:
        >>> from autotimm import SemanticSegmentor
        >>> from autotimm.interpretation.adapters import explain_segmentation
        >>> model = SemanticSegmentor(backbone="resnet50", num_classes=19)
        >>> results = explain_segmentation(
        ...     model, "image.jpg", target_class=5, save_path="segmentation.png"
        ... )
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

    # Get prediction
    model.eval()
    with torch.inference_mode():
        input_tensor = explainer._preprocess_image(image_pil)
        seg_output = model(input_tensor)

    # Parse segmentation output
    if isinstance(seg_output, dict):
        logits = seg_output.get("logits", seg_output.get("output", seg_output))
    else:
        logits = seg_output

    # Get predicted mask
    probs = torch.softmax(logits, dim=1)
    prediction = logits.argmax(dim=1).squeeze().cpu().numpy()

    # Determine target class
    if target_class is None:
        # Use most frequent class (excluding background)
        unique, counts = np.unique(prediction, return_counts=True)
        # Sort by count
        sorted_idx = np.argsort(-counts)
        # Find first non-zero class
        for idx in sorted_idx:
            if unique[idx] != 0:  # Assuming 0 is background
                target_class = int(unique[idx])
                break
        if target_class is None:
            target_class = int(unique[sorted_idx[0]])

    # Generate heatmap for target class
    heatmap = explainer.explain(image_pil, target_class=target_class)

    # Prepare results
    results = {
        "heatmap": heatmap,
        "prediction": prediction,
        "target_class": target_class,
    }

    # Compute uncertainty if requested
    if show_uncertainty:
        uncertainty = _compute_uncertainty(probs, method=uncertainty_method)
        results["uncertainty"] = uncertainty

    # Create visualization if save path provided
    if save_path:
        viz = _visualize_segmentation_explanation(
            image_np,
            heatmap,
            prediction,
            target_class,
            uncertainty=results.get("uncertainty", None),
            show_mask=show_mask_overlay,
            colormap=colormap,
            alpha=alpha,
        )
        Image.fromarray(viz).save(save_path)
        results["visualization"] = viz

    return results


def _compute_uncertainty(
    probs: torch.Tensor,
    method: str = "entropy",
) -> np.ndarray:
    """
    Compute prediction uncertainty.

    Args:
        probs: Class probabilities (B, C, H, W)
        method: Uncertainty method ('entropy' or 'margin')

    Returns:
        Uncertainty map as numpy array
    """
    if method == "entropy":
        # Entropy: -sum(p * log(p))
        eps = 1e-10
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
        uncertainty = entropy.squeeze().cpu().numpy()

    elif method == "margin":
        # Margin: difference between top-2 probabilities
        top2_probs = torch.topk(probs, k=2, dim=1)[0]
        margin = top2_probs[:, 0] - top2_probs[:, 1]
        uncertainty = (1 - margin).squeeze().cpu().numpy()

    else:
        raise ValueError(f"Unknown uncertainty method: {method}")

    # Normalize to [0, 1]
    uncertainty = (uncertainty - uncertainty.min()) / (
        uncertainty.max() - uncertainty.min() + 1e-8
    )

    return uncertainty


def _visualize_segmentation_explanation(
    image: np.ndarray,
    heatmap: np.ndarray,
    prediction: np.ndarray,
    target_class: int,
    uncertainty: np.ndarray | None = None,
    show_mask: bool = True,
    colormap: str = "viridis",
    alpha: float = 0.4,
) -> np.ndarray:
    """Create visualization of segmentation explanation."""
    import cv2

    from autotimm.interpretation.visualization.heatmap import overlay_heatmap

    # Determine number of panels
    num_panels = 2  # Original + Heatmap
    if show_mask:
        num_panels += 1
    if uncertainty is not None:
        num_panels += 1

    fig, axes = plt.subplots(1, num_panels, figsize=(4 * num_panels, 4))
    if num_panels == 1:
        axes = [axes]

    panel_idx = 0

    # Original image
    axes[panel_idx].imshow(image)
    axes[panel_idx].set_title("Original", fontsize=12, fontweight="bold")
    axes[panel_idx].axis("off")
    panel_idx += 1

    # Heatmap overlay
    overlayed = overlay_heatmap(image, heatmap, alpha=alpha, colormap=colormap)
    axes[panel_idx].imshow(overlayed)
    axes[panel_idx].set_title(
        f"Explanation (Class {target_class})", fontsize=12, fontweight="bold"
    )
    axes[panel_idx].axis("off")
    panel_idx += 1

    # Predicted mask
    if show_mask:
        # Create colored mask
        # Resize prediction to image size if needed
        if prediction.shape != image.shape[:2]:
            prediction_resized = cv2.resize(
                prediction.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            prediction_resized = prediction

        # Highlight target class
        mask_vis = np.zeros_like(image)
        mask_vis[prediction_resized == target_class] = [
            255,
            0,
            0,
        ]  # Red for target class

        # Blend with original
        mask_overlay = cv2.addWeighted(image, 0.6, mask_vis, 0.4, 0)

        axes[panel_idx].imshow(mask_overlay)
        axes[panel_idx].set_title(
            f"Predicted Mask (Class {target_class})", fontsize=12, fontweight="bold"
        )
        axes[panel_idx].axis("off")
        panel_idx += 1

    # Uncertainty
    if uncertainty is not None:
        # Resize if needed
        if uncertainty.shape != image.shape[:2]:
            uncertainty_resized = cv2.resize(
                uncertainty, (image.shape[1], image.shape[0])
            )
        else:
            uncertainty_resized = uncertainty

        im = axes[panel_idx].imshow(uncertainty_resized, cmap="hot", vmin=0, vmax=1)
        axes[panel_idx].set_title("Uncertainty", fontsize=12, fontweight="bold")
        axes[panel_idx].axis("off")
        plt.colorbar(im, ax=axes[panel_idx], fraction=0.046)

    plt.tight_layout()

    # Convert to numpy array
    fig.canvas.draw()
    viz = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    viz = viz.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)

    return viz


__all__ = [
    "explain_segmentation",
]
