"""High-level API for model interpretation."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from autotimm.interpretation.gradcam import GradCAM, GradCAMPlusPlus
from autotimm.interpretation.integrated_gradients import IntegratedGradients
from autotimm.interpretation.visualization.heatmap import (
    create_comparison_figure,
    save_heatmap,
)


def explain_prediction(
    model: nn.Module,
    image: str | Path | Image.Image | np.ndarray,
    method: str = "gradcam",
    target_class: int | None = None,
    target_layer: str | nn.Module | None = None,
    colormap: str = "viridis",
    alpha: float = 0.4,
    save_path: str | Path | None = None,
    show_original: bool = True,
    dpi: int = 100,
    return_heatmap: bool = False,
) -> dict[str, Any]:
    """
    Explain a model's prediction on a single image.

    This is the recommended high-level API for most users. It automatically
    handles image loading, explanation generation, and visualization.

    Args:
        model: PyTorch model to interpret
        image: Input image (path, PIL Image, or numpy array)
        method: Interpretation method:
            - 'gradcam': GradCAM
            - 'gradcam++': GradCAM++
        target_class: Class to explain (None = predicted class)
        target_layer: Layer to use (None = auto-detect)
        colormap: Matplotlib colormap name
        alpha: Overlay transparency
        save_path: If provided, saves visualization to this path
        show_original: Whether to show original alongside explanation
        dpi: Output DPI for saved image
        return_heatmap: If True, returns raw heatmap array

    Returns:
        Dictionary with:
            - 'predicted_class': Predicted class index
            - 'target_class': Target class used for explanation
            - 'heatmap': Raw heatmap array (if return_heatmap=True)
            - 'method': Method used
            - 'target_layer': Layer used

    Example:
        >>> from autotimm import ImageClassifier
        >>> from autotimm.interpretation import explain_prediction
        >>> model = ImageClassifier(backbone="resnet18", num_classes=10)
        >>> result = explain_prediction(
        ...     model, "photo.jpg", method="gradcam", save_path="explanation.png"
        ... )
        >>> print(f"Predicted class: {result['predicted_class']}")
    """
    # Load image if path
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    # Select interpretation method
    if method.lower() == "gradcam":
        explainer = GradCAM(model, target_layer=target_layer)
    elif method.lower() == "gradcam++" or method.lower() == "gradcampp":
        explainer = GradCAMPlusPlus(model, target_layer=target_layer)
    elif method.lower() == "integrated_gradients" or method.lower() == "ig":
        explainer = IntegratedGradients(model, baseline="black", steps=50)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Supported: 'gradcam', 'gradcam++', 'integrated_gradients'"
        )

    # Generate explanation
    heatmap = explainer.explain(image, target_class=target_class)

    # Get predicted class if not specified
    model.eval()
    with torch.inference_mode():
        input_tensor = explainer._preprocess_image(image)
        output = model(input_tensor)

        if isinstance(output, dict):
            output = output.get("logits", output.get("output", output))

        predicted_class = output.argmax(dim=1).item()

    # Use predicted class if target not specified
    if target_class is None:
        target_class = predicted_class

    # Save visualization if requested
    if save_path is not None:
        save_heatmap(
            image,
            heatmap,
            str(save_path),
            colormap=colormap,
            alpha=alpha,
            dpi=dpi,
            show_original=show_original,
        )

    # Prepare result
    result = {
        "predicted_class": predicted_class,
        "target_class": target_class,
        "method": method,
        "target_layer": explainer.get_target_layer_name(),
    }

    if return_heatmap:
        result["heatmap"] = heatmap

    return result


def compare_methods(
    model: nn.Module,
    image: str | Path | Image.Image | np.ndarray,
    methods: list[str] | None = None,
    target_class: int | None = None,
    target_layer: str | nn.Module | None = None,
    layout: str = "horizontal",
    colormap: str = "viridis",
    alpha: float = 0.4,
    save_path: str | Path | None = None,
    dpi: int = 100,
):
    """
    Compare multiple interpretation methods side-by-side.

    Args:
        model: PyTorch model to interpret
        image: Input image (path, PIL Image, or numpy array)
        methods: List of methods to compare
        target_class: Class to explain (None = predicted class)
        target_layer: Layer to use (None = auto-detect)
        layout: Layout for comparison ('horizontal', 'vertical', 'grid')
        colormap: Matplotlib colormap name
        alpha: Overlay transparency
        save_path: If provided, saves comparison to this path
        dpi: Output DPI

    Returns:
        Dictionary with heatmaps for each method

    Example:
        >>> results = compare_methods(
        ...     model, "photo.jpg",
        ...     methods=["gradcam", "gradcam++"],
        ...     save_path="comparison.png"
        ... )
    """
    # Load image if path
    if methods is None:
        methods = ["gradcam", "gradcam++"]
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    # Generate heatmaps for each method
    heatmaps = []
    titles = []
    results = {}

    for method in methods:
        result = explain_prediction(
            model,
            image,
            method=method,
            target_class=target_class,
            target_layer=target_layer,
            return_heatmap=True,
        )

        heatmaps.append(result["heatmap"])
        titles.append(method.upper())
        results[method] = result

    # Create comparison figure
    fig = create_comparison_figure(
        image,
        heatmaps,
        titles,
        layout=layout,
        colormap=colormap,
        alpha=alpha,
        dpi=dpi,
    )

    # Save if requested
    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)

    return results


def visualize_batch(
    model: nn.Module,
    images: list[str | Path | Image.Image | np.ndarray],
    method: str = "gradcam",
    target_classes: list[int] | None = None,
    target_layer: str | nn.Module | None = None,
    output_dir: str | Path | None = None,
    colormap: str = "viridis",
    alpha: float = 0.4,
    dpi: int = 100,
    show_predictions: bool = True,
) -> list[dict[str, Any]]:
    """
    Generate explanations for a batch of images.

    Args:
        model: PyTorch model to interpret
        images: List of input images (paths, PIL Images, or numpy arrays)
        method: Interpretation method to use
        target_classes: List of target classes (None = use predicted)
        target_layer: Layer to use (None = auto-detect)
        output_dir: Directory to save visualizations (None = don't save)
        colormap: Matplotlib colormap name
        alpha: Overlay transparency
        dpi: Output DPI
        show_predictions: Whether to include prediction info in filenames

    Returns:
        List of result dictionaries, one per image

    Example:
        >>> from pathlib import Path
        >>> images = list(Path("test_images/").glob("*.jpg"))
        >>> results = visualize_batch(
        ...     model, images, method="gradcam", output_dir="explanations/"
        ... )
        >>> print(f"Processed {len(results)} images")
    """
    # Create output directory if needed
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare target classes
    if target_classes is None:
        target_classes = [None] * len(images)

    # Process each image
    results = []
    for idx, (image, target_class) in enumerate(zip(images, target_classes)):
        # Load image if path
        image_path = None
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            image_loaded = Image.open(image).convert("RGB")
        else:
            image_loaded = image

        # Generate explanation
        result = explain_prediction(
            model,
            image_loaded,
            method=method,
            target_class=target_class,
            target_layer=target_layer,
            return_heatmap=True,
        )

        # Save if output directory provided
        if output_dir is not None:
            # Generate filename
            if image_path is not None:
                base_name = image_path.stem
            else:
                base_name = f"image_{idx:04d}"

            if show_predictions:
                filename = f"{base_name}_class{result['predicted_class']}.png"
            else:
                filename = f"{base_name}_explanation.png"

            save_path = output_dir / filename

            # Save visualization
            save_heatmap(
                image_loaded,
                result["heatmap"],
                str(save_path),
                colormap=colormap,
                alpha=alpha,
                dpi=dpi,
                show_original=True,
            )

            result["save_path"] = str(save_path)

        results.append(result)

    return results


# Convenience function for common use case
def quick_explain(
    model: nn.Module,
    image: str | Path | Image.Image | np.ndarray,
    save_path: str | Path = "explanation.png",
) -> dict[str, Any]:
    """
    Quick explanation with sensible defaults.

    This is the simplest way to get started with model interpretation.
    Uses GradCAM with auto-detected layer and saves a nice visualization.

    Args:
        model: PyTorch model to interpret
        image: Input image
        save_path: Where to save the explanation

    Returns:
        Result dictionary with prediction info

    Example:
        >>> from autotimm import ImageClassifier
        >>> from autotimm.interpretation import quick_explain
        >>> model = ImageClassifier(backbone="resnet18", num_classes=10)
        >>> result = quick_explain(model, "photo.jpg")
        >>> print(f"Explained prediction: class {result['predicted_class']}")
    """
    return explain_prediction(
        model=model,
        image=image,
        method="gradcam",
        save_path=save_path,
        show_original=True,
        dpi=150,
    )


__all__ = [
    "compare_methods",
    "explain_prediction",
    "quick_explain",
    "visualize_batch",
]
