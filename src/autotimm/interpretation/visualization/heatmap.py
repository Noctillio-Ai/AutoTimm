"""Heatmap visualization utilities."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.figure import Figure
from PIL import Image


def apply_colormap(
    heatmap: np.ndarray,
    colormap: str = "viridis",
) -> np.ndarray:
    """
    Apply colormap to heatmap.

    Args:
        heatmap: 2D heatmap array in [0, 1]
        colormap: Matplotlib colormap name. Popular options:
            - 'viridis': Default, colorblind-friendly
            - 'jet': Traditional, high contrast
            - 'plasma', 'inferno', 'magma': Perceptually uniform
            - 'hot', 'cool': Temperature-based

    Returns:
        RGB image as uint8 array with shape (H, W, 3)

    Example:
        >>> heatmap = np.random.rand(224, 224)
        >>> colored = apply_colormap(heatmap, colormap='viridis')
        >>> colored.shape  # (224, 224, 3)
    """
    # Ensure heatmap is in [0, 1]
    heatmap = np.clip(heatmap, 0, 1)

    # Get colormap (use new API to avoid deprecation warning)
    try:
        # Matplotlib >= 3.7
        cmap = plt.colormaps.get_cmap(colormap)
    except AttributeError:
        # Matplotlib < 3.7
        cmap = cm.get_cmap(colormap)

    # Apply colormap
    colored_heatmap = cmap(heatmap)

    # Convert to RGB uint8 (remove alpha channel)
    colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)

    return colored_heatmap


def overlay_heatmap(
    image: np.ndarray | Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "viridis",
    resize_heatmap: bool = True,
) -> np.ndarray:
    """
    Overlay heatmap on original image.

    Args:
        image: Original image as numpy array or PIL Image
        heatmap: 2D heatmap array in [0, 1]
        alpha: Overlay transparency (0 = only image, 1 = only heatmap)
        colormap: Matplotlib colormap name
        resize_heatmap: Whether to resize heatmap to match image size

    Returns:
        Overlayed image as uint8 array with shape (H, W, 3)

    Example:
        >>> from PIL import Image
        >>> image = np.array(Image.open("photo.jpg"))
        >>> heatmap = np.random.rand(224, 224)
        >>> overlayed = overlay_heatmap(image, heatmap, alpha=0.4)
        >>> Image.fromarray(overlayed).save("explanation.jpg")
    """
    # Convert PIL Image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (
            (image * 255).astype(np.uint8)
            if image.max() <= 1
            else image.astype(np.uint8)
        )

    # Handle grayscale images
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Ensure RGB (not RGBA)
    if image.shape[2] == 4:
        image = image[:, :, :3]

    # Resize heatmap to match image size if requested
    if resize_heatmap and heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Apply colormap to heatmap
    colored_heatmap = apply_colormap(heatmap, colormap)

    # Overlay
    overlayed = cv2.addWeighted(image, 1 - alpha, colored_heatmap, alpha, 0)

    return overlayed


def create_comparison_figure(
    image: np.ndarray | Image.Image,
    heatmaps: list[np.ndarray],
    titles: list[str],
    layout: str = "grid",
    colormap: str = "viridis",
    alpha: float = 0.4,
    figsize: tuple[int, int] | None = None,
    dpi: int = 100,
) -> Figure:
    """
    Create multi-panel comparison figure with original image and heatmaps.

    Args:
        image: Original image
        heatmaps: List of heatmaps to display
        titles: Titles for each heatmap
        layout: Layout style:
            - 'grid': Automatic grid layout
            - 'horizontal': Single row
            - 'vertical': Single column
        colormap: Matplotlib colormap name
        alpha: Overlay transparency
        figsize: Figure size (width, height) in inches. If None, auto-calculated
        dpi: Dots per inch for output quality

    Returns:
        Matplotlib Figure object

    Example:
        >>> heatmaps = [gradcam_result, gradcam_plus_plus_result]
        >>> titles = ["GradCAM", "GradCAM++"]
        >>> fig = create_comparison_figure(
        ...     image, heatmaps, titles, layout='horizontal'
        ... )
        >>> fig.savefig("comparison.png", dpi=300, bbox_inches='tight')
    """
    # Convert PIL Image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Calculate number of panels (original + heatmaps)
    num_panels = len(heatmaps) + 1

    # Determine subplot layout
    if layout == "grid":
        rows = int(np.ceil(np.sqrt(num_panels)))
        cols = int(np.ceil(num_panels / rows))
    elif layout == "horizontal":
        rows, cols = 1, num_panels
    elif layout == "vertical":
        rows, cols = num_panels, 1
    else:
        raise ValueError(
            f"Unknown layout '{layout}'. Must be 'grid', 'horizontal', or 'vertical'"
        )

    # Calculate figure size if not provided
    if figsize is None:
        figsize = (cols * 4, rows * 4)

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)

    # Ensure axes is always a flat array
    if num_panels == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Plot heatmaps
    for idx, (heatmap, title) in enumerate(zip(heatmaps, titles), 1):
        overlayed = overlay_heatmap(image, heatmap, alpha=alpha, colormap=colormap)
        axes[idx].imshow(overlayed)
        axes[idx].set_title(title, fontsize=12, fontweight="bold")
        axes[idx].axis("off")

    # Hide unused axes
    for idx in range(num_panels, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    return fig


def save_heatmap(
    image: np.ndarray | Image.Image,
    heatmap: np.ndarray,
    save_path: str,
    colormap: str = "viridis",
    alpha: float = 0.4,
    dpi: int = 100,
    show_original: bool = True,
    show_heatmap_only: bool = False,
):
    """
    Save heatmap visualization to file.

    Args:
        image: Original image
        heatmap: 2D heatmap array
        save_path: Output file path
        colormap: Matplotlib colormap name
        alpha: Overlay transparency
        dpi: Output DPI
        show_original: Whether to show original image alongside overlay
        show_heatmap_only: If True, only save colormap-applied heatmap (no overlay)

    Example:
        >>> save_heatmap(
        ...     image, heatmap, "output.png",
        ...     show_original=True, dpi=300
        ... )
    """
    if show_heatmap_only:
        # Save only the colored heatmap
        colored = apply_colormap(heatmap, colormap)
        Image.fromarray(colored).save(save_path, dpi=(dpi, dpi))
        return

    if show_original:
        # Create side-by-side comparison
        fig = create_comparison_figure(
            image,
            [heatmap],
            ["Explanation"],
            layout="horizontal",
            colormap=colormap,
            alpha=alpha,
            dpi=dpi,
        )
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        # Save only overlay
        overlayed = overlay_heatmap(image, heatmap, alpha=alpha, colormap=colormap)
        Image.fromarray(overlayed).save(save_path, dpi=(dpi, dpi))
