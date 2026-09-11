"""Feature map visualization and analysis."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from PIL import Image
from torch import nn


class FeatureVisualizer:
    """
    Visualize and analyze feature maps from any layer.

    Useful for understanding what features the model learns at different
    depths and how they respond to specific inputs.

    Args:
        model: PyTorch model
        use_cuda: Whether to use CUDA if available

    Example:
        >>> from autotimm import ImageClassifier
        >>> from autotimm.interpretation import FeatureVisualizer
        >>>
        >>> model = ImageClassifier(backbone="resnet50", num_classes=10)
        >>> viz = FeatureVisualizer(model)
        >>>
        >>> # Visualize features
        >>> viz.plot_feature_maps(image, layer_name="backbone.layer3")
        >>>
        >>> # Get statistics
        >>> stats = viz.get_feature_statistics(image, layer_name="backbone.layer4")
        >>> print(f"Mean activation: {stats['mean']:.3f}")
    """

    def __init__(
        self,
        model: nn.Module,
        use_cuda: bool = True,
    ):
        self.model = model
        self.model.eval()
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.features = {}
        self._hooks = []

    def plot_feature_maps(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        layer_name: str,
        num_features: int = 16,
        sort_by: str = "activation",
        save_path: str | None = None,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot feature maps from a specific layer.

        Args:
            image: Input image
            layer_name: Name of layer to visualize
            num_features: Number of feature maps to show
            sort_by: How to select features ('activation', 'variance', 'random')
            save_path: Optional path to save figure
            figsize: Figure size (width, height)

        Returns:
            Matplotlib figure

        Example:
            >>> viz.plot_feature_maps(
            ...     image,
            ...     layer_name="backbone.layer3",
            ...     num_features=16,
            ...     sort_by="activation",
            ...     save_path="features.png"
            ... )
        """
        # Get features
        features = self.get_features(image, layer_name)

        # Select features to display
        selected_indices = self._select_features(features, num_features, sort_by)
        selected_features = features[0, selected_indices]  # (num_features, H, W)

        # Create figure
        grid_size = int(np.ceil(np.sqrt(num_features)))
        if figsize is None:
            figsize = (grid_size * 3, grid_size * 3)

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(grid_size, grid_size, hspace=0.3, wspace=0.3)

        for idx in range(min(num_features, len(selected_indices))):
            ax = fig.add_subplot(gs[idx])
            feature_map = selected_features[idx].cpu().numpy()

            # Normalize for visualization
            if feature_map.max() > feature_map.min():
                feature_map = (feature_map - feature_map.min()) / (
                    feature_map.max() - feature_map.min()
                )

            ax.imshow(feature_map, cmap="viridis")
            ax.set_title(f"Ch {selected_indices[idx]}", fontsize=10)
            ax.axis("off")

            # Add activation info
            mean_act = selected_features[idx].mean().item()
            ax.text(
                0.02,
                0.98,
                f"{mean_act:.2f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
                fontsize=8,
            )

        plt.suptitle(f"Feature Maps: {layer_name}", fontsize=14, fontweight="bold")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def get_features(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        """
        Extract features from a specific layer.

        Args:
            image: Input image
            layer_name: Name of layer

        Returns:
            Feature tensor (B, C, H, W)
        """
        # Clear previous features
        self.features = {}

        # Register hook
        layer = self._get_layer_by_name(layer_name)
        if layer is None:
            raise ValueError(f"Layer {layer_name} not found")

        def hook(module, input, output):
            self.features[layer_name] = output.detach()

        handle = layer.register_forward_hook(hook)

        # Forward pass
        with torch.inference_mode():
            input_tensor = self._preprocess_image(image)
            self.model(input_tensor)

        # Remove hook
        handle.remove()

        return self.features[layer_name]

    def get_feature_statistics(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        layer_name: str,
    ) -> dict[str, float]:
        """
        Compute feature statistics for a layer.

        Args:
            image: Input image
            layer_name: Name of layer

        Returns:
            Dictionary with statistics:
                - mean: Mean activation
                - std: Standard deviation
                - sparsity: Fraction of zero activations
                - max: Maximum activation
                - active_channels: Number of channels with mean > threshold

        Example:
            >>> stats = viz.get_feature_statistics(image, "backbone.layer4")
            >>> print(f"Sparsity: {stats['sparsity']:.2%}")
        """
        features = self.get_features(image, layer_name)

        stats = {
            "mean": features.mean().item(),
            "std": features.std().item(),
            "sparsity": (features == 0).float().mean().item(),
            "max": features.max().item(),
            "min": features.min().item(),
            "active_channels": (features.mean(dim=(2, 3)) > 0.01).sum().item(),
            "num_channels": features.shape[1],
            "spatial_size": (features.shape[2], features.shape[3]),
        }

        return stats

    def compare_layers(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        layer_names: list[str],
        save_path: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compare feature statistics across multiple layers.

        Args:
            image: Input image
            layer_names: List of layer names to compare
            save_path: Optional path to save comparison plot

        Returns:
            Dictionary mapping layer names to statistics

        Example:
            >>> layers = ["backbone.layer2", "backbone.layer3", "backbone.layer4"]
            >>> stats = viz.compare_layers(image, layers)
            >>> for layer, stat in stats.items():
            ...     print(f"{layer}: mean={stat['mean']:.3f}")
        """
        all_stats = {}

        for layer_name in layer_names:
            stats = self.get_feature_statistics(image, layer_name)
            all_stats[layer_name] = stats

        # Create comparison visualization if requested
        if save_path:
            self._plot_layer_comparison(all_stats, save_path)

        return all_stats

    def _plot_layer_comparison(
        self,
        all_stats: dict[str, dict[str, float]],
        save_path: str,
    ):
        """Plot comparison of layer statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Layer Statistics Comparison", fontsize=14, fontweight="bold")

        layer_names = list(all_stats.keys())
        metrics = ["mean", "std", "sparsity", "active_channels"]
        titles = ["Mean Activation", "Std Deviation", "Sparsity", "Active Channels"]

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            values = [stats[metric] for stats in all_stats.values()]

            ax.bar(range(len(layer_names)), values)
            ax.set_xticks(range(len(layer_names)))
            ax.set_xticklabels(
                [name.split(".")[-1] for name in layer_names], rotation=45
            )
            ax.set_title(title, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

            # Add values on bars
            for i, v in enumerate(values):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def get_top_activating_features(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        layer_name: str,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Get channels with highest mean activation.

        Args:
            image: Input image
            layer_name: Name of layer
            top_k: Number of top channels to return

        Returns:
            List of (channel_index, mean_activation) tuples

        Example:
            >>> top_features = viz.get_top_activating_features(
            ...     image, "backbone.layer4", top_k=5
            ... )
            >>> for channel, activation in top_features:
            ...     print(f"Channel {channel}: {activation:.3f}")
        """
        features = self.get_features(image, layer_name)

        # Compute mean activation per channel
        channel_means = features.mean(dim=(2, 3)).squeeze()  # (C,)

        # Get top-k
        top_values, top_indices = torch.topk(
            channel_means, k=min(top_k, len(channel_means))
        )

        return [(idx.item(), val.item()) for idx, val in zip(top_indices, top_values)]

    def visualize_receptive_field(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        layer_name: str,
        channel: int,
        position: tuple[int, int] | None = None,
        save_path: str | None = None,
    ) -> np.ndarray:
        """
        Approximate receptive field visualization using occlusion.

        Args:
            image: Input image
            layer_name: Name of layer
            channel: Channel index to visualize
            position: (h, w) position in feature map (None = center)
            save_path: Optional path to save visualization

        Returns:
            Receptive field heatmap

        Example:
            >>> rf = viz.visualize_receptive_field(
            ...     image,
            ...     layer_name="backbone.layer3",
            ...     channel=42,
            ...     save_path="receptive_field.png"
            ... )
        """
        # Get baseline features
        baseline_features = self.get_features(image, layer_name)

        if position is None:
            # Use center position
            h, w = baseline_features.shape[2:]
            position = (h // 2, w // 2)

        baseline_activation = baseline_features[
            0, channel, position[0], position[1]
        ].item()

        # Prepare image
        input_tensor = self._preprocess_image(image)
        img_h, img_w = input_tensor.shape[2:]

        # Occlusion parameters
        patch_size = 16
        stride = 8

        # Compute sensitivity map
        sensitivity = np.zeros((img_h, img_w))

        for y in range(0, img_h - patch_size, stride):
            for x in range(0, img_w - patch_size, stride):
                # Create occluded image
                occluded = input_tensor.clone()
                occluded[:, :, y : y + patch_size, x : x + patch_size] = 0

                # Get features
                occluded_features = self.get_features(occluded, layer_name)
                occluded_activation = occluded_features[
                    0, channel, position[0], position[1]
                ].item()

                # Compute sensitivity
                sensitivity[y : y + patch_size, x : x + patch_size] = (
                    baseline_activation - occluded_activation
                )

        # Normalize
        sensitivity = np.abs(sensitivity)
        if sensitivity.max() > 0:
            sensitivity = sensitivity / sensitivity.max()

        # Visualize
        if save_path:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Original image
            img_np = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            axes[0].imshow(img_np)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            # Receptive field
            axes[1].imshow(sensitivity, cmap="hot")
            axes[1].set_title(
                f"Receptive Field\nLayer: {layer_name}, Channel: {channel}"
            )
            axes[1].axis("off")

            plt.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return sensitivity

    def _select_features(
        self,
        features: torch.Tensor,
        num_features: int,
        sort_by: str,
    ) -> list[int]:
        """Select which features to display."""
        num_channels = features.shape[1]

        if sort_by == "activation":
            # Sort by mean activation
            channel_means = features.mean(dim=(2, 3)).squeeze()
            _, indices = torch.topk(channel_means, k=min(num_features, num_channels))
            return indices.tolist()

        elif sort_by == "variance":
            # Sort by variance
            channel_vars = features.var(dim=(2, 3)).squeeze()
            _, indices = torch.topk(channel_vars, k=min(num_features, num_channels))
            return indices.tolist()

        elif sort_by == "random":
            # Random selection
            indices = torch.randperm(num_channels)[:num_features]
            return indices.tolist()

        else:
            raise ValueError(f"Unknown sort_by: {sort_by}")

    def _get_layer_by_name(self, name: str) -> nn.Module | None:
        """Get layer by name."""
        try:
            parts = name.split(".")
            module = self.model
            for part in parts:
                module = getattr(module, part)
            return module
        except AttributeError:
            return None

    def _preprocess_image(
        self, image: Image.Image | np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """Preprocess image to tensor."""
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.unsqueeze(0)
            return image.to(self.device)

        if isinstance(image, np.ndarray):
            if image.dtype in [np.float32, np.float64]:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        # Convert PIL to tensor
        import torchvision.transforms as T

        transform = T.Compose([T.ToTensor()])
        tensor = transform(image).unsqueeze(0)
        return tensor.to(self.device)


__all__ = [
    "FeatureVisualizer",
]
