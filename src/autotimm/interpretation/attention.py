"""Attention visualization for Vision Transformers."""

from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch import nn

from autotimm.interpretation.base import BaseInterpreter


class AttentionRollout(BaseInterpreter):
    """
    Attention Rollout for Vision Transformers.

    Computes attention flow through transformer layers by recursively
    multiplying attention matrices, showing which input patches the
    model attends to for making predictions.

    Reference:
        Abnar & Zuidema "Quantifying Attention Flow in Transformers" (ACL 2020)

    Args:
        model: Vision Transformer model
        head_fusion: How to combine attention from multiple heads:
            - 'mean': Average across heads (default)
            - 'max': Take maximum across heads
            - 'min': Take minimum across heads
        discard_ratio: Discard lowest attention values (default: 0.9)
        use_cuda: Whether to use CUDA if available

    Example:
        >>> from autotimm import ImageClassifier
        >>> from autotimm.interpretation import AttentionRollout
        >>> model = ImageClassifier(backbone="vit_base_patch16_224", num_classes=10)
        >>> attention = AttentionRollout(model, head_fusion='mean')
        >>> attention_map = attention(image)
    """

    def __init__(
        self,
        model: nn.Module,
        head_fusion: Literal["mean", "max", "min"] = "mean",
        discard_ratio: float = 0.9,
        use_cuda: bool = True,
    ):
        self.model = model
        self.model.eval()
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

        # Storage for attention matrices
        self.attention_matrices = []
        self._hooks = []

        # Don't need target layer for attention rollout
        self.target_layer = None
        self.activations = None
        self.gradients = None

    def _register_attention_hooks(self):
        """Register hooks to capture attention matrices from transformer blocks."""
        self.attention_matrices = []

        def attention_hook(module, input, output):
            # Capture attention weights
            # Different ViT implementations have different formats
            if hasattr(module, "attn_drop"):
                # timm ViT format - attention is in the forward output
                if isinstance(output, tuple) and len(output) > 1:
                    attn = output[1]  # (B, H, N, N)
                    self.attention_matrices.append(attn.detach())

        # Register hooks on attention layers
        for name, module in self.model.named_modules():
            if "attn" in name.lower() and "attn_drop" not in name.lower():
                if (
                    isinstance(module, nn.MultiheadAttention)
                    or "attention" in module.__class__.__name__.lower()
                ):
                    self._hooks.append(module.register_forward_hook(attention_hook))

    def explain(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Generate attention rollout map.

        Args:
            image: Input image
            target_class: Not used for attention rollout (kept for API compatibility)

        Returns:
            Attention map as numpy array
        """
        # Register hooks
        self._register_attention_hooks()

        # Preprocess image
        input_tensor = self._preprocess_image(image)

        # Forward pass
        with torch.inference_mode():
            _ = self.model(input_tensor)

        # Remove hooks
        self._remove_hooks()

        # If no attention matrices captured, return error
        if len(self.attention_matrices) == 0:
            raise ValueError(
                "No attention matrices captured. Model may not be a Vision Transformer "
                "or attention format is not supported. "
                "Supported formats: timm ViT models."
            )

        # Compute attention rollout
        attention_map = self._compute_rollout()

        return attention_map

    def _compute_rollout(self) -> np.ndarray:
        """
        Compute attention rollout from captured attention matrices.

        Returns:
            Attention map as numpy array
        """
        # Start with identity matrix
        num_tokens = self.attention_matrices[0].shape[-1]
        rollout = torch.eye(num_tokens).to(self.attention_matrices[0].device)

        # Roll out through layers
        for attn_matrix in self.attention_matrices:
            # attn_matrix shape: (B, H, N, N)
            # Fuse heads
            if self.head_fusion == "mean":
                attn_matrix = attn_matrix.mean(dim=1)  # (B, N, N)
            elif self.head_fusion == "max":
                attn_matrix = attn_matrix.max(dim=1)[0]
            elif self.head_fusion == "min":
                attn_matrix = attn_matrix.min(dim=1)[0]

            attn_matrix = attn_matrix.squeeze(0)  # (N, N)

            # Discard lowest attention values
            if self.discard_ratio > 0:
                flat = attn_matrix.view(-1)
                threshold_index = int(flat.shape[0] * self.discard_ratio)
                threshold = torch.sort(flat)[0][threshold_index]
                attn_matrix = torch.where(
                    attn_matrix < threshold, torch.zeros_like(attn_matrix), attn_matrix
                )

            # Add residual connection (identity)
            attn_matrix = attn_matrix + torch.eye(num_tokens).to(attn_matrix.device)

            # Normalize
            attn_matrix = attn_matrix / attn_matrix.sum(dim=-1, keepdim=True)

            # Multiply with rollout
            rollout = torch.matmul(attn_matrix, rollout)

        # Get attention for CLS token (first token) to all patches
        # Assuming token 0 is CLS token
        attention_map = rollout[0, 1:]  # Exclude CLS token itself

        # Reshape to 2D grid
        grid_size = int(np.sqrt(attention_map.shape[0]))
        attention_map = attention_map.reshape(grid_size, grid_size)

        return attention_map.cpu().numpy()

    def visualize(
        self,
        attention_map: np.ndarray,
        image: Image.Image | np.ndarray,
        save_path: str | None = None,
        alpha: float = 0.5,
        colormap: str = "viridis",
    ) -> np.ndarray:
        """
        Visualize attention map overlaid on image.

        Args:
            attention_map: Attention map from explain()
            image: Original image
            save_path: Optional path to save visualization
            alpha: Overlay transparency
            colormap: Matplotlib colormap name

        Returns:
            Visualization as numpy array
        """
        from autotimm.interpretation.visualization.heatmap import overlay_heatmap

        # Convert image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Create overlay
        overlayed = overlay_heatmap(
            image,
            attention_map,
            alpha=alpha,
            colormap=colormap,
            resize_heatmap=True,
        )

        # Save if requested
        if save_path:
            Image.fromarray(overlayed).save(save_path)

        return overlayed


class AttentionFlow:
    """
    Visualize attention flow between specific patches in Vision Transformers.

    Shows how attention flows from one patch to other patches through
    the network, useful for understanding spatial relationships learned
    by the model.

    Args:
        model: Vision Transformer model
        use_cuda: Whether to use CUDA if available

    Example:
        >>> from autotimm.interpretation import AttentionFlow
        >>> flow = AttentionFlow(model)
        >>> flow_map = flow(image, from_patch=50, layer_idx=-1)
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

        self.attention_matrices = []
        self._hooks = []

    def __call__(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        from_patch: int,
        layer_idx: int = -1,
    ) -> np.ndarray:
        """
        Compute attention flow from specific patch.

        Args:
            image: Input image
            from_patch: Source patch index (0 = CLS token)
            layer_idx: Layer to visualize (-1 = last layer)

        Returns:
            Attention flow map as numpy array
        """
        return self.get_attention_flow(image, from_patch, layer_idx)

    def get_attention_flow(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        from_patch: int,
        layer_idx: int = -1,
    ) -> np.ndarray:
        """
        Get attention flow from a specific patch at a specific layer.

        Args:
            image: Input image
            from_patch: Source patch index
            layer_idx: Layer index (-1 for last layer)

        Returns:
            Attention flow map
        """
        # Register hooks (similar to AttentionRollout)
        self._register_attention_hooks()

        # Preprocess and forward
        if isinstance(image, (Image.Image, np.ndarray)):
            # Convert to tensor
            import torchvision.transforms as T

            if isinstance(image, Image.Image):
                transform = T.Compose([T.ToTensor()])
                input_tensor = transform(image).unsqueeze(0).to(self.device)
            else:
                if image.dtype in [np.float32, np.float64]:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                pil_image = Image.fromarray(image)
                transform = T.Compose([T.ToTensor()])
                input_tensor = transform(pil_image).unsqueeze(0).to(self.device)
        else:
            input_tensor = image.to(self.device)
            if input_tensor.ndim == 3:
                input_tensor = input_tensor.unsqueeze(0)

        # Forward pass
        with torch.inference_mode():
            _ = self.model(input_tensor)

        # Remove hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

        if len(self.attention_matrices) == 0:
            raise ValueError("No attention matrices captured.")

        # Get attention from specified layer
        attn_matrix = self.attention_matrices[layer_idx]  # (B, H, N, N)

        # Average across heads
        attn_matrix = attn_matrix.mean(dim=1).squeeze(0)  # (N, N)

        # Get attention from specific patch
        attention_from_patch = attn_matrix[from_patch]  # (N,)

        # Exclude CLS token and reshape to grid
        if from_patch == 0:
            # From CLS token
            attention_map = attention_from_patch[1:]
        else:
            attention_map = attention_from_patch[1:]

        # Reshape to 2D
        grid_size = int(np.sqrt(attention_map.shape[0]))
        attention_map = attention_map.reshape(grid_size, grid_size)

        return attention_map.cpu().numpy()

    def _register_attention_hooks(self):
        """Register hooks to capture attention matrices."""
        self.attention_matrices = []

        def attention_hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attn = output[1]
                self.attention_matrices.append(attn.detach())

        for name, module in self.model.named_modules():
            if "attn" in name.lower() and "attn_drop" not in name.lower():
                if (
                    isinstance(module, nn.MultiheadAttention)
                    or "attention" in module.__class__.__name__.lower()
                ):
                    self._hooks.append(module.register_forward_hook(attention_hook))


__all__ = [
    "AttentionFlow",
    "AttentionRollout",
]
