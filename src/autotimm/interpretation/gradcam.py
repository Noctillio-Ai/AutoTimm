"""GradCAM and GradCAM++ implementations."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from autotimm.interpretation.base import BaseInterpreter


class GradCAM(BaseInterpreter):
    """
    GradCAM: Gradient-weighted Class Activation Mapping.

    GradCAM uses the gradients of the target class flowing into the final
    convolutional layer to produce a coarse localization map highlighting
    important regions in the image.

    Reference:
        Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization" (ICCV 2017)

    Args:
        model: PyTorch model to interpret
        target_layer: Layer to use for GradCAM. If None, auto-detects last conv layer
        use_cuda: Whether to use CUDA if available

    Example:
        >>> from autotimm import ImageClassifier
        >>> from autotimm.interpretation import GradCAM
        >>> model = ImageClassifier(backbone="resnet18", num_classes=10)
        >>> explainer = GradCAM(model)
        >>> heatmap = explainer(image, target_class=5)
        >>> # heatmap is numpy array in [0, 1]
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: str | torch.nn.Module | None = None,
        use_cuda: bool = True,
    ):
        super().__init__(model, target_layer, use_cuda)

    def explain(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        target_class: int | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap for input image.

        Args:
            image: Input image
            target_class: Target class to explain. If None, uses predicted class
            normalize: Whether to normalize heatmap to [0, 1]

        Returns:
            Heatmap as numpy array. If normalize=True, values in [0, 1]
        """
        # Register hooks
        self._register_hooks()

        # Preprocess image
        input_tensor = self._preprocess_image(image)
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)

        # Handle different output formats
        if isinstance(output, dict):
            # For AutoTimm models that return dict
            output = output.get("logits", output.get("output", output))

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()

        # Create one-hot output for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1

        # Backward
        output.backward(gradient=one_hot, retain_graph=False)

        # Generate CAM
        # Weight by global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of forward activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # Apply ReLU to focus on positive influences
        cam = F.relu(cam)

        # Normalize if requested
        if normalize:
            cam = self._normalize_cam(cam)

        # Remove hooks
        self._remove_hooks()

        # Convert to numpy
        heatmap = cam.squeeze().cpu().numpy()

        return heatmap

    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        """
        Normalize CAM to [0, 1] range.

        Args:
            cam: Raw CAM tensor

        Returns:
            Normalized CAM
        """
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def explain_batch(
        self,
        images: list[Image.Image | np.ndarray | torch.Tensor],
        target_classes: list[int] | None = None,
    ) -> list[np.ndarray]:
        """
        Generate GradCAM heatmaps for a batch of images.

        Args:
            images: List of input images
            target_classes: List of target classes (None = use predicted)

        Returns:
            List of heatmaps as numpy arrays
        """
        if target_classes is None:
            target_classes = [None] * len(images)

        heatmaps = []
        for image, target_class in zip(images, target_classes):
            heatmap = self.explain(image, target_class)
            heatmaps.append(heatmap)

        return heatmaps


class GradCAMPlusPlus(GradCAM):
    """
    GradCAM++: Improved GradCAM with better localization.

    GradCAM++ uses a weighted combination of the positive partial derivatives
    of the last convolutional layer feature maps with respect to the target
    class score as weights. This provides better localization, especially
    when there are multiple occurrences of the same class in the image.

    Reference:
        Chattopadhyay et al. "Grad-CAM++: Improved Visual Explanations for
        Deep Convolutional Networks" (WACV 2018)

    Args:
        model: PyTorch model to interpret
        target_layer: Layer to use for GradCAM++. If None, auto-detects last conv layer
        use_cuda: Whether to use CUDA if available

    Example:
        >>> from autotimm import ImageClassifier
        >>> from autotimm.interpretation import GradCAMPlusPlus
        >>> model = ImageClassifier(backbone="resnet50", num_classes=10)
        >>> explainer = GradCAMPlusPlus(model)
        >>> heatmap = explainer(image, target_class=5)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: str | torch.nn.Module | None = None,
        use_cuda: bool = True,
    ):
        super().__init__(model, target_layer, use_cuda)

    def explain(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        target_class: int | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate GradCAM++ heatmap for input image.

        Args:
            image: Input image
            target_class: Target class to explain. If None, uses predicted class
            normalize: Whether to normalize heatmap to [0, 1]

        Returns:
            Heatmap as numpy array. If normalize=True, values in [0, 1]
        """
        # Register hooks
        self._register_hooks()

        # Preprocess image
        input_tensor = self._preprocess_image(image)
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)

        # Handle different output formats
        if isinstance(output, dict):
            output = output.get("logits", output.get("output", output))

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()

        # Create one-hot output for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1

        # Backward
        output.backward(gradient=one_hot, retain_graph=True)

        # GradCAM++ weighting
        # Calculate alpha weights
        grad_2 = self.gradients.pow(2)
        grad_3 = self.gradients.pow(3)

        # Sum activations over spatial dimensions
        sum_activations = self.activations.sum(dim=(2, 3), keepdim=True)

        # Alpha calculation
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom

        # Weights calculation
        # Only consider positive gradients
        relu_grad = F.relu(self.gradients)
        weights = (alpha * relu_grad).sum(dim=(2, 3), keepdim=True)

        # Weighted combination
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize if requested
        if normalize:
            cam = self._normalize_cam(cam)

        # Remove hooks
        self._remove_hooks()

        # Convert to numpy
        heatmap = cam.squeeze().cpu().numpy()

        return heatmap
