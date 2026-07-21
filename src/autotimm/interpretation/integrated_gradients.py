"""Integrated Gradients implementation for attribution."""

from typing import Optional, Union, Literal
import torch
import numpy as np
from PIL import Image
import cv2

from autotimm.interpretation.base import BaseInterpreter


class IntegratedGradients(BaseInterpreter):
    """
    Integrated Gradients: Attribution method through path integration.

    Integrated Gradients attributes a prediction to input features by
    integrating gradients along a path from a baseline to the input.
    This provides a principled attribution that satisfies completeness
    and sensitivity axioms.

    Reference:
        Sundararajan et al. "Axiomatic Attribution for Deep Networks"
        (ICML 2017)

    Args:
        model: PyTorch model to interpret
        baseline: Baseline type or custom tensor:
            - 'black': All zeros (default)
            - 'white': All ones
            - 'blur': Gaussian blurred version of input
            - 'random': Random noise
            - torch.Tensor: Custom baseline
        steps: Number of integration steps (default: 50)
        use_cuda: Whether to use CUDA if available

    Example:
        >>> from autotimm import ImageClassifier
        >>> from autotimm.interpretation import IntegratedGradients
        >>> model = ImageClassifier(backbone="resnet18", num_classes=10)
        >>> ig = IntegratedGradients(model, baseline='black', steps=50)
        >>> attribution = ig(image, target_class=5)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        baseline: Union[
            Literal["black", "white", "blur", "random"], torch.Tensor
        ] = "black",
        steps: int = 50,
        use_cuda: bool = True,
    ):
        # Don't call super().__init__ with target_layer since IG doesn't need it
        self.model = model
        self.model.eval()
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Unwrap torch.compile'd modules so backward() can reach leaf inputs
        self._unwrap_compiled_modules()

        self.baseline_type = baseline if isinstance(baseline, str) else "custom"
        self.custom_baseline = baseline if isinstance(baseline, torch.Tensor) else None
        self.steps = steps

        # IG doesn't use target layer, but we need these for compatibility
        self.target_layer = None
        self.activations = None
        self.gradients = None
        self._hooks = []

    def _unwrap_compiled_modules(self):
        """Replace any torch.compile'd submodules with their originals."""
        for name, module in self.model.named_children():
            orig = getattr(module, "_orig_mod", None)
            if orig is not None:
                setattr(self.model, name, orig)

    def _generate_baseline(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate baseline based on type.

        Args:
            input_tensor: Input tensor to match shape

        Returns:
            Baseline tensor
        """
        if self.custom_baseline is not None:
            baseline = self.custom_baseline.to(self.device)
            if baseline.shape != input_tensor.shape:
                raise ValueError(
                    f"Custom baseline shape {baseline.shape} doesn't match "
                    f"input shape {input_tensor.shape}"
                )
            return baseline

        if self.baseline_type == "black":
            return torch.zeros_like(input_tensor)
        elif self.baseline_type == "white":
            return torch.ones_like(input_tensor)
        elif self.baseline_type == "blur":
            # Apply Gaussian blur as baseline
            img_np = input_tensor.detach().squeeze(0).cpu().numpy().transpose(1, 2, 0)
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            blurred = cv2.GaussianBlur(img_np, (51, 51), 50)
            if blurred.max() > 1.0:
                blurred = blurred.astype(np.float32) / 255.0
            blurred = torch.from_numpy(blurred.transpose(2, 0, 1)).unsqueeze(0)
            return blurred.float().to(self.device)
        elif self.baseline_type == "random":
            return torch.rand_like(input_tensor)
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline_type}")

    def explain(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate Integrated Gradients attribution.

        Args:
            image: Input image
            target_class: Target class to explain (None = predicted class)
            normalize: Whether to normalize attribution to [0, 1]

        Returns:
            Attribution map as numpy array
        """
        # Preprocess image
        input_tensor = self._preprocess_image(image)

        # Get predicted class if needed
        with torch.no_grad():
            output = self.model(input_tensor)
            if isinstance(output, dict):
                output = output.get("logits", output.get("output", output))
            if target_class is None:
                target_class = output.argmax(dim=1).item()

        # Generate baseline
        baseline = self._generate_baseline(input_tensor)

        # Compute integrated gradients
        attribution = self._compute_integrated_gradients(
            input_tensor, baseline, target_class
        )

        # Sum across channels to get spatial attribution
        attribution = attribution.sum(dim=1).squeeze()

        # Normalize if requested
        if normalize:
            attribution = torch.abs(attribution)
            attribution = attribution - attribution.min()
            attribution = attribution / (attribution.max() + 1e-8)

        return attribution.cpu().numpy()

    def _compute_integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor,
        target_class: int,
    ) -> torch.Tensor:
        """
        Compute integrated gradients along path from baseline to input.

        Uses the Riemann sum approximation: sample at alpha = k/m for
        k = 1, ..., m (excluding the baseline at alpha=0), average the
        gradients, and multiply by (input - baseline).

        Args:
            input_tensor: Input tensor
            baseline: Baseline tensor
            target_class: Target class index

        Returns:
            Integrated gradients
        """
        # Accumulate gradients
        accumulated_gradients = torch.zeros_like(input_tensor)
        diff = (input_tensor - baseline).detach()

        # Integrate gradients along path (k=1..steps, exclude baseline)
        with torch.enable_grad():
            for step in range(1, self.steps + 1):
                # Linear interpolation
                alpha = step / self.steps
                interpolated = (baseline + alpha * diff).detach().requires_grad_(True)

                # Forward pass
                output = self.model(interpolated)
                if isinstance(output, dict):
                    output = output.get("logits", output.get("output", output))

                # Backward pass
                self.model.zero_grad()
                target_score = output[0, target_class]
                target_score.backward()

                # Accumulate gradients
                if interpolated.grad is not None:
                    accumulated_gradients += interpolated.grad.detach().clone()

        # Average gradients and multiply by (input - baseline)
        averaged_gradients = accumulated_gradients / self.steps
        integrated_gradients = diff * averaged_gradients

        return integrated_gradients.detach()

    def visualize_polarity(
        self,
        attribution: np.ndarray,
        image: Union[Image.Image, np.ndarray],
        save_path: Optional[str] = None,
    ):
        """
        Visualize positive and negative attributions separately.

        Args:
            attribution: Attribution map from explain()
            image: Original image
            save_path: Optional path to save visualization
        """
        import matplotlib.pyplot as plt

        # Convert image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Split into positive and negative
        pos_attr = np.maximum(attribution, 0)
        neg_attr = np.maximum(-attribution, 0)

        # Normalize
        if pos_attr.max() > 0:
            pos_attr = pos_attr / pos_attr.max()
        if neg_attr.max() > 0:
            neg_attr = neg_attr / neg_attr.max()

        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Original
        axes[0].imshow(image)
        axes[0].set_title("Original", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Positive attribution
        axes[1].imshow(pos_attr, cmap="Reds")
        axes[1].set_title("Positive Attribution", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        # Negative attribution
        axes[2].imshow(neg_attr, cmap="Blues")
        axes[2].set_title("Negative Attribution", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        # Combined
        # Create RGB visualization: R=positive, B=negative, G=0
        combined = np.zeros((*attribution.shape, 3))
        combined[:, :, 0] = pos_attr  # Red channel
        combined[:, :, 2] = neg_attr  # Blue channel
        axes[3].imshow(combined)
        axes[3].set_title("Combined (R=Pos, B=Neg)", fontsize=12, fontweight="bold")
        axes[3].axis("off")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)

    def get_completeness_score(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        attribution: np.ndarray,
        target_class: Optional[int] = None,
    ) -> float:
        """
        Compute completeness score (attribution sum vs score difference).

        For Integrated Gradients, the completeness axiom states that the
        sum of attributions should equal the difference between the model
        output at the input and the baseline.

        Args:
            image: Input image
            attribution: Attribution map
            target_class: Target class

        Returns:
            Completeness score (should be close to 0 for perfect completeness)
        """
        input_tensor = self._preprocess_image(image)
        baseline = self._generate_baseline(input_tensor)

        with torch.no_grad():
            # Score at input
            output_input = self.model(input_tensor)
            if isinstance(output_input, dict):
                output_input = output_input.get(
                    "logits", output_input.get("output", output_input)
                )

            # Score at baseline
            output_baseline = self.model(baseline)
            if isinstance(output_baseline, dict):
                output_baseline = output_baseline.get(
                    "logits", output_baseline.get("output", output_baseline)
                )

            if target_class is None:
                target_class = output_input.argmax(dim=1).item()

            score_diff = (
                output_input[0, target_class] - output_baseline[0, target_class]
            ).item()

        # Sum of attributions (need to recompute with full channels)
        full_attribution = self._compute_integrated_gradients(
            input_tensor, baseline, target_class
        )
        attr_sum = full_attribution.sum().item()

        # Completeness error
        completeness_error = abs(score_diff - attr_sum)

        return completeness_error


class SmoothGrad:
    """
    SmoothGrad: Noise reduction for attribution methods.

    SmoothGrad adds Gaussian noise to the input multiple times and
    averages the resulting attributions. This reduces noise in the
    attribution maps and provides smoother visualizations.

    Reference:
        Smilkov et al. "SmoothGrad: removing noise by adding noise"
        (ICML 2017 Workshop)

    Args:
        base_explainer: Base interpretation method (GradCAM, IntegratedGradients, etc.)
        noise_level: Standard deviation of Gaussian noise (default: 0.15)
        num_samples: Number of noisy samples (default: 50)

    Example:
        >>> from autotimm.interpretation import GradCAM, SmoothGrad
        >>> base = GradCAM(model)
        >>> smooth = SmoothGrad(base, noise_level=0.15, num_samples=50)
        >>> attribution = smooth(image, target_class=5)
    """

    def __init__(
        self,
        base_explainer: BaseInterpreter,
        noise_level: float = 0.15,
        num_samples: int = 50,
    ):
        self.base_explainer = base_explainer
        self.noise_level = noise_level
        self.num_samples = num_samples

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate smoothed attribution.

        Args:
            image: Input image
            target_class: Target class to explain
            **kwargs: Additional arguments for base explainer

        Returns:
            Smoothed attribution map
        """
        return self.explain(image, target_class, **kwargs)

    def explain(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate smoothed attribution by averaging over noisy samples.

        Args:
            image: Input image
            target_class: Target class to explain
            **kwargs: Additional arguments for base explainer

        Returns:
            Smoothed attribution map
        """
        # Preprocess to tensor
        input_tensor = self.base_explainer._preprocess_image(image)

        # Accumulate attributions
        accumulated_attr = None

        for i in range(self.num_samples):
            # Add Gaussian noise. No clamping to [0, 1]: `input_tensor` comes
            # from BaseInterpreter._preprocess_image, which normalizes with
            # ImageNet (or backbone-specific) mean/std — values are not bounded
            # to [0, 1], so clamping there would silently destroy most of the
            # signal before attribution is computed.
            noise = torch.randn_like(input_tensor) * self.noise_level
            noisy_input = input_tensor + noise

            # Get attribution from base explainer
            attr = self.base_explainer.explain(noisy_input, target_class, **kwargs)

            # Accumulate
            if accumulated_attr is None:
                accumulated_attr = attr
            else:
                accumulated_attr += attr

        # Average
        smoothed_attr = accumulated_attr / self.num_samples

        return smoothed_attr


__all__ = [
    "IntegratedGradients",
    "SmoothGrad",
]
