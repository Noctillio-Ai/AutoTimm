"""Quantitative evaluation metrics for model interpretations.

This module provides metrics to evaluate the quality of explanation methods:
- Faithfulness: How well does the explanation reflect model behavior?
- Sensitivity: Does the explanation respond to meaningful changes?
- Localization: Does the explanation correctly identify important regions?
- Sanity checks: Does the method behave reasonably?
"""

from typing import Optional, Union, Dict, List
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from autotimm.core.logging import logger

# Guards against division by a near-zero prediction score (e.g. the target
# class starts at ~0 confidence), which would otherwise blow up AUC/rise
# values to inf without any indication something degenerate happened.
_EPS = 1e-8


class ExplanationMetrics:
    """
    Compute quantitative metrics for explanation quality.

    Args:
        model: The model being explained
        explainer: The interpretation method to evaluate
        use_cuda: Whether to use CUDA if available

    Example:
        >>> from autotimm import ImageClassifier
        >>> from autotimm.interpretation import GradCAM, ExplanationMetrics
        >>>
        >>> model = ImageClassifier(backbone="resnet50", num_classes=10)
        >>> explainer = GradCAM(model)
        >>> metrics = ExplanationMetrics(model, explainer)
        >>>
        >>> # Evaluate faithfulness
        >>> deletion_score = metrics.deletion(image, steps=50)
        >>> insertion_score = metrics.insertion(image, steps=50)
    """

    def __init__(
        self,
        model: nn.Module,
        explainer,
        use_cuda: bool = True,
    ):
        self.model = model
        self.model.eval()
        self.explainer = explainer
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def deletion(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
        steps: int = 50,
        baseline: str = "blur",
    ) -> Dict[str, float]:
        """
        Deletion metric: progressively delete most important pixels.

        Measures how prediction confidence drops as most important pixels
        (according to the explanation) are removed.

        Args:
            image: Input image
            target_class: Target class (None = predicted class)
            steps: Number of deletion steps
            baseline: What to replace deleted pixels with ('blur', 'black', 'mean')

        Returns:
            Dictionary with:
                - auc: Area under the curve (lower = better explanation)
                - final_drop: Final prediction drop (higher = better)
                - scores: List of prediction scores at each step

        Example:
            >>> result = metrics.deletion(image, steps=50)
            >>> print(f"Deletion AUC: {result['auc']:.3f}")
            >>> print(f"Final drop: {result['final_drop']:.2%}")
        """
        # Get original prediction
        input_tensor = self._preprocess_image(image)
        _, original_score, target_class = self._get_prediction(
            input_tensor, target_class
        )

        # Get explanation
        heatmap = self.explainer.explain(image, target_class=target_class)

        # Resize heatmap to image size if needed
        if heatmap.shape != input_tensor.shape[2:]:
            heatmap = self._resize_heatmap(heatmap, input_tensor.shape[2:])

        # Create baseline
        baseline_tensor = self._create_baseline(input_tensor, baseline)

        # Get pixel importance ranking
        pixel_order = np.argsort(heatmap.flatten())[::-1]  # Most important first

        # Progressively delete pixels
        scores = [original_score]
        modified = input_tensor.clone()

        pixels_per_step = len(pixel_order) // steps

        for step in range(1, steps + 1):
            # Delete pixels
            pixels_to_delete = pixel_order[: step * pixels_per_step]

            # Convert flat indices to 2D
            rows = pixels_to_delete // heatmap.shape[1]
            cols = pixels_to_delete % heatmap.shape[1]

            # Replace with baseline (vectorized — avoids a Python-level loop
            # over up to H*W pixels per step)
            modified[:, :, rows, cols] = baseline_tensor[:, :, rows, cols]

            # Get new prediction
            with torch.no_grad():
                output = self.model(modified)
                if isinstance(output, dict):
                    output = output.get(
                        "logits", output.get("output", list(output.values())[0])
                    )
                probs = torch.softmax(output, dim=1)
                score = probs[0, target_class].item()

            scores.append(score)

        # Compute metrics
        safe_original_score = max(original_score, _EPS)
        try:
            auc = np.trapezoid(scores, dx=1.0 / steps) / safe_original_score
        except AttributeError:
            # Fallback for older numpy versions
            auc = np.trapz(scores, dx=1.0 / steps) / safe_original_score
        final_drop = (original_score - scores[-1]) / safe_original_score

        return {
            "auc": auc,
            "final_drop": final_drop,
            "scores": scores,
            "original_score": original_score,
        }

    def insertion(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
        steps: int = 50,
        baseline: str = "blur",
    ) -> Dict[str, float]:
        """
        Insertion metric: progressively insert most important pixels.

        Measures how prediction confidence rises as most important pixels
        (according to the explanation) are added to a baseline.

        Args:
            image: Input image
            target_class: Target class (None = predicted class)
            steps: Number of insertion steps
            baseline: Initial baseline ('blur', 'black', 'mean')

        Returns:
            Dictionary with:
                - auc: Area under the curve (higher = better explanation)
                - final_rise: Final prediction rise (higher = better)
                - scores: List of prediction scores at each step

        Example:
            >>> result = metrics.insertion(image, steps=50)
            >>> print(f"Insertion AUC: {result['auc']:.3f}")
            >>> print(f"Final rise: {result['final_rise']:.2%}")
        """
        # Get original prediction
        input_tensor = self._preprocess_image(image)
        _, original_score, target_class = self._get_prediction(
            input_tensor, target_class
        )

        # Get explanation
        heatmap = self.explainer.explain(image, target_class=target_class)

        # Resize heatmap to image size if needed
        if heatmap.shape != input_tensor.shape[2:]:
            heatmap = self._resize_heatmap(heatmap, input_tensor.shape[2:])

        # Create baseline
        baseline_tensor = self._create_baseline(input_tensor, baseline)

        # Get baseline prediction
        with torch.no_grad():
            output = self.model(baseline_tensor)
            if isinstance(output, dict):
                output = output.get(
                    "logits", output.get("output", list(output.values())[0])
                )
            probs = torch.softmax(output, dim=1)
            baseline_score = probs[0, target_class].item()

        # Get pixel importance ranking
        pixel_order = np.argsort(heatmap.flatten())[::-1]  # Most important first

        # Progressively insert pixels
        scores = [baseline_score]
        modified = baseline_tensor.clone()

        pixels_per_step = len(pixel_order) // steps

        for step in range(1, steps + 1):
            # Insert pixels
            pixels_to_insert = pixel_order[: step * pixels_per_step]

            # Convert flat indices to 2D
            rows = pixels_to_insert // heatmap.shape[1]
            cols = pixels_to_insert % heatmap.shape[1]

            # Replace with original pixels (vectorized — avoids a Python-level
            # loop over up to H*W pixels per step)
            modified[:, :, rows, cols] = input_tensor[:, :, rows, cols]

            # Get new prediction
            with torch.no_grad():
                output = self.model(modified)
                if isinstance(output, dict):
                    output = output.get(
                        "logits", output.get("output", list(output.values())[0])
                    )
                probs = torch.softmax(output, dim=1)
                score = probs[0, target_class].item()

            scores.append(score)

        # Compute metrics
        safe_original_score = max(original_score, _EPS)
        try:
            auc = np.trapezoid(scores, dx=1.0 / steps) / safe_original_score
        except AttributeError:
            # Fallback for older numpy versions
            auc = np.trapz(scores, dx=1.0 / steps) / safe_original_score

        # Compute final rise - handle edge case where baseline >= original
        denominator = original_score - baseline_score
        if abs(denominator) < 1e-6:
            # Baseline and original are essentially the same
            final_rise = 0.0
        else:
            # Use absolute value to avoid negative denominators
            final_rise = (scores[-1] - baseline_score) / abs(denominator)

        return {
            "auc": auc,
            "final_rise": final_rise,
            "scores": scores,
            "original_score": original_score,
            "baseline_score": baseline_score,
        }

    def sensitivity_n(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
        n_samples: int = 50,
        noise_level: float = 0.15,
    ) -> Dict[str, float]:
        """
        Sensitivity-n metric: explanation stability under input perturbations.

        Measures how much the explanation changes when small noise is added
        to the input. Lower values indicate more stable explanations.

        Args:
            image: Input image
            target_class: Target class
            n_samples: Number of noisy samples
            noise_level: Standard deviation of Gaussian noise

        Returns:
            Dictionary with:
                - sensitivity: Average explanation change
                - std: Standard deviation of changes
                - max_change: Maximum change observed

        Example:
            >>> result = metrics.sensitivity_n(image, n_samples=30)
            >>> print(f"Sensitivity: {result['sensitivity']:.4f}")
        """
        input_tensor = self._preprocess_image(image)
        _, _, target_class = self._get_prediction(input_tensor, target_class)

        # Get original explanation
        original_heatmap = self.explainer.explain(image, target_class=target_class)

        # Use the explainer's own preprocessing (which may apply model-specific
        # normalization) so noisy samples live in the same domain as the tensor
        # `explainer.explain()` used to produce `original_heatmap` above. Using
        # this class's own `_preprocess_image` (plain [0, 1] ToTensor) here would
        # feed the explainer an incorrectly-scaled tensor, since a raw tensor is
        # passed through by BaseInterpreter._preprocess_image without renormalizing.
        explainer_input = self.explainer._preprocess_image(image)

        # Generate noisy versions
        changes = []
        for _ in range(n_samples):
            # Add noise. No clamping: `explainer_input` may be ImageNet-normalized
            # (roughly in [-2, 2.6]), so clamping to [0, 1] would destroy signal.
            noise = torch.randn_like(explainer_input) * noise_level
            noisy_input = explainer_input + noise

            # Get explanation for noisy input
            noisy_heatmap = self.explainer.explain(
                noisy_input, target_class=target_class
            )

            # Compute change
            change = np.abs(noisy_heatmap - original_heatmap).mean()
            changes.append(change)

        return {
            "sensitivity": np.mean(changes),
            "std": np.std(changes),
            "max_change": np.max(changes),
            "changes": changes,
        }

    def model_parameter_randomization_test(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Sanity check: explanation should change significantly if model is randomized.

        Tests whether the explanation method is sensitive to model parameters.
        A good explanation method should produce very different results when
        model weights are randomized.

        Args:
            image: Input image
            target_class: Target class

        Returns:
            Dictionary with:
                - correlation: Correlation between original and randomized explanations
                  (lower = better, method is sensitive to model)
                - change: Mean absolute difference
                - passes: True if correlation < 0.5 (reasonable threshold)

        Example:
            >>> result = metrics.model_parameter_randomization_test(image)
            >>> print(f"Passes sanity check: {result['passes']}")
            >>> print(f"Correlation: {result['correlation']:.3f}")
        """
        input_tensor = self._preprocess_image(image)
        _, _, target_class = self._get_prediction(input_tensor, target_class)

        # Get original explanation
        original_heatmap = self.explainer.explain(image, target_class=target_class)

        # Save original model state
        original_state = {
            name: param.clone() for name, param in self.model.named_parameters()
        }

        # Randomize model parameters
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    param.data = torch.randn_like(param.data)

        try:
            # Get explanation with randomized model
            randomized_heatmap = self.explainer.explain(
                image, target_class=target_class
            )
        finally:
            # Always restore original parameters, even if explain() raises —
            # otherwise the model is left permanently corrupted.
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.data = original_state[name].data

        # Compute correlation
        correlation = np.corrcoef(
            original_heatmap.flatten(), randomized_heatmap.flatten()
        )[0, 1]

        # Compute mean absolute change
        change = np.abs(randomized_heatmap - original_heatmap).mean()

        # Test passes if correlation is low (< 0.5)
        passes = correlation < 0.5

        return {
            "correlation": correlation,
            "change": change,
            "passes": passes,
        }

    def data_randomization_test(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Sanity check: explanation should change significantly if input is randomized.

        Tests whether the explanation is sensitive to input data.
        A good explanation method should produce very different results when
        labels are randomized (i.e., explaining wrong class).

        Args:
            image: Input image
            target_class: Target class

        Returns:
            Dictionary with:
                - correlation: Correlation with random label explanation
                - change: Mean absolute difference
                - passes: True if correlation < 0.5

        Example:
            >>> result = metrics.data_randomization_test(image)
            >>> print(f"Passes sanity check: {result['passes']}")
        """
        input_tensor = self._preprocess_image(image)
        _, _, target_class = self._get_prediction(input_tensor, target_class)

        # Get original explanation
        original_heatmap = self.explainer.explain(image, target_class=target_class)

        # Get explanation for random class (different from target)
        num_classes = self._get_num_classes()
        random_classes = [c for c in range(num_classes) if c != target_class]
        if random_classes:
            random_class = np.random.choice(random_classes)
            random_heatmap = self.explainer.explain(image, target_class=random_class)
        else:
            # If only one class, use same class (test will fail, which is correct)
            random_heatmap = original_heatmap

        # Compute correlation
        correlation = np.corrcoef(original_heatmap.flatten(), random_heatmap.flatten())[
            0, 1
        ]

        # Compute change
        change = np.abs(random_heatmap - original_heatmap).mean()

        # Test passes if correlation is low
        passes = correlation < 0.5

        return {
            "correlation": correlation,
            "change": change,
            "passes": passes,
        }

    def pointing_game(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        bbox: List[float],
        target_class: Optional[int] = None,
    ) -> Dict[str, bool]:
        """
        Pointing game: does the max attention fall within the object bbox?

        Used for object detection/localization. Tests whether the most
        important pixel (according to explanation) falls within the
        ground-truth bounding box.

        Args:
            image: Input image
            bbox: Ground truth bounding box [x1, y1, x2, y2]
            target_class: Target class

        Returns:
            Dictionary with:
                - hit: True if max attention is inside bbox
                - max_location: (row, col) of maximum attention
                - bbox: Input bbox

        Example:
            >>> bbox = [100, 150, 300, 400]  # Object location
            >>> result = metrics.pointing_game(image, bbox)
            >>> print(f"Hit: {result['hit']}")
        """
        # Get explanation
        heatmap = self.explainer.explain(image, target_class=target_class)

        # Find max location
        max_idx = np.argmax(heatmap)
        max_row = max_idx // heatmap.shape[1]
        max_col = max_idx % heatmap.shape[1]

        # Convert to original image coordinates
        input_tensor = self._preprocess_image(image)
        h_ratio = input_tensor.shape[2] / heatmap.shape[0]
        w_ratio = input_tensor.shape[3] / heatmap.shape[1]

        max_y = max_row * h_ratio
        max_x = max_col * w_ratio

        # Check if inside bbox
        x1, y1, x2, y2 = bbox
        hit = (x1 <= max_x <= x2) and (y1 <= max_y <= y2)

        return {
            "hit": hit,
            "max_location": (max_y, max_x),
            "bbox": bbox,
        }

    def evaluate_all(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
        bbox: Optional[List[float]] = None,
    ) -> Dict[str, Dict]:
        """
        Run all applicable metrics on an image.

        Args:
            image: Input image
            target_class: Target class
            bbox: Optional bounding box for pointing game

        Returns:
            Dictionary with all metric results

        Example:
            >>> results = metrics.evaluate_all(image)
            >>> print(f"Deletion AUC: {results['deletion']['auc']:.3f}")
            >>> print(f"Insertion AUC: {results['insertion']['auc']:.3f}")
            >>> print(f"Sensitivity: {results['sensitivity']['sensitivity']:.4f}")
        """
        results = {}

        # Faithfulness metrics
        logger.info("Computing deletion metric...")
        results["deletion"] = self.deletion(image, target_class=target_class)

        logger.info("Computing insertion metric...")
        results["insertion"] = self.insertion(image, target_class=target_class)

        # Sensitivity
        logger.info("Computing sensitivity...")
        results["sensitivity"] = self.sensitivity_n(image, target_class=target_class)

        # Sanity checks
        logger.info("Running model parameter randomization test...")
        results["param_randomization"] = self.model_parameter_randomization_test(
            image, target_class=target_class
        )

        logger.info("Running data randomization test...")
        results["data_randomization"] = self.data_randomization_test(
            image, target_class=target_class
        )

        # Pointing game (if bbox provided)
        if bbox is not None:
            results["pointing_game"] = self.pointing_game(
                image, bbox, target_class=target_class
            )

        return results

    # Helper methods

    def _preprocess_image(
        self, image: Union[Image.Image, np.ndarray, torch.Tensor]
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

    def _get_prediction(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ):
        """Get model prediction."""
        with torch.no_grad():
            output = self.model(input_tensor)
            if isinstance(output, dict):
                output = output.get(
                    "logits", output.get("output", list(output.values())[0])
                )
            probs = torch.softmax(output, dim=1)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            pred_class = output.argmax(dim=1).item()
            score = probs[0, target_class].item()

        return pred_class, score, target_class

    def _create_baseline(
        self, input_tensor: torch.Tensor, baseline: str
    ) -> torch.Tensor:
        """Create baseline tensor."""
        if baseline == "black":
            return torch.zeros_like(input_tensor)
        elif baseline == "mean":
            mean = input_tensor.mean()
            return torch.full_like(input_tensor, mean)
        elif baseline == "blur":
            import torchvision.transforms.functional as TF

            # Apply Gaussian blur
            blurred = TF.gaussian_blur(input_tensor, kernel_size=51, sigma=20.0)
            return blurred
        else:
            raise ValueError(f"Unknown baseline: {baseline}")

    def _resize_heatmap(self, heatmap: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize heatmap to target size."""
        import cv2

        return cv2.resize(heatmap, (target_size[1], target_size[0]))

    def _get_num_classes(self) -> int:
        """Get number of output classes."""
        try:
            # Try to infer from model
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                output = self.model(dummy_input)
                if isinstance(output, dict):
                    output = output.get(
                        "logits", output.get("output", list(output.values())[0])
                    )
                return output.shape[1]
        except Exception:
            # Default to 1000 (ImageNet)
            return 1000


__all__ = [
    "ExplanationMetrics",
]
