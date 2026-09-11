"""Interactive visualization tools using Plotly for model interpretations.

Provides interactive HTML-based visualizations for exploring explanations:
- Interactive heatmap overlays with zoom/pan
- Side-by-side method comparisons
- HTML report generation
- Hover information and statistics
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class InteractiveVisualizer:
    """
    Create interactive Plotly visualizations for model interpretations.

    Requires: plotly (install with `pip install plotly`)

    Args:
        model: The model being explained
        use_cuda: Whether to use CUDA if available

    Example:
        >>> from autotimm import ImageClassifier
        >>> from autotimm.interpretation import GradCAM, InteractiveVisualizer
        >>>
        >>> model = ImageClassifier(backbone="resnet50", num_classes=10)
        >>> explainer = GradCAM(model)
        >>> viz = InteractiveVisualizer(model)
        >>>
        >>> # Create interactive visualization
        >>> fig = viz.visualize_explanation(
        ...     image,
        ...     explainer,
        ...     title="GradCAM Explanation",
        ...     save_path="explanation.html"
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        use_cuda: bool = True,
    ):
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is required for interactive visualizations. "
                "Install with: pip install plotly"
            )

        self.model = model
        self.model.eval()
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def visualize_explanation(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        explainer,
        target_class: int | None = None,
        title: str = "Model Explanation",
        colorscale: str = "Viridis",
        opacity: float = 0.6,
        save_path: str | None = None,
        show_colorbar: bool = True,
        width: int = 800,
        height: int = 600,
    ) -> go.Figure:
        """
        Create interactive visualization of a single explanation.

        Args:
            image: Input image
            explainer: Interpretation method (e.g., GradCAM)
            target_class: Target class to explain
            title: Figure title
            colorscale: Plotly colorscale (e.g., 'Viridis', 'Hot', 'Jet')
            opacity: Heatmap opacity (0-1)
            save_path: Path to save HTML file
            show_colorbar: Whether to show colorbar
            width: Figure width in pixels
            height: Figure height in pixels

        Returns:
            Plotly Figure object

        Example:
            >>> fig = viz.visualize_explanation(
            ...     image,
            ...     gradcam_explainer,
            ...     title="GradCAM Analysis",
            ...     save_path="gradcam.html"
            ... )
            >>> fig.show()  # Open in browser
        """
        # Get prediction
        input_tensor = self._preprocess_image(image)
        with torch.inference_mode():
            output = self.model(input_tensor)
            if isinstance(output, dict):
                output = output.get(
                    "logits", output.get("output", next(iter(output.values())))
                )
            pred_class = output.argmax(dim=1).item()
            pred_score = torch.softmax(output, dim=1)[0, pred_class].item()

        if target_class is None:
            target_class = pred_class

        # Get explanation
        heatmap = explainer.explain(image, target_class=target_class)

        # Convert image to numpy
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.ndim == 3:
                image_np = image_np.transpose(1, 2, 0)
        elif isinstance(image, np.ndarray):
            image_np = image
        else:  # PIL Image
            image_np = np.array(image)

        # Normalize image for display
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)

        # Resize heatmap if needed
        if heatmap.shape != image_np.shape[:2]:
            import cv2

            heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))

        # Create figure with subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Original Image", "Explanation Overlay"),
            horizontal_spacing=0.05,
        )

        # Original image
        fig.add_trace(go.Image(z=image_np), row=1, col=1)

        # Overlay
        fig.add_trace(go.Image(z=image_np), row=1, col=2)

        # Add heatmap overlay
        fig.add_trace(
            go.Heatmap(
                z=heatmap,
                colorscale=colorscale,
                opacity=opacity,
                showscale=show_colorbar,
                hovertemplate="X: %{x}<br>Y: %{y}<br>Importance: %{z:.3f}<extra></extra>",
                colorbar=(
                    {
                        "title": "Importance",
                        "x": 1.15,
                    }
                    if show_colorbar
                    else None
                ),
            ),
            row=1,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title={
                "text": f"{title}<br><sub>Predicted: Class {pred_class} (confidence: {pred_score:.3f})</sub>",
                "x": 0.5,
                "xanchor": "center",
            },
            width=width,
            height=height,
            showlegend=False,
            hovermode="closest",
        )

        # Remove axes
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

        # Save if requested
        if save_path:
            fig.write_html(save_path)

        return fig

    def compare_methods(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        explainers: dict[str, object],
        target_class: int | None = None,
        title: str = "Method Comparison",
        colorscale: str = "Viridis",
        opacity: float = 0.6,
        save_path: str | None = None,
        width: int = 1200,
        height: int = 400,
    ) -> go.Figure:
        """
        Compare multiple explanation methods interactively.

        Args:
            image: Input image
            explainers: Dictionary of {method_name: explainer_instance}
            target_class: Target class to explain
            title: Figure title
            colorscale: Plotly colorscale
            opacity: Heatmap opacity
            save_path: Path to save HTML file
            width: Figure width
            height: Figure height

        Returns:
            Plotly Figure object

        Example:
            >>> explainers = {
            ...     'GradCAM': GradCAM(model),
            ...     'GradCAM++': GradCAMPlusPlus(model),
            ...     'IntegratedGradients': IntegratedGradients(model),
            ... }
            >>> fig = viz.compare_methods(
            ...     image,
            ...     explainers,
            ...     save_path="comparison.html"
            ... )
        """
        # Get prediction
        input_tensor = self._preprocess_image(image)
        with torch.inference_mode():
            output = self.model(input_tensor)
            if isinstance(output, dict):
                output = output.get(
                    "logits", output.get("output", next(iter(output.values())))
                )
            pred_class = output.argmax(dim=1).item()
            pred_score = torch.softmax(output, dim=1)[0, pred_class].item()

        if target_class is None:
            target_class = pred_class

        # Convert image to numpy
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.ndim == 3:
                image_np = image_np.transpose(1, 2, 0)
        elif isinstance(image, np.ndarray):
            image_np = image
        else:
            image_np = np.array(image)

        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)

        # Create subplots
        n_methods = len(explainers)
        fig = make_subplots(
            rows=1,
            cols=n_methods + 1,
            subplot_titles=["Original"] + list(explainers.keys()),
            horizontal_spacing=0.02,
        )

        # Original image
        fig.add_trace(go.Image(z=image_np), row=1, col=1)

        # Add each method
        for idx, (method_name, explainer) in enumerate(explainers.items(), start=2):
            # Get explanation
            heatmap = explainer.explain(image, target_class=target_class)

            # Resize if needed
            if heatmap.shape != image_np.shape[:2]:
                import cv2

                heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))

            # Add base image
            fig.add_trace(go.Image(z=image_np), row=1, col=idx)

            # Add heatmap overlay
            fig.add_trace(
                go.Heatmap(
                    z=heatmap,
                    colorscale=colorscale,
                    opacity=opacity,
                    showscale=(idx == n_methods + 1),  # Only show colorbar for last
                    hovertemplate=f"{method_name}<br>X: %{{x}}<br>Y: %{{y}}<br>Importance: %{{z:.3f}}<extra></extra>",
                    colorbar=(
                        {
                            "title": "Importance",
                            "x": 1.02,
                        }
                        if idx == n_methods + 1
                        else None
                    ),
                ),
                row=1,
                col=idx,
            )

        # Update layout
        fig.update_layout(
            title={
                "text": f"{title}<br><sub>Predicted: Class {pred_class} (confidence: {pred_score:.3f})</sub>",
                "x": 0.5,
                "xanchor": "center",
            },
            width=width,
            height=height,
            showlegend=False,
            hovermode="closest",
        )

        # Remove axes
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

        if save_path:
            fig.write_html(save_path)

        return fig

    def create_report(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        explainer,
        target_class: int | None = None,
        include_statistics: bool = True,
        save_path: str = "report.html",
        title: str = "Interpretation Report",
    ) -> str:
        """
        Generate comprehensive HTML report with multiple visualizations.

        Args:
            image: Input image
            explainer: Interpretation method
            target_class: Target class to explain
            include_statistics: Include heatmap statistics
            save_path: Path to save HTML report
            title: Report title

        Returns:
            Path to saved HTML file

        Example:
            >>> report_path = viz.create_report(
            ...     image,
            ...     gradcam_explainer,
            ...     save_path="detailed_report.html"
            ... )
            >>> print(f"Report saved to: {report_path}")
        """
        # Get prediction and explanation
        input_tensor = self._preprocess_image(image)
        with torch.inference_mode():
            output = self.model(input_tensor)
            if isinstance(output, dict):
                output = output.get(
                    "logits", output.get("output", next(iter(output.values())))
                )
            pred_class = output.argmax(dim=1).item()
            pred_score = torch.softmax(output, dim=1)[0, pred_class].item()
            top5_scores, top5_classes = torch.topk(torch.softmax(output, dim=1)[0], k=5)

        if target_class is None:
            target_class = pred_class

        heatmap = explainer.explain(image, target_class=target_class)

        # Convert image
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.ndim == 3:
                image_np = image_np.transpose(1, 2, 0)
        elif isinstance(image, np.ndarray):
            image_np = image
        else:
            image_np = np.array(image)

        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)

        # Resize heatmap
        if heatmap.shape != image_np.shape[:2]:
            import cv2

            heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))

        # Create HTML report
        html_parts = []

        # Header
        html_parts.append(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 2px solid #4CAF50;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #555;
                    margin-top: 30px;
                }}
                .info-box {{
                    background-color: #e8f5e9;
                    border-left: 4px solid #4CAF50;
                    padding: 15px;
                    margin: 20px 0;
                }}
                .stat-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background-color: #f9f9f9;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 15px;
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #4CAF50;
                }}
                .stat-label {{
                    font-size: 14px;
                    color: #666;
                    margin-top: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>

                <div class="info-box">
                    <strong>Prediction:</strong> Class {pred_class} (confidence: {pred_score:.2%})<br>
                    <strong>Target Class:</strong> {target_class}<br>
                    <strong>Method:</strong> {explainer.__class__.__name__}
                </div>
        """)

        # Top-5 predictions
        html_parts.append("<h2>Top-5 Predictions</h2>")
        html_parts.append("<table style='width:100%; border-collapse: collapse;'>")
        html_parts.append(
            "<tr style='background-color:#f0f0f0;'><th style='padding:10px; text-align:left;'>Rank</th><th style='padding:10px; text-align:left;'>Class</th><th style='padding:10px; text-align:left;'>Confidence</th></tr>"
        )
        for rank, (cls, score) in enumerate(
            zip(top5_classes.tolist(), top5_scores.tolist()), 1
        ):
            html_parts.append(
                f"<tr><td style='padding:8px;'>{rank}</td><td style='padding:8px;'>Class {cls}</td><td style='padding:8px;'>{score:.2%}</td></tr>"
            )
        html_parts.append("</table>")

        # Statistics
        if include_statistics:
            stats = {
                "Mean": float(heatmap.mean()),
                "Std": float(heatmap.std()),
                "Min": float(heatmap.min()),
                "Max": float(heatmap.max()),
                "Sparsity": float((heatmap == 0).mean()),
            }

            html_parts.append("<h2>Heatmap Statistics</h2>")
            html_parts.append("<div class='stat-grid'>")
            for stat_name, stat_value in stats.items():
                html_parts.append(f"""
                    <div class="stat-card">
                        <div class="stat-value">{stat_value:.3f}</div>
                        <div class="stat-label">{stat_name}</div>
                    </div>
                """)
            html_parts.append("</div>")

        # Main visualization
        html_parts.append("<h2>Explanation Visualization</h2>")
        fig1 = self.visualize_explanation(
            image, explainer, target_class=target_class, title="", show_colorbar=True
        )
        html_parts.append(fig1.to_html(full_html=False, include_plotlyjs=False))

        # Distribution
        html_parts.append("<h2>Importance Distribution</h2>")
        fig2 = go.Figure()
        fig2.add_trace(
            go.Histogram(
                x=heatmap.flatten(),
                nbinsx=50,
                name="Importance",
                marker_color="#4CAF50",
            )
        )
        fig2.update_layout(
            xaxis_title="Importance Value",
            yaxis_title="Frequency",
            height=400,
            showlegend=False,
        )
        html_parts.append(fig2.to_html(full_html=False, include_plotlyjs=False))

        # Footer
        html_parts.append("""
            </div>
        </body>
        </html>
        """)

        # Write file
        with open(save_path, "w") as f:
            f.write("\n".join(html_parts))

        return save_path

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
    "InteractiveVisualizer",
]
