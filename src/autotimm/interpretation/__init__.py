"""Model interpretation and visualization tools for AutoTimm."""

from autotimm.interpretation.adapters import (
    explain_detection,
    explain_segmentation,
)
from autotimm.interpretation.api import (
    compare_methods,
    explain_prediction,
    quick_explain,
    visualize_batch,
)
from autotimm.interpretation.attention import AttentionFlow, AttentionRollout
from autotimm.interpretation.base import BaseInterpreter
from autotimm.interpretation.callbacks import (
    FeatureMonitorCallback,
    InterpretationCallback,
)
from autotimm.interpretation.feature_viz import FeatureVisualizer
from autotimm.interpretation.gradcam import GradCAM, GradCAMPlusPlus
from autotimm.interpretation.integrated_gradients import (
    IntegratedGradients,
    SmoothGrad,
)
from autotimm.interpretation.metrics import ExplanationMetrics

# Optional: Interactive visualization (requires plotly)
try:
    from autotimm.interpretation.interactive import InteractiveVisualizer

    _INTERACTIVE_AVAILABLE = True
except ImportError:
    _INTERACTIVE_AVAILABLE = False
    InteractiveVisualizer = None

# Performance optimization utilities
from autotimm.interpretation.optimization import (
    BatchProcessor,
    ExplanationCache,
    PerformanceProfiler,
    optimize_for_inference,
)

__all__ = [
    "AttentionFlow",
    "AttentionRollout",
    # Base
    "BaseInterpreter",
    "BatchProcessor",
    # Performance optimization
    "ExplanationCache",
    # Metrics
    "ExplanationMetrics",
    "FeatureMonitorCallback",
    # Feature visualization
    "FeatureVisualizer",
    # Methods
    "GradCAM",
    "GradCAMPlusPlus",
    "IntegratedGradients",
    # Interactive (optional)
    "InteractiveVisualizer",
    # Callbacks
    "InterpretationCallback",
    "PerformanceProfiler",
    "SmoothGrad",
    "compare_methods",
    # Task-specific
    "explain_detection",
    # High-level API
    "explain_prediction",
    "explain_segmentation",
    "optimize_for_inference",
    "quick_explain",
    "visualize_batch",
]
