# Model Interpretation & Visualization

AutoTimm provides comprehensive tools for interpreting and visualizing deep learning models. Understanding what your models learn and how they make decisions is crucial for debugging, building trust, and improving performance.

## Interpretation Workflow

### Choose a Method

```mermaid
graph LR
    A[Model + Image] --> B{Method}
    B --> C1[<b>GradCAM</b><br/>Gradient-weighted<br/>activation maps]
    B --> C2[<b>Integrated Gradients</b><br/>Path-based<br/>attribution]
    B --> C3[<b>Attention Rollout</b><br/>ViT attention<br/>aggregation]
    B --> C4[<b>SmoothGrad</b><br/>Noise-averaged<br/>gradients]

    style A fill:#1565C0,stroke:#0D47A1
    style B fill:#FF9800,stroke:#F57C00
    style C1 fill:#1976D2,stroke:#1565C0
    style C2 fill:#1976D2,stroke:#1565C0
    style C3 fill:#1976D2,stroke:#1565C0
    style C4 fill:#1976D2,stroke:#1565C0
```

### Generate & Visualize

```mermaid
graph LR
    A[Heatmap] --> B{Task Adapter}
    B --> C1[Classification<br/>Class attribution overlay]
    B --> C2[Detection<br/>Per-box attribution]
    B --> C3[Segmentation<br/>Per-class pixel maps]
    C1 --> D[Visualization<br/>Plots + HTML Reports]
    C2 --> D
    C3 --> D

    style A fill:#1565C0,stroke:#0D47A1
    style B fill:#FF9800,stroke:#F57C00
    style C1 fill:#1976D2,stroke:#1565C0
    style C2 fill:#1976D2,stroke:#1565C0
    style C3 fill:#1976D2,stroke:#1565C0
    style D fill:#4CAF50,stroke:#388E3C
```

### Evaluate Quality

```mermaid
graph LR
    A[Explanation] --> B1[Faithfulness<br/>Perturbation test]
    A --> B2[Sensitivity<br/>Input variation]
    A --> B3[Localization<br/>Ground truth IoU]
    B1 --> C[Metric Scores<br/>+ Comparison Report]
    B2 --> C
    B3 --> C

    style A fill:#1565C0,stroke:#0D47A1
    style B1 fill:#1976D2,stroke:#1565C0
    style B2 fill:#1976D2,stroke:#1565C0
    style B3 fill:#1976D2,stroke:#1565C0
    style C fill:#4CAF50,stroke:#388E3C
```

## Overview

The interpretation module offers:

- **Multiple Explanation Methods**: GradCAM, GradCAM++, Integrated Gradients, SmoothGrad, Attention Visualization
- **Task-Specific Adapters**: Support for classification, object detection, and semantic segmentation
- **Feature Visualization**: Analyze and visualize feature maps from any layer
- **Training Integration**: Automatic interpretation logging during training via callbacks
- **Quality Metrics**: Quantitatively evaluate explanation faithfulness, sensitivity, and localization
- **Interactive Visualizations**: Plotly-based HTML reports with zoom, pan, and hover capabilities
- **Performance Optimization**: Caching, batch processing, and profiling for up to 100x speedup
- **Production-Ready**: High-level API with sensible defaults and extensive customization options

## Quick Start

```python
import autotimm as at  # recommended alias
from autotimm import ImageClassifier
from autotimm.interpretation import explain_prediction
from PIL import Image

# Load model
model = ImageClassifier.load_from_checkpoint("model.ckpt", compile_model=False)

# Load image
image = Image.open("cat.jpg")

# Explain prediction
result = explain_prediction(
    model,
    image,
    method="gradcam",
    save_path="explanation.png"
)

print(f"Predicted class: {result['predicted_class']}")
```

## Interpretation Methods

### GradCAM (Gradient-weighted Class Activation Mapping)

GradCAM uses gradients flowing into the final convolutional layer to produce a localization map highlighting important regions.

```python
from autotimm.interpretation import GradCAM

explainer = GradCAM(model, target_layer="backbone.layer4")
heatmap = explainer.explain(image, target_class=5)
```

**Best for**: Quick visualizations, CNN models, class-discriminative localization

### GradCAM++

An improved version of GradCAM that provides better localization for multiple occurrences of objects.

```python
from autotimm.interpretation import GradCAMPlusPlus

explainer = GradCAMPlusPlus(model, target_layer="backbone.layer4")
heatmap = explainer.explain(image)
```

**Best for**: Multiple objects, overlapping objects, improved localization

### Integrated Gradients

Path-based attribution method that satisfies axioms like completeness and sensitivity.

```python
from autotimm.interpretation import IntegratedGradients

explainer = IntegratedGradients(
    model,
    baseline='black',  # or 'white', 'blur', 'random'
    steps=50
)
heatmap = explainer.explain(image, target_class=3)
```

**Best for**: Pixel-level attributions, theoretical guarantees, understanding feature importance

### SmoothGrad

Reduces noise in attribution maps by averaging over multiple noisy versions of the input.

```python
from autotimm.interpretation import SmoothGrad, GradCAM

base_explainer = GradCAM(model)
smooth_explainer = SmoothGrad(
    base_explainer,
    noise_level=0.15,
    num_samples=50
)
heatmap = smooth_explainer.explain(image)
```

**Best for**: Cleaner visualizations, reducing noise, improving visual quality

### Attention Visualization (Vision Transformers)

For Vision Transformers, visualize attention patterns to understand which patches the model focuses on.

```python
from autotimm.interpretation import AttentionRollout, AttentionFlow

# Attention Rollout (recursive aggregation)
rollout = AttentionRollout(vit_model, head_fusion='mean')
attention_map = rollout.explain(image)

# Attention Flow (patch-to-patch attention)
flow = AttentionFlow(vit_model, target_patch=0)
flow_map = flow.explain(image)
```

**Best for**: Vision Transformers, understanding attention patterns, patch-level analysis

## Task-Specific Interpretation

### Object Detection

Explain individual detections with bounding box highlighting:

```python
from autotimm.interpretation import explain_detection

results = explain_detection(
    detector_model,
    image,
    method='gradcam',
    detection_threshold=0.5,
    save_path='detection_explanation.png'
)
```

### Semantic Segmentation

Explain predictions with optional uncertainty visualization:

```python
from autotimm.interpretation import explain_segmentation

results = explain_segmentation(
    segmentation_model,
    image,
    target_class=5,  # Explain specific class
    show_uncertainty=True,
    uncertainty_method='entropy',
    save_path='segmentation_explanation.png'
)
```

## Feature Visualization

Analyze and visualize what features your model learns:

```python
from autotimm.interpretation import FeatureVisualizer

viz = FeatureVisualizer(model)

# Visualize feature maps
viz.plot_feature_maps(
    image,
    layer_name="backbone.layer3",
    num_features=16,
    sort_by="activation",
    save_path="features.png"
)

# Get feature statistics
stats = viz.get_feature_statistics(image, layer_name="backbone.layer4")
print(f"Mean activation: {stats['mean']:.3f}")
print(f"Sparsity: {stats['sparsity']:.2%}")

# Compare multiple layers
layer_stats = viz.compare_layers(
    image,
    ["backbone.layer2", "backbone.layer3", "backbone.layer4"],
    save_path="layer_comparison.png"
)

# Find most active channels
top_channels = viz.get_top_activating_features(
    image,
    layer_name="backbone.layer4",
    top_k=10
)
```

## Training Integration

### Automatic Interpretation Logging

Monitor model interpretations during training:

```python
from autotimm import AutoTrainer, LoggerConfig
from autotimm.interpretation import InterpretationCallback

# Sample images for monitoring
sample_images = [load_image(f"sample_{i}.jpg") for i in range(8)]

# Create callback
interp_callback = InterpretationCallback(
    sample_images=sample_images,
    method="gradcam",
    log_every_n_epochs=5,
    num_samples=8,
    colormap="viridis",
)

# Train with automatic interpretation
trainer = AutoTrainer(
    max_epochs=100,
    callbacks=[interp_callback],
    logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
)
trainer.fit(model, datamodule=data)
```

### Feature Monitoring

Track feature statistics during training:

```python
from autotimm.interpretation import FeatureMonitorCallback

feature_callback = FeatureMonitorCallback(
    layer_names=["backbone.layer2", "backbone.layer3", "backbone.layer4"],
    log_every_n_epochs=1,
    num_batches=10,
)

trainer = AutoTrainer(
    max_epochs=100,
    callbacks=[feature_callback],
)
```

## High-Level API

### Explain Single Prediction

```python
from autotimm.interpretation import explain_prediction

result = explain_prediction(
    model,
    image,
    method="gradcam",
    target_class=None,  # Auto-detect
    target_layer=None,  # Auto-detect
    colormap="viridis",
    alpha=0.4,
    save_path="explanation.png",
    return_heatmap=True,
)
```

### Compare Multiple Methods

```python
from autotimm.interpretation import compare_methods

results = compare_methods(
    model,
    image,
    methods=["gradcam", "gradcam++", "integrated_gradients"],
    save_path="comparison.png",
)
```

### Batch Visualization

```python
from autotimm.interpretation import visualize_batch

images = [load_image(f"test_{i}.jpg") for i in range(10)]

results = visualize_batch(
    model,
    images,
    method="gradcam",
    output_dir="explanations/",
)
```

## Advanced Usage

### Custom Target Layer

Specify which layer to use for interpretation:

```python
# By name
explainer = GradCAM(model, target_layer="backbone.layer3.2.conv2")

# By module reference
explainer = GradCAM(model, target_layer=model.backbone.layer3)
```

### Customize Visualization

```python
from autotimm.interpretation.visualization import overlay_heatmap, apply_colormap

# Apply custom colormap
colored_heatmap = apply_colormap(heatmap, colormap="hot")

# Create custom overlay
overlayed = overlay_heatmap(
    image,
    heatmap,
    alpha=0.5,
    colormap="plasma",
    resize_heatmap=True,
)
```

### Receptive Field Analysis

Understand what input regions affect specific features:

```python
viz = FeatureVisualizer(model)

# Visualize receptive field for a specific channel
sensitivity = viz.visualize_receptive_field(
    image,
    layer_name="backbone.layer3",
    channel=42,
    save_path="receptive_field.png"
)
```

## Best Practices

### 1. Choose the Right Method

- **GradCAM**: Fast, good for CNNs, class-discriminative
- **GradCAM++**: Better for multiple objects
- **Integrated Gradients**: Theoretical guarantees, pixel-level attributions
- **Attention Visualization**: For Vision Transformers

### 2. Validate Explanations

```python
# Use multiple methods to cross-validate
methods = ["gradcam", "gradcam++", "integrated_gradients"]
results = compare_methods(model, image, methods=methods)

# Check for consistency across methods
```

### 3. Monitor During Training

```python
# Track both interpretations and feature statistics
callbacks = [
    InterpretationCallback(sample_images, log_every_n_epochs=5),
    FeatureMonitorCallback(layer_names, log_every_n_epochs=1),
]
```

### 4. Production Deployment

```python
# For production, use efficient methods
# GradCAM is fast and suitable for real-time systems
explainer = GradCAM(model, use_cuda=True)

# Pre-compute for batch inference
heatmaps = explainer.explain_batch(images, batch_size=32)
```

## Performance Considerations

### GPU Acceleration

```python
# Enable CUDA for faster computation
explainer = GradCAM(model, use_cuda=True)
```

### Batch Processing

```python
# Process multiple images at once
heatmaps = explainer.explain_batch(images, batch_size=16)
```

### Memory Management

```python
# For large images, reduce resolution before interpretation
from torchvision import transforms

resize = transforms.Resize((224, 224))
small_image = resize(image)
heatmap = explainer.explain(small_image)
```

## Troubleshooting

For interpretation issues, see the [Troubleshooting - Interpretation](../../troubleshooting/task-specific/interpretation.md) including:

- No heatmap visible
- Poor localization
- Slow performance
- Method-specific issues

## Examples

See the complete examples:

- `examples/interpretation/interpretation_demo.py` - Basic interpretation methods
- `examples/interpretation/interpretation_phase2_demo.py` - Advanced methods and task-specific adapters
- `examples/interpretation/interpretation_phase3_demo.py` - Training integration and feature visualization
- `examples/interpretation/interpretation_metrics_demo.py` - Quantitative evaluation of explanation quality

## Command-Line Interpretation

Run interpretation methods directly from the command line without writing Python code:

```bash
python -m autotimm.cli.interpret_cli \
    --checkpoint path/to/checkpoint.ckpt \
    --image path/to/image.jpg \
    --methods gradcam,gradcampp,integrated_gradients,smoothgrad \
    --output-dir ./interpretations \
    --task-class ImageClassifier
```

This outputs JSON to stdout with heatmap file paths and the predicted class:

```json
{
  "results": {
    "gradcam": "./interpretations/gradcam.png",
    "gradcampp": "./interpretations/gradcampp.png",
    "integrated_gradients": "./interpretations/integrated_gradients.png",
    "smoothgrad": "./interpretations/smoothgrad.png"
  },
  "predicted_class": 5,
  "errors": {}
}
```

Attention methods (`attention_rollout`, `attention_flow`) are automatically skipped with descriptive messages for non-ViT models.

See [CLI API Reference](../../api/cli.md) for full argument details.

## API Reference

For detailed API documentation, see:

- [Interpretation Methods](methods.md)
- [Feature Visualization](feature-visualization.md)
- [Callbacks](callbacks.md)
- [Task-Specific Adapters](task-adapters.md)
- [Quality Metrics](metrics.md)
- [Interactive Visualizations](interactive-visualizations.md)
- [Performance Optimization](optimization.md)
