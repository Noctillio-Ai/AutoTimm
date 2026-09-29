---
title: Examples
description: Runnable code examples for AutoTimm — image classification, object detection, segmentation, HuggingFace integration, CLI usage, and more.
---

# Examples

The [`examples/`](https://github.com/theja-vanka/AutoTimm/tree/main/examples) directory contains runnable scripts demonstrating AutoTimm features.

## Examples Organization

```mermaid
graph LR
    A[<b>AutoTimm<br/>Examples</b>] --> B[<b>Getting Started</b><br/>Classification,<br/>data loading]
    A --> C[<b>Computer Vision</b><br/>Detection,<br/>segmentation]
    A --> D[<b>HuggingFace</b><br/>Hub integration,<br/>model sharing]
    A --> E[<b>Training</b><br/>HPO, multi-GPU,<br/>auto-tuning]
    A --> F[<b>Understanding</b><br/>GradCAM, attention,<br/>feature maps]
    A --> G[<b>CLI</b><br/>YAML configs,<br/>command line]

    style A fill:#1565C0,stroke:#0D47A1
    style B fill:#1976D2,stroke:#1565C0
    style C fill:#1565C0,stroke:#0D47A1
    style D fill:#1976D2,stroke:#1565C0
    style E fill:#1565C0,stroke:#0D47A1
    style F fill:#1976D2,stroke:#1565C0
    style G fill:#1565C0,stroke:#0D47A1
```

## Quick Reference

### :material-rocket-launch: Getting Started
- **[Classification](tasks/classification.md)** - Start with CIFAR-10 and custom datasets
- **[HuggingFace Hub Basics](integration/huggingface-hub.md)** - Load and use HF Hub models

### :material-bullseye: Computer Vision Tasks
- **[Object Detection](tasks/object-detection.md)** - FCOS, YOLOX, RT-DETR, and transformers
- **[Semantic Segmentation](tasks/semantic-segmentation.md)** - DeepLabV3+ and FCN
- **[Instance Segmentation](tasks/instance-segmentation.md)** - Mask R-CNN style

### :simple-huggingface: HuggingFace Advanced
- **[Model Interpretation](integration/hf_interpretation.md)** - GradCAM, attention visualization, metrics
- **[Transfer Learning](integration/hf_transfer_learning.md)** - LLRD, progressive unfreezing
- **[Ensemble & Distillation](integration/hf_ensemble.md)** - Model ensembles and knowledge distillation
- **[Deployment](integration/hf_deployment.md)** - ONNX, quantization, serving
- **[Custom Data](utilities/hf_custom_data.md)** - Advanced augmentation and data handling
- **[Hyperparameter Tuning](utilities/hf_hyperparameter_tuning.md)** - Optuna integration

### :material-chart-bar: Training & Optimization
- **[Training Utilities](utilities/training-utilities.md)** - Auto-tuning, multi-GPU, presets
- **[Data Handling](utilities/data-handling.md)** - Balanced sampling, augmentation
- **[Logging & Metrics](utilities/logging-metrics.md)** - TensorBoard, MLflow, evaluation

### :material-console: CLI
- **[CLI Examples](utilities/cli.md)** - Train from YAML configs on the command line

### :material-magnify: Model Understanding
- **[Interpretation Methods](utilities/interpretation.md)** - Comprehensive interpretation toolkit
- **[Backbone Utilities](utilities/backbone-utilities.md)** - Discover and compare models

## All Examples by Category

### :material-rocket-launch: Getting Started

| Script | Description |
|--------|-------------|
| [`getting_started/classify_cifar10.py`](tasks/classification.md#cifar-10-classification) | ResNet-18 on CIFAR-10 with MetricManager |
| [`getting_started/classify_custom_folder.py`](tasks/classification.md#custom-folder-dataset) | Train on custom ImageFolder dataset with W&B |
| [`getting_started/vit_finetuning.py`](tasks/classification.md#vit-fine-tuning) | Two-phase ViT fine-tuning |

### :material-bullseye: Computer Vision Tasks

**Object Detection:**

| Script | Description |
|--------|-------------|
| [`computer_vision/object_detection_coco.py`](tasks/object-detection.md#object-detection-on-coco) | FCOS-style object detection on COCO |
| [`computer_vision/object_detection_yolox.py`](tasks/object-detection.md#yolox-object-detection) | YOLOX object detection |
| [`computer_vision/object_detection_rtdetr.py`](tasks/object-detection.md#rt-detr-real-time-detection-transformer) | RT-DETR transformer detection |
| [`computer_vision/object_detection_transformers.py`](tasks/object-detection.md#transformer-based-object-detection) | Vision Transformers for detection |
| [`computer_vision/explore_yolox_models.py`](tasks/object-detection.md#explore-yolox-models) | Explore YOLOX model variants |
| [`computer_vision/yolox_official.py`](tasks/object-detection.md#yolox-official) | Official YOLOX implementation |

**Segmentation:**

| Script | Description |
|--------|-------------|
| [`computer_vision/semantic_segmentation.py`](tasks/semantic-segmentation.md#basic-example-cityscapes) | DeepLabV3+ semantic segmentation |
| [`computer_vision/instance_segmentation.py`](tasks/instance-segmentation.md#basic-example-coco) | Mask R-CNN style instance segmentation |

### :simple-huggingface: HuggingFace Hub Integration

**Basic Integration:**

| Script | Description |
|--------|-------------|
| [`huggingface/huggingface_hub_models.py`](integration/huggingface-hub.md#1-introduction-to-hf-hub-models) | Introduction to HF Hub |
| [`huggingface/hf_hub_classification.py`](integration/huggingface-hub.md#2-image-classification) | Classification with HF Hub |
| [`huggingface/hf_hub_segmentation.py`](integration/huggingface-hub.md#3-semantic-segmentation) | Segmentation with HF Hub |
| [`huggingface/hf_hub_object_detection.py`](integration/huggingface-hub.md#4-object-detection) | Detection with HF Hub |
| [`huggingface/hf_hub_instance_segmentation.py`](integration/huggingface-hub.md#5-instance-segmentation) | Instance segmentation with HF Hub |
| [`huggingface/hf_hub_advanced.py`](integration/huggingface-hub.md#6-advanced-usage) | Advanced HF Hub features |
| [`huggingface/hf_hub_lightning_integration.py`](../user-guide/integration/huggingface-hub-integration.md#pytorch-lightning-compatibility) | Lightning compatibility |
| [`huggingface/hf_direct_models_lightning.py`](../user-guide/integration/huggingface-transformers-integration.md) | Direct HF Transformers models |

**Advanced HF Hub:**

| Script | Description |
|--------|-------------|
| [`huggingface/hf_interpretation.py`](integration/hf_interpretation.md) | Model interpretation (GradCAM, attention, metrics) |
| [`huggingface/hf_transfer_learning.py`](integration/hf_transfer_learning.md) | LLRD, progressive unfreezing |
| [`huggingface/hf_ensemble.py`](integration/hf_ensemble.md) | Ensembles and knowledge distillation |
| [`huggingface/hf_deployment.py`](integration/hf_deployment.md) | ONNX export, quantization, serving |

### :material-chart-bar: Data & Augmentation

| Script | Description |
|--------|-------------|
| [`data_training/balanced_sampling.py`](utilities/data-handling.md#balanced-sampling) | Weighted sampling for imbalanced data |
| [`data_training/albumentations_cifar10.py`](utilities/data-handling.md#albumentations-strong-augmentation) | Albumentations strong augmentation |
| [`data_training/albumentations_custom_folder.py`](utilities/data-handling.md#custom-albumentations-pipeline) | Custom albumentations pipeline |
| [`data_training/multilabel_classification.py`](tasks/classification.md#multi-label-classification) | Multi-label classification with CSV data |
| [`data_training/csv_classification.py`](utilities/csv-data-loading.md#csv-classification) | Classification from CSV files |
| [`data_training/csv_detection.py`](utilities/csv-data-loading.md#csv-object-detection) | Object detection from CSV annotations |
| [`data_training/csv_segmentation.py`](utilities/csv-data-loading.md#csv-semantic-segmentation) | Semantic segmentation from CSV |
| [`data_training/csv_instance_segmentation.py`](utilities/csv-data-loading.md#csv-instance-segmentation) | Instance segmentation from CSV |
| [`data_training/hf_custom_data.py`](utilities/hf_custom_data.md) | Advanced augmentation, multi-label, validation |

### :material-cog: Training & Optimization

| Script | Description |
|--------|-------------|
| [`data_training/auto_tuning.py`](utilities/training-utilities.md#auto-tuning) | Automatic LR and batch size finding |
| [`data_training/multi_gpu_training.py`](utilities/training-utilities.md#multi-gpu-training) | Multi-GPU and distributed training |
| [`data_training/preset_manager.py`](utilities/training-utilities.md#transform-backend-recommendation) | Transform backend recommendation and presets |
| [`data_training/performance_optimization_demo.py`](utilities/training-utilities.md#performance-optimization) | Performance optimization techniques |
| [`data_training/hf_hyperparameter_tuning.py`](utilities/hf_hyperparameter_tuning.md) | Optuna hyperparameter optimization |

### :material-magnify: Model Interpretation

| Script | Description |
|--------|-------------|
| [`interpretation/interpretation_demo.py`](utilities/interpretation.md#basic-interpretation) | Basic interpretation and visualization |
| [`interpretation/interpretation_metrics_demo.py`](utilities/interpretation.md#interpretation-metrics) | Interpretation with metrics analysis |
| [`interpretation/interpretation_phase2_demo.py`](utilities/interpretation.md#interpretation-phase-2) | Advanced techniques (Phase 2) |
| [`interpretation/interpretation_phase3_demo.py`](utilities/interpretation.md#interpretation-phase-3) | Advanced techniques (Phase 3) |
| [`interpretation/interactive_visualization_demo.py`](utilities/interpretation.md#interactive-visualization) | Interactive Plotly visualizations |
| [`interpretation/comprehensive_interpretation_tutorial.ipynb`](utilities/interpretation.md#comprehensive-tutorial) | Comprehensive tutorial (Notebook) |

### :material-trending-up: Logging & Experiment Tracking

| Script | Description |
|--------|-------------|
| [`logging_inference/multiple_loggers.py`](utilities/logging-metrics.md#multiple-loggers) | TensorBoard + CSV logging |
| [`logging_inference/mlflow_tracking.py`](utilities/logging-metrics.md#mlflow-tracking) | MLflow experiment tracking |
| [`logging_inference/detailed_evaluation.py`](utilities/logging-metrics.md#detailed-evaluation) | Confusion matrix and per-class metrics |

### :material-console: CLI

| Config | Description |
|--------|-------------|
| [`cli/classification.yaml`](utilities/cli.md#classification) | CLI config for image classification |
| [`cli/detection.yaml`](utilities/cli.md#object-detection) | CLI config for object detection |
| [`cli/segmentation.yaml`](utilities/cli.md#semantic-segmentation) | CLI config for semantic segmentation |

### :material-cube-send: Deployment & Export

| Script | Description |
|--------|-------------|
| [`deployment/export_to_onnx.py`](../user-guide/deployment/onnx-export.md) | Export models to ONNX for cross-platform deployment |
| [`deployment/export_to_torchscript.py`](../user-guide/deployment/torchscript-export.md) | Export models to TorchScript for production |
| [`deployment/deploy_torchscript_cpp.py`](../user-guide/deployment/cpp-deployment.md) | C++ and mobile deployment |

### :material-play-circle: Inference & Utilities

| Script | Description |
|--------|-------------|
| [`logging_inference/inference.py`](../user-guide/inference/classification-inference.md) | Model inference and batch prediction |
| [`logging_inference/inference_without_metrics.py`](../user-guide/inference/classification-inference.md) | Inference without metrics |
| [`logging_inference/multilabel_inference.py`](../user-guide/inference/classification-inference.md#multi-label-inference) | Multi-label inference and per-label probabilities |
| [`logging_inference/detection_inference.py`](../user-guide/inference/object-detection-inference.md) | Object detection inference |
| [`logging_inference/segmentation_inference.py`](../user-guide/inference/semantic-segmentation-inference.md) | Segmentation inference |
| [`logging_inference/backbone_discovery.py`](utilities/backbone-utilities.md#backbone-discovery) | Explore timm backbones |

---

## Getting Started

To run any example:

```bash
# Clone the repository
git clone https://github.com/theja-vanka/AutoTimm.git
cd AutoTimm

# Install AutoTimm with all dependencies
pip install -e ".[all]"

# Run an example
python examples/getting_started/classify_cifar10.py
```

## Example Categories Summary

### :material-rocket-launch: Getting Started (3 examples)
Start your journey with CIFAR-10 classification and custom datasets.

### :material-bullseye: Computer Vision Tasks (8 examples)
- **Classification** - Basic to advanced classification
- **Object Detection** - FCOS, YOLOX, RT-DETR, transformers (6 examples)
- **Segmentation** - Semantic and instance segmentation (2 examples)

### :simple-huggingface: HuggingFace Hub (12 examples)
- **Basic Integration** - 8 task-specific examples
- **Advanced Techniques** - Interpretation, transfer learning, ensemble, deployment (4 examples)

### :material-chart-bar: Data & Training (14 examples)
- **Data & Augmentation** - Balanced sampling, augmentation, multi-label, CSV loading (9 examples)
- **Training & Optimization** - Auto-tuning, multi-GPU, HPO (5 examples)

### :material-console: CLI (3 configs)
Train from the command line with YAML configuration files.

### :material-magnify: Understanding & Deployment (13 examples)
- **Model Interpretation** - GradCAM, attention, interactive viz (6 examples)
- **Logging & Tracking** - TensorBoard, MLflow, evaluation (3 examples)
- **Inference & Utilities** - Model inference, multi-label inference, and backbone discovery (4 examples)

**Total: 50+ runnable examples and configs**



