---
title: AutoTimm - Automated Deep Learning for Computer Vision
description: Build computer vision models with less setup. Explore 1000+ timm backbones for classification, detection, and segmentation with PyTorch Lightning.
hide:
  - navigation
  - toc
---

<div class="at-home">
<section class="at-hero" aria-labelledby="hero-title">
<img class="at-floating-logo" src="autotimm.png" alt="AutoTimm" width="1166" height="444" decoding="async">
<h1 id="hero-title">Build vision models.<br><span>Keep your focus.</span></h1>
<p class="at-lead">From your first dataset to your next breakthrough.<br> Classification, detection, and segmentation with the tools you already know.</p>
<div class="at-actions"><a class="at-button at-primary" href="getting-started/quickstart/">Get started <span aria-hidden="true">→</span></a><a class="at-button at-secondary" href="https://github.com/theja-vanka/AutoTimm">View on GitHub <span aria-hidden="true">↗</span></a></div>
<div class="at-install no-select">

```bash
pip install autotimm
```

</div>
<p class="at-compatibility">Python 3.10+ <span aria-hidden="true">·</span> Open source <span aria-hidden="true">·</span> Apache 2.0</p>
</section>

<section class="at-task-section" aria-labelledby="tasks-title">
<div class="at-section-heading"><h2 id="tasks-title">One workflow. Every vision task.</h2><p>Choose what you want to build.</p></div>
<div class="at-task-picker">
<fieldset class="at-options"><legend class="at-sr-only">Choose a vision task</legend>
<label><input type="radio" name="at-task" id="at-classification" checked><span>Classification</span></label>
<label><input type="radio" name="at-task" id="at-detection"><span>Object detection</span></label>
<label><input type="radio" name="at-task" id="at-semantic"><span>Semantic segmentation</span></label>
<label><input type="radio" name="at-task" id="at-instance"><span>Instance segmentation</span></label>
</fieldset>
<div class="at-task-panels">
<section class="at-task-panel" id="at-panel-classification" aria-labelledby="classification-title"><div class="at-task-description"><span class="at-task-caption">Understand the whole image</span><h3 id="classification-title">An image in.<br>A label out.</h3><p>Build an image classifier with a pretrained CNN or vision transformer. Choose your backbone, define your metrics, and start training.</p><a href="examples/tasks/classification/">Explore classification <span aria-hidden="true">→</span></a></div><div class="at-pipeline" aria-label="Image classification workflow"><div><span>DATA</span><code>ImageDataModule</code><small>Your images, ready for training</small></div><span class="at-pipeline-arrow" aria-hidden="true">↓</span><div class="at-pipeline-model"><span>MODEL</span><code>ImageClassifier</code><small>ResNet · EfficientNet · ViT</small></div><span class="at-pipeline-arrow" aria-hidden="true">↓</span><div><span>TRAIN</span><code>AutoTrainer</code><small>Powered by PyTorch Lightning</small></div></div></section>
<section class="at-task-panel" id="at-panel-detection" aria-labelledby="detection-title"><div class="at-task-description"><span class="at-task-caption">Find what matters</span><h3 id="detection-title">Know what’s there.<br>And where it is.</h3><p>Locate objects with bounding boxes using FCOS or YOLOX. Load COCO-format data and track detection metrics in the same training workflow.</p><a href="examples/tasks/object-detection/">Explore object detection <span aria-hidden="true">→</span></a></div><div class="at-pipeline" aria-label="Object detection workflow"><div><span>DATA</span><code>DetectionDataModule</code><small>Images and bounding boxes</small></div><span class="at-pipeline-arrow" aria-hidden="true">↓</span><div class="at-pipeline-model"><span>MODEL</span><code>ObjectDetector</code><small>FCOS with your choice of backbone</small></div><span class="at-pipeline-arrow" aria-hidden="true">↓</span><div><span>TRAIN</span><code>AutoTrainer</code><small>Powered by PyTorch Lightning</small></div></div></section>
<section class="at-task-panel" id="at-panel-semantic" aria-labelledby="semantic-title"><div class="at-task-description"><span class="at-task-caption">See the complete picture</span><h3 id="semantic-title">Give every pixel<br>a purpose.</h3><p>Assign a class to every pixel with DeepLabV3+ or FCN. Configure segmentation losses and metrics to fit your dataset.</p><a href="examples/tasks/semantic-segmentation/">Explore semantic segmentation <span aria-hidden="true">→</span></a></div><div class="at-pipeline" aria-label="Semantic segmentation workflow"><div><span>DATA</span><code>SegmentationDataModule</code><small>Images and semantic masks</small></div><span class="at-pipeline-arrow" aria-hidden="true">↓</span><div class="at-pipeline-model"><span>MODEL</span><code>SemanticSegmentor</code><small>DeepLabV3+ · FCN</small></div><span class="at-pipeline-arrow" aria-hidden="true">↓</span><div><span>TRAIN</span><code>AutoTrainer</code><small>Powered by PyTorch Lightning</small></div></div></section>
<section class="at-task-panel" id="at-panel-instance" aria-labelledby="instance-title"><div class="at-task-description"><span class="at-task-caption">Separate every instance</span><h3 id="instance-title">Every object.<br>Its own outline.</h3><p>Combine object detection with per-instance masks. Distinguish individual objects, even when they belong to the same class.</p><a href="examples/tasks/instance-segmentation/">Explore instance segmentation <span aria-hidden="true">→</span></a></div><div class="at-pipeline" aria-label="Instance segmentation workflow"><div><span>DATA</span><code>InstanceSegmentationDataModule</code><small>Images and per-instance masks</small></div><span class="at-pipeline-arrow" aria-hidden="true">↓</span><div class="at-pipeline-model"><span>MODEL</span><code>InstanceSegmentor</code><small>Detection with a mask head</small></div><span class="at-pipeline-arrow" aria-hidden="true">↓</span><div><span>TRAIN</span><code>AutoTrainer</code><small>Powered by PyTorch Lightning</small></div></div></section>
</div>
<div class="at-workspace-footer"><span><strong>1000+</strong> timm backbones</span><span><strong>4</strong> vision tasks</span><span><strong>One</strong> training interface</span></div>
</div>
</section>

<section class="at-example" aria-labelledby="example-title">
<div class="at-section-heading"><h2 id="example-title">Less setup. More experimenting.</h2><p>A complete classification example, from data to training.</p></div>
<div class="at-code-panel">

```python
import autotimm as at

# Your data. Your choice of backbone.
data = at.ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    num_workers=0,
)

metrics = [at.MetricConfig(
    name="accuracy",
    backend="torchmetrics",
    metric_class="Accuracy",
    params={"task": "multiclass"},
    stages=["train", "val"],
)]

model = at.ImageClassifier(
    backbone="resnet18",
    num_classes=10,
    metrics=metrics,
)

trainer = at.AutoTrainer(max_epochs=10)
trainer.fit(model, datamodule=data)
```

</div>
<div class="at-example-footer"><p>Start with ResNet. Switch to ConvNeXt, EfficientNet, ViT, or Swin.</p><a href="getting-started/quickstart/">Follow the quick start <span aria-hidden="true">→</span></a></div>
</section>

<section class="at-resources" aria-labelledby="resources-title"><div class="at-section-heading"><h2 id="resources-title">Room to go further.</h2><p>The essentials are connected. The choices stay yours.</p></div>
<div class="at-resource-links"><a href="user-guide/training/training/"><strong>Refine your training</strong><span>Automatic learning rate and batch size finding, mixed precision, and distributed training.</span><span aria-hidden="true">↗</span></a><a href="user-guide/guides/logging/"><strong>Track every experiment</strong><span>Configurable metrics with TensorBoard, MLflow, W&B, and CSV logging.</span><span aria-hidden="true">↗</span></a><a href="user-guide/interpretation/"><strong>Understand predictions</strong><span>GradCAM, integrated gradients, and interactive visualizations.</span><span aria-hidden="true">↗</span></a><a href="user-guide/inference/model-export/"><strong>Put your model to work</strong><span>Export to TorchScript or ONNX and build your inference workflow.</span><span aria-hidden="true">↗</span></a></div>
</section>

<div class="at-foundation"><p>Built on a familiar foundation</p><div><a href="https://pytorch.org/">PyTorch</a><a href="https://github.com/huggingface/pytorch-image-models">timm</a><a href="https://github.com/Lightning-AI/pytorch-lightning">Lightning</a><a href="user-guide/integration/huggingface-hub-integration/">Hugging Face</a></div></div>
<section class="at-finale" aria-labelledby="start-title"><h2 id="start-title">Your next model starts here.</h2><div class="at-actions"><a class="at-button at-primary" href="getting-started/quickstart/">Get started <span aria-hidden="true">→</span></a><a class="at-button at-secondary" href="examples/">Browse examples <span aria-hidden="true">↗</span></a></div></section>
</div>
