---
title: AutoTimm - Automated Deep Learning for Computer Vision
description: Train image classification, object detection, and segmentation models with 1000+ timm backbones. A flexible computer vision workflow built on PyTorch Lightning.
hide:
  - navigation
  - toc
---

<div class="at-home">

<section class="at-hero" aria-labelledby="hero-title">
<div class="at-hero-copy">

<img class="at-floating-logo" src="autotimm.png" alt="AutoTimm" width="1166" height="444" decoding="async">

<p class="at-eyebrow"><span class="at-status" aria-hidden="true"></span> OPEN SOURCE COMPUTER VISION</p>
<h1 id="hero-title">Your next vision model.<br><span>Less boilerplate.</span></h1>
<p class="at-lead">Go from dataset to trained model with a few lines of Python. Build on 1000+ backbones, with the flexibility of PyTorch and the structure of Lightning.</p>

<div class="at-actions">
<a class="at-button at-button-primary" href="getting-started/quickstart/">Start building <span aria-hidden="true">↗</span></a>
<a class="at-button at-button-secondary" href="https://github.com/theja-vanka/AutoTimm">View on GitHub <span aria-hidden="true">↗</span></a>
</div>

<div class="at-install">

```bash
pip install autotimm
```

</div>
<p class="at-install-note">Python 3.10+ <span aria-hidden="true">·</span> Apache 2.0 <span aria-hidden="true">·</span> <a href="getting-started/installation/">Installation guide</a></p>

</div>
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
</section>

<div class="at-foundation">
<p>BUILT ON THE TOOLS YOU KNOW</p>
<div><a href="https://pytorch.org/">PyTorch</a><span aria-hidden="true">/</span><a href="https://github.com/huggingface/pytorch-image-models">timm</a><span aria-hidden="true">/</span><a href="https://github.com/Lightning-AI/pytorch-lightning">Lightning</a><span aria-hidden="true">/</span><a href="user-guide/integration/huggingface-hub-integration/">Hugging Face</a></div>
</div>

<section class="at-section" aria-labelledby="tasks-title">
<div class="at-section-heading">
<div><p class="at-eyebrow">ONE LIBRARY. FOUR VISION TASKS.</p><h2 id="tasks-title">What will you build?</h2></div>
<p>Choose your task. Keep the same familiar training workflow.</p>
</div>
<div class="at-task-grid">
<a class="at-task" href="examples/tasks/classification/">
<div class="at-task-top"><svg viewBox="0 0 48 48" fill="none" aria-hidden="true"><rect x="8" y="8" width="32" height="32" rx="5"/><path d="m15 25 6 6 13-14"/></svg><span>01</span></div>
<h3>Image classification</h3><p>Turn images into labels with pretrained CNNs and vision transformers.</p><span class="at-card-link">Explore classification <span aria-hidden="true">↗</span></span>
</a>
<a class="at-task" href="examples/tasks/object-detection/">
<div class="at-task-top"><svg viewBox="0 0 48 48" fill="none" aria-hidden="true"><path d="M16 6H6v10m26-10h10v10M6 32v10h10m26-10v10H32"/><rect x="15" y="15" width="18" height="18" rx="2"/></svg><span>02</span></div>
<h3>Object detection</h3><p>Find and localize objects with FCOS and YOLOX detectors.</p><span class="at-card-link">Explore detection <span aria-hidden="true">↗</span></span>
</a>
<a class="at-task" href="examples/tasks/semantic-segmentation/">
<div class="at-task-top"><svg viewBox="0 0 48 48" fill="none" aria-hidden="true"><rect x="7" y="7" width="34" height="34" rx="4"/><path d="M7 27h12V16h12v25M19 27v14M31 24h10"/></svg><span>03</span></div>
<h3>Semantic segmentation</h3><p>Give every pixel a class with DeepLabV3+ and FCN architectures.</p><span class="at-card-link">Explore segmentation <span aria-hidden="true">↗</span></span>
</a>
<a class="at-task" href="examples/tasks/instance-segmentation/">
<div class="at-task-top"><svg viewBox="0 0 48 48" fill="none" aria-hidden="true"><rect x="6" y="6" width="23" height="23" rx="6"/><rect x="19" y="19" width="23" height="23" rx="6"/></svg><span>04</span></div>
<h3>Instance segmentation</h3><p>Separate individual objects with detection and per-instance masks.</p><span class="at-card-link">Explore instances <span aria-hidden="true">↗</span></span>
</a>
</div>
</section>

<section class="at-capabilities at-section" aria-labelledby="features-title">
<div class="at-capabilities-intro"><p class="at-eyebrow">LESS SETUP. MORE EXPERIMENTING.</p><h2 id="features-title">The building blocks.<br>Already connected.</h2><p>Spend your time on the model and the data. AutoTimm brings the training essentials together, while keeping you in control.</p><a class="at-text-link" href="user-guide/">Explore the user guide <span aria-hidden="true">→</span></a><div class="at-backbone-stat"><strong>1000<span>+</span></strong><span>backbones from timm</span></div><div class="at-model-tags"><span>ResNet</span><span>EfficientNet</span><span>ConvNeXt</span><span>ViT</span><span>Swin</span></div></div>
<div class="at-feature-grid">
<div class="at-feature"><span class="at-feature-number">01 / TRAIN</span><h3>Find your training rhythm</h3><p>Automatic learning rate and batch size finding, mixed precision, and distributed training through Lightning.</p><a href="user-guide/training/training/">Training guide <span aria-hidden="true">↗</span></a></div>
<div class="at-feature"><span class="at-feature-number">02 / MEASURE</span><h3>Make every run count</h3><p>Configure torchmetrics and track experiments with TensorBoard, MLflow, W&B, or CSV loggers.</p><a href="user-guide/guides/logging/">Logging guide <span aria-hidden="true">↗</span></a></div>
<div class="at-feature"><span class="at-feature-number">03 / UNDERSTAND</span><h3>Look inside your model</h3><p>Explore predictions with GradCAM, integrated gradients, and interactive visualizations.</p><a href="user-guide/interpretation/">Interpretation guide <span aria-hidden="true">↗</span></a></div>
<div class="at-feature"><span class="at-feature-number">04 / DEPLOY</span><h3>Take the next step</h3><p>Export trained models with TorchScript or ONNX and bring your work into inference workflows.</p><a href="user-guide/inference/model-export/">Export guide <span aria-hidden="true">↗</span></a></div>
</div>
</section>

<section class="at-bottom-cta" aria-labelledby="start-title">
<div><p class="at-eyebrow">FROM IDEA TO FIRST EXPERIMENT</p><h2 id="start-title">Let’s get your model training.</h2><p>Start with the quick start guide, or find an example for your task.</p></div>
<div class="at-actions"><a class="at-button at-button-primary" href="getting-started/quickstart/">Get started <span aria-hidden="true">↗</span></a><a class="at-text-link" href="examples/">Browse examples <span aria-hidden="true">→</span></a></div>
</section>

</div>
