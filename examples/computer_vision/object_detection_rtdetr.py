"""RT-DETR (Real-Time DEtection TRansformer) for object detection.

This example demonstrates:
- Using RT-DETR with AutoTimm's data loading capabilities
- End-to-end transformer-based detection (no NMS needed)
- Integration with Hugging Face transformers library
- Real-time performance with transformer architecture

RT-DETR Architecture:
- Efficient hybrid encoder (CNN + Transformer)
- Transformer decoder with query-based detection
- No anchor boxes or NMS required
- Fast inference with high accuracy

Note: This example requires the transformers library:
    pip install transformers
"""

import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from autotimm import AutoTrainer, DetectionDataModule, LoggerConfig


class RTDetrModule(torch.nn.Module):
    """Wrapper for RT-DETR model to work with PyTorch Lightning."""

    def __init__(
        self,
        model_name: str = "PekingU/rtdetr_r50vd",
        num_classes: int = 80,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load RT-DETR model
        self.model = RTDetrForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        self.processor = RTDetrImageProcessor.from_pretrained(model_name)

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        # Convert targets to RT-DETR format
        labels = self._convert_targets(targets, images.shape[0])

        # Forward pass
        outputs = self(pixel_values=images, labels=labels)

        # RT-DETR returns combined loss
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        labels = self._convert_targets(targets, images.shape[0])

        outputs = self(pixel_values=images, labels=labels)
        loss = outputs.loss
        self.log("val/loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def _convert_targets(self, targets, batch_size):
        """Convert COCO-format targets to RT-DETR format."""
        labels = []
        for i in range(batch_size):
            # Extract boxes and labels for this image
            mask = targets["image_id"] == i
            boxes = targets["boxes"][mask]
            class_labels = targets["labels"][mask]

            labels.append(
                {
                    "class_labels": class_labels,
                    "boxes": boxes,  # RT-DETR expects normalized xyxy format
                }
            )
        return labels


def main():
    """
    RT-DETR Detection Example

    RT-DETR is a real-time end-to-end transformer detector that eliminates
    the need for hand-crafted components like NMS. It achieves real-time
    performance while maintaining high accuracy.
    """

    # ========================================================================
    # Data - Use AutoTimm's DetectionDataModule
    # ========================================================================
    print("=" * 60)
    print("Loading COCO Dataset")
    print("=" * 60)

    data = DetectionDataModule(
        data_dir="./coco",
        image_size=640,
        batch_size=4,  # RT-DETR can use larger images, so smaller batch
        num_workers=4,
        augmentation_preset="default",
    )

    # ========================================================================
    # Option 1: RT-DETR with ResNet-50 backbone (balanced)
    # ========================================================================
    print("\n" + "=" * 60)
    print("Option 1: RT-DETR with ResNet-50 Backbone")
    print("=" * 60)

    model_r50 = RTDetrModule(
        model_name="PekingU/rtdetr_r50vd",
        num_classes=80,
        lr=1e-4,
        weight_decay=1e-4,
    )

    # Model characteristics:
    # - Balanced speed and accuracy
    # - 50M parameters
    # - Good for general use

    # ========================================================================
    # Option 2: RT-DETR with ResNet-101 backbone (higher accuracy)
    # ========================================================================
    print("\n" + "=" * 60)
    print("Option 2: RT-DETR with ResNet-101 Backbone")
    print("=" * 60)

    model_r101 = RTDetrModule(
        model_name="PekingU/rtdetr_r101vd",
        num_classes=80,
        lr=1e-4,
        weight_decay=1e-4,
    )

    # Model characteristics:
    # - Higher accuracy than R50
    # - 76M parameters
    # - Slightly slower but still real-time

    # ========================================================================
    # Option 3: Training RT-DETR
    # ========================================================================
    print("\n" + "=" * 60)
    print("Training RT-DETR Model")
    print("=" * 60)

    # Create model
    model = RTDetrModule(
        model_name="PekingU/rtdetr_r50vd",
        num_classes=80,
        lr=1e-4,
        weight_decay=1e-4,
    )

    # Configure trainer
    trainer = AutoTrainer(
        max_epochs=12,
        logger=[
            LoggerConfig(backend="tensorboard", params={"save_dir": "logs/rtdetr"}),
        ],
        gradient_clip_val=0.1,  # RT-DETR benefits from gradient clipping
        accelerator="auto",
        precision="bf16-mixed",  # Use mixed precision for speed
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, datamodule=data)

    # Test
    print("\nEvaluating on test set...")
    trainer.test(model, datamodule=data)

    # ========================================================================
    # Inference Example
    # ========================================================================
    print("\n" + "=" * 60)
    print("RT-DETR Inference")
    print("=" * 60)

    # Load trained model
    model.eval()

    # Example inference
    from PIL import Image

    # Load and process image
    image = Image.open("test_image.jpg").convert("RGB")
    processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
    inputs = processor(images=image, return_tensors="pt")

    # Inference (no NMS needed!)
    with torch.inference_mode():
        outputs = model.model(**inputs)

    # Post-process results
    results = processor.post_process_object_detection(
        outputs,
        threshold=0.3,  # Confidence threshold
        target_sizes=torch.tensor([image.size[::-1]]),
    )

    # Print detections
    for score, label, box in zip(
        results[0]["scores"],
        results[0]["labels"],
        results[0]["boxes"],
    ):
        print(f"Detected {label.item()} with confidence {score.item():.3f} at {box}")

    # ========================================================================
    # RT-DETR vs FCOS Comparison
    # ========================================================================
    print("\n" + "=" * 60)
    print("RT-DETR vs FCOS Architecture Comparison")
    print("=" * 60)
    print("""
┌─────────────────────┬──────────────────────┬──────────────────────┐
│ Feature             │ RT-DETR              │ FCOS (AutoTimm)      │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Architecture        │ Transformer-based    │ CNN-based            │
│ Detection paradigm  │ Query-based          │ Anchor-free points   │
│ NMS required        │ No                   │ Yes                  │
│ Training            │ End-to-end           │ End-to-end           │
│ Inference speed     │ Real-time            │ Real-time            │
│ Accuracy            │ High                 │ High                 │
│ Memory usage        │ Higher               │ Lower                │
│ Small objects       │ Good                 │ Excellent            │
│ Large objects       │ Excellent            │ Good                 │
└─────────────────────┴──────────────────────┴──────────────────────┘

When to use RT-DETR:
✓ Need end-to-end differentiable pipeline
✓ Want to avoid NMS post-processing
✓ Detecting large objects or scenes
✓ Have sufficient GPU memory
✓ Want transformer-based architecture

When to use FCOS:
✓ Need maximum efficiency
✓ Detecting small objects
✓ Limited GPU memory
✓ Want CNN-based architecture
✓ Need multi-scale detection (P3-P7)

Both architectures:
- Achieve real-time performance
- Support any image size
- Work with COCO format datasets
- Can use pretrained backbones
""")

    # ========================================================================
    # Available RT-DETR Models
    # ========================================================================
    print("\n" + "=" * 60)
    print("Available RT-DETR Pretrained Models")
    print("=" * 60)
    print("""
From Hugging Face (PekingU):
- PekingU/rtdetr_r18vd  - Smallest, fastest (20M params)
- PekingU/rtdetr_r34vd  - Good balance (31M params)
- PekingU/rtdetr_r50vd  - Recommended (42M params) ⭐
- PekingU/rtdetr_r101vd - Highest accuracy (76M params)

All models are pretrained on COCO and ready for fine-tuning.

Usage:
    model = RTDetrModule(
        model_name="PekingU/rtdetr_r50vd",
        num_classes=80,  # or your custom number of classes
    )
""")


if __name__ == "__main__":
    main()
