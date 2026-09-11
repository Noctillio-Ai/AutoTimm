"""Demonstrate model inference and prediction.

This example demonstrates:
- Loading a trained model from checkpoint
- Running inference on single images
- Running batch predictions
- Exporting predictions to file

Usage:
    python examples/inference.py
"""

import torch
from PIL import Image
from torchvision import transforms

from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    MetricConfig,
)


def get_transforms(image_size: int = 224):
    """Get inference transforms."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def predict_single_image(
    model: ImageClassifier,
    image_path: str,
    class_names: list[str],
    image_size: int = 224,
) -> tuple[str, float, dict[str, float]]:
    """Predict class for a single image.

    Returns:
        Tuple of (predicted_class, confidence, all_probabilities)
    """
    model.eval()
    transform = get_transforms(image_size)

    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Move to same device as model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Get predictions
    with torch.inference_mode():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=-1)

    # Get top prediction
    confidence, pred_idx = probs.max(dim=-1)
    predicted_class = class_names[pred_idx.item()]

    # Get all probabilities
    all_probs = {name: probs[0, i].item() for i, name in enumerate(class_names)}

    return predicted_class, confidence.item(), all_probs


def predict_batch(
    model: ImageClassifier,
    dataloader,
) -> tuple[list[int], list[float]]:
    """Run batch prediction on a dataloader.

    Returns:
        Tuple of (predictions, probabilities)
    """
    model.eval()
    device = next(model.parameters()).device

    all_preds = []
    all_probs = []

    with torch.inference_mode():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=-1)

            confidences, preds = probs.max(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(confidences.cpu().tolist())

    return all_preds, all_probs


def main():
    # ========================================================================
    # Part 1: Train a model (or load from checkpoint)
    # ========================================================================
    print("=" * 60)
    print("Training a model...")
    print("=" * 60)

    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=64,
        num_workers=4,
    )
    data.setup("fit")

    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
    ]

    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        metrics=metrics,
        lr=1e-3,
        scheduler="cosine",
    )

    trainer = AutoTrainer(
        max_epochs=2,  # Short training for demo
        accelerator="auto",
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule=data)

    # ========================================================================
    # Part 2: Load from checkpoint (if you have a saved model)
    # ========================================================================
    # model = ImageClassifier.load_from_checkpoint(
    #     "path/to/checkpoint.ckpt",
    #     backbone="resnet18",
    #     num_classes=10,
    #     metrics=metrics,
    # )

    # ========================================================================
    # Part 3: Run inference
    # ========================================================================
    print("\n" + "=" * 60)
    print("Running inference...")
    print("=" * 60)

    # CIFAR-10 class names
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Option A: Using trainer.predict() with dataloader
    print("\nBatch prediction using trainer.predict():")
    data.setup("test")
    predictions = trainer.predict(model, datamodule=data)

    # predictions is a list of batches, each batch is a tensor of probabilities
    all_probs = torch.cat(predictions, dim=0)
    print(f"  Total samples: {len(all_probs)}")
    print(f"  Predictions shape: {all_probs.shape}")

    # Get top-1 predictions
    confidences, _pred_indices = all_probs.max(dim=-1)
    print(f"  Mean confidence: {confidences.mean():.4f}")

    # Option B: Manual batch prediction
    print("\nManual batch prediction:")
    preds, probs = predict_batch(model, data.test_dataloader())
    print(f"  Total predictions: {len(preds)}")
    print(f"  First 10 predictions: {preds[:10]}")
    print(f"  First 10 confidences: {[f'{p:.3f}' for p in probs[:10]]}")

    # Option C: Single image prediction (if you have an image file)
    # print("\nSingle image prediction:")
    # pred_class, conf, all_probs = predict_single_image(
    #     model, "path/to/image.jpg", class_names
    # )
    # print(f"  Predicted: {pred_class} (confidence: {conf:.4f})")

    # ========================================================================
    # Part 4: Export predictions
    # ========================================================================
    print("\n" + "=" * 60)
    print("Exporting predictions...")
    print("=" * 60)

    # Save predictions to CSV
    import csv

    output_file = "predictions.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "predicted_class", "predicted_idx", "confidence"])
        for i, (pred_idx, conf) in enumerate(zip(preds, probs)):
            writer.writerow([i, class_names[pred_idx], pred_idx, f"{conf:.4f}"])

    print(f"Predictions saved to: {output_file}")

    # Show sample predictions
    print("\nSample predictions:")
    for i in range(min(5, len(preds))):
        print(f"  Sample {i}: {class_names[preds[i]]} (confidence: {probs[i]:.4f})")


if __name__ == "__main__":
    main()
