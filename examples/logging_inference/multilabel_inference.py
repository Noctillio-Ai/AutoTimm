"""Demonstrate multi-label classification inference.

This example demonstrates:
- Loading a multi-label model from checkpoint
- Running inference on single images with per-label probabilities
- Running batch predictions with thresholding
- Exporting multi-label predictions to CSV

Usage:
    python examples/logging_inference/multilabel_inference.py
"""

from __future__ import annotations

import csv
import os
import tempfile

import torch
from PIL import Image
from torchvision import transforms

from autotimm import (
    AutoTrainer,
    ImageClassifier,
    MetricConfig,
    MultiLabelImageDataModule,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    label_names: list[str],
    threshold: float = 0.5,
    image_size: int = 224,
) -> dict:
    """Predict labels for a single image.

    Returns a dict with:
        - probabilities: per-label sigmoid probabilities
        - predicted_labels: labels above the threshold
        - binary: binary prediction vector
    """
    model.eval()
    transform = get_transforms(image_size)

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    with torch.inference_mode():
        logits = model(input_tensor)
        probs = logits.sigmoid().squeeze(0)  # (num_labels,)

    binary = (probs > threshold).int()
    predicted = [name for name, b in zip(label_names, binary) if b]

    return {
        "probabilities": {n: p.item() for n, p in zip(label_names, probs)},
        "predicted_labels": predicted,
        "binary": binary.cpu().tolist(),
    }


def predict_batch(
    model: ImageClassifier,
    dataloader,
    threshold: float = 0.5,
) -> tuple[list[list[int]], list[list[float]]]:
    """Run batch prediction on a dataloader.

    Returns:
        Tuple of (binary_predictions, probabilities) per sample.
    """
    model.eval()
    device = next(model.parameters()).device

    all_preds: list[list[int]] = []
    all_probs: list[list[float]] = []

    with torch.inference_mode():
        for batch in dataloader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device)

            logits = model(images)
            probs = logits.sigmoid()
            preds = (probs > threshold).int()

            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return all_preds, all_probs


# ---------------------------------------------------------------------------
# Demo dataset (same helper as multilabel_classification.py)
# ---------------------------------------------------------------------------


def create_demo_dataset(root: str, num_images: int = 100):
    """Create a tiny synthetic multi-label dataset."""
    img_dir = os.path.join(root, "images")
    os.makedirs(img_dir, exist_ok=True)
    label_names = ["cat", "dog", "outdoor", "indoor"]

    for split, n in [("train", num_images), ("val", num_images // 5)]:
        csv_path = os.path.join(root, f"{split}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path"] + label_names)
            for i in range(n):
                fname = f"{split}_{i:04d}.jpg"
                Image.new("RGB", (64, 64), color=(i * 17 % 256, 100, 150)).save(
                    os.path.join(img_dir, fname)
                )
                labels = [(i + j) % 2 for j in range(len(label_names))]
                writer.writerow([fname] + labels)

    return (
        os.path.join(root, "train.csv"),
        os.path.join(root, "val.csv"),
        img_dir,
        label_names,
    )


def main():
    # ==================================================================
    # Part 1: Train a multi-label model (or load from checkpoint)
    # ==================================================================
    print("=" * 60)
    print("Training a multi-label model...")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    train_csv, val_csv, image_dir, label_names = create_demo_dataset(tmpdir)
    num_labels = len(label_names)

    data = MultiLabelImageDataModule(
        train_csv=train_csv,
        val_csv=val_csv,
        image_dir=image_dir,
        image_size=224,
        batch_size=16,
        num_workers=0,
    )

    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="MultilabelAccuracy",
            params={"num_labels": num_labels},
            stages=["train", "val"],
            prog_bar=True,
        ),
    ]

    model = ImageClassifier(
        backbone="resnet18",
        num_classes=num_labels,
        multi_label=True,
        threshold=0.5,
        metrics=metrics,
        lr=1e-3,
        compile_model=False,  # Disable for demo compatibility
    )

    trainer = AutoTrainer(
        max_epochs=2,  # Short training for demo
        accelerator="auto",
        logger=False,
        enable_checkpointing=False,
        tuner_config=False,
    )
    trainer.fit(model, datamodule=data)

    # To load from checkpoint instead:
    # model = ImageClassifier.load_from_checkpoint(
    #     "path/to/checkpoint.ckpt",
    #     backbone="resnet18",
    #     num_classes=4,
    #     multi_label=True,
    #     threshold=0.5,
    #     metrics=metrics,
    # )

    # ==================================================================
    # Part 2: Inference
    # ==================================================================
    print("\n" + "=" * 60)
    print("Running multi-label inference...")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- Option A: trainer.predict() ---
    print("\nBatch prediction using trainer.predict():")
    data.setup("fit")
    predictions = trainer.predict(model, dataloaders=data.val_dataloader())

    # Each batch is a tensor of sigmoid probabilities (N, num_labels)
    all_probs = torch.cat(predictions, dim=0)
    print(f"  Total samples: {len(all_probs)}")
    print(f"  Predictions shape: {all_probs.shape}")

    all_binary = (all_probs > 0.5).int()
    avg_labels = all_binary.float().sum(dim=1).mean()
    print(f"  Avg labels per sample: {avg_labels:.2f}")

    # --- Option B: Manual batch prediction ---
    print("\nManual batch prediction:")
    preds, probs = predict_batch(model, data.val_dataloader(), threshold=0.5)
    print(f"  Total predictions: {len(preds)}")
    print(f"  First sample binary: {preds[0]}")
    print(f"  First sample probs:  {[f'{p:.3f}' for p in probs[0]]}")

    # --- Option C: Single image prediction ---
    sample_image = os.path.join(image_dir, "val_0000.jpg")
    print(f"\nSingle image prediction ({sample_image}):")
    result = predict_single_image(model, sample_image, label_names, threshold=0.5)
    for name, prob in result["probabilities"].items():
        marker = "*" if name in result["predicted_labels"] else " "
        print(f"  [{marker}] {name}: {prob:.4f}")
    print(f"  Predicted labels: {result['predicted_labels']}")

    # ==================================================================
    # Part 3: Export predictions
    # ==================================================================
    print("\n" + "=" * 60)
    print("Exporting multi-label predictions...")
    print("=" * 60)

    output_file = os.path.join(tmpdir, "multilabel_predictions.csv")
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index"] + label_names + [f"{n}_prob" for n in label_names])
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            writer.writerow([i] + pred + [f"{p:.4f}" for p in prob])

    print(f"Predictions saved to: {output_file}")

    # Show sample predictions
    print("\nSample predictions:")
    for i in range(min(5, len(preds))):
        active = [n for n, b in zip(label_names, preds[i]) if b]
        print(f"  Sample {i}: {active if active else '(none)'}")


if __name__ == "__main__":
    main()
