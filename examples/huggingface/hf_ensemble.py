"""Example: Model Ensemble and Knowledge Distillation with HuggingFace Models.

This example demonstrates ensemble methods and knowledge distillation techniques:
- Simple averaging ensemble
- Weighted ensemble with learned weights
- Stacking ensemble
- Knowledge distillation (teacher-student)
- Performance comparison

Usage:
    python examples/hf_ensemble.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from autotimm import ImageClassifier


def create_sample_data(batch_size: int = 8, num_classes: int = 10):
    """Create sample data for demonstration."""
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, num_classes, (batch_size,))
    return images, labels


class SimpleEnsemble(nn.Module):
    """Simple averaging ensemble of multiple models."""

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average predictions from all models."""
        # Get predictions from each model
        predictions = []
        for model in self.models:
            with torch.inference_mode():
                pred = model(x)
                if isinstance(pred, dict):
                    pred = pred.get(
                        "logits", pred.get("output", next(iter(pred.values())))
                    )
                predictions.append(pred)

        # Average logits
        avg_prediction = torch.stack(predictions).mean(dim=0)
        return avg_prediction


class WeightedEnsemble(nn.Module):
    """Weighted ensemble with learnable weights."""

    def __init__(self, models: list[nn.Module], num_classes: int):
        super().__init__()
        self.models = nn.ModuleList(models)
        # Learnable weights for each model
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted average of predictions."""
        # Get predictions from each model
        predictions = []
        for model in self.models:
            with torch.inference_mode():
                pred = model(x)
                if isinstance(pred, dict):
                    pred = pred.get(
                        "logits", pred.get("output", next(iter(pred.values())))
                    )
                predictions.append(pred)

        # Apply softmax to weights to ensure they sum to 1
        normalized_weights = F.softmax(self.weights, dim=0)

        # Weighted average
        weighted_pred = sum(
            w * pred for w, pred in zip(normalized_weights, predictions)
        )
        return weighted_pred


class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining hard and soft targets."""

    def __init__(self, alpha: float = 0.5, temperature: float = 3.0):
        """
        Args:
            alpha: Weight for distillation loss (1-alpha for hard target loss)
            temperature: Temperature for softening probability distributions
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distillation loss.

        Args:
            student_logits: Raw logits from student model
            teacher_logits: Raw logits from teacher model
            labels: Ground truth labels
        """
        # Hard target loss (student vs ground truth)
        hard_loss = self.ce_loss(student_logits, labels)

        # Soft target loss (student vs teacher)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction="batchmean",
        ) * (self.temperature**2)

        # Combined loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return loss


def example_1_simple_ensemble():
    """Example 1: Simple averaging ensemble."""
    print("=" * 80)
    print("Example 1: Simple Averaging Ensemble")
    print("=" * 80)

    # Create diverse models from HF Hub
    model_configs = [
        ("ResNet-18", "hf-hub:timm/resnet18.a1_in1k"),
        ("MobileNet-V3", "hf-hub:timm/mobilenetv3_small_100.lamb_in1k"),
        ("EfficientNet-B0", "hf-hub:timm/efficientnet_b0.ra_in1k"),
    ]

    print("\nCreating ensemble members:")
    models = []
    for name, backbone in model_configs:
        model = ImageClassifier(backbone=backbone, num_classes=10)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  • {name:20s}: {n_params:>10,} parameters")
        models.append(model)

    # Create ensemble
    ensemble = SimpleEnsemble(models)
    print(f"\n✓ Created ensemble with {len(models)} models")

    # Test ensemble
    images, _labels = create_sample_data(batch_size=4)
    with torch.inference_mode():
        ensemble_pred = ensemble(images)

    print(f"✓ Ensemble prediction shape: {ensemble_pred.shape}")
    print(f"✓ Averaged logits from {len(models)} diverse architectures")

    # Compare individual vs ensemble predictions
    print("\nPrediction comparison (first sample):")
    for i, (name, model) in enumerate(zip([n for n, _ in model_configs], models)):
        with torch.inference_mode():
            pred = model(images[0:1])
            if isinstance(pred, dict):
                pred = pred.get("logits", pred.get("output", next(iter(pred.values()))))
            top_class = pred.argmax(dim=1).item()
            confidence = F.softmax(pred, dim=1).max().item()
            print(f"  {name:20s}: Class {top_class}, Confidence {confidence:.3f}")

    ensemble_class = ensemble_pred[0].argmax().item()
    ensemble_conf = F.softmax(ensemble_pred[0:1], dim=1).max().item()
    print(f"  {'Ensemble':20s}: Class {ensemble_class}, Confidence {ensemble_conf:.3f}")

    print("\nBenefits of averaging ensemble:")
    print("  • Reduces variance (more stable predictions)")
    print("  • Often improves accuracy by 1-3%")
    print("  • No additional training required")
    print("  • Works best with diverse architectures")


def example_2_weighted_ensemble():
    """Example 2: Weighted ensemble with learned weights."""
    print("\n" + "=" * 80)
    print("Example 2: Weighted Ensemble (Learned Weights)")
    print("=" * 80)

    # Create models
    model_configs = [
        ("ResNet-18", "hf-hub:timm/resnet18.a1_in1k"),
        ("ResNet-34", "hf-hub:timm/resnet34.a1_in1k"),
    ]

    print("\nCreating ensemble members:")
    models = []
    for name, backbone in model_configs:
        model = ImageClassifier(backbone=backbone, num_classes=10)
        model.eval()
        print(f"  • {name}")
        models.append(model)

    # Create weighted ensemble
    weighted_ensemble = WeightedEnsemble(models, num_classes=10)
    print("\n✓ Created weighted ensemble")
    print(f"✓ Initial weights: {weighted_ensemble.weights.data.tolist()}")

    # Simulate learning optimal weights on validation set
    print("\nOptimizing ensemble weights on validation set...")
    print("(Simulated - in practice, train on validation data)")

    # Manually set example weights (in practice, these would be learned)
    with torch.inference_mode():
        # Assume ResNet-34 performs better, give it more weight
        weighted_ensemble.weights.data = torch.tensor([0.4, 0.6])

    normalized = F.softmax(weighted_ensemble.weights, dim=0)
    print(f"✓ Optimized weights: {normalized.tolist()}")

    # Test weighted ensemble
    images, _ = create_sample_data(batch_size=4)
    with torch.inference_mode():
        weighted_pred = weighted_ensemble(images)

    print(f"✓ Weighted ensemble prediction shape: {weighted_pred.shape}")

    print("\nBenefits of weighted ensemble:")
    print("  • Learns optimal combination of models")
    print("  • Can handle models of different quality")
    print("  • Usually outperforms simple averaging")
    print("  • Weights can be optimized on validation set")


def example_3_knowledge_distillation():
    """Example 3: Knowledge distillation (teacher-student)."""
    print("\n" + "=" * 80)
    print("Example 3: Knowledge Distillation")
    print("=" * 80)

    # Teacher: Large, accurate model
    teacher = ImageClassifier(
        backbone="hf-hub:timm/resnet50.a1_in1k",
        num_classes=10,
    )
    teacher.eval()

    # Student: Small, efficient model
    student = ImageClassifier(
        backbone="hf-hub:timm/mobilenetv3_small_100.lamb_in1k",
        num_classes=10,
    )

    # Count parameters
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())

    print("\nModel comparison:")
    print(f"  Teacher (ResNet-50):     {teacher_params:>10,} parameters")
    print(f"  Student (MobileNet-V3):  {student_params:>10,} parameters")
    print(f"  Compression ratio:       {teacher_params / student_params:.1f}x smaller")

    # Create distillation loss
    distill_loss = DistillationLoss(alpha=0.7, temperature=3.0)
    print(
        f"\n✓ Created distillation loss (α={distill_loss.alpha}, T={distill_loss.temperature})"
    )

    # Simulate distillation training
    print("\nDistillation training process:")
    images, labels = create_sample_data(batch_size=8)

    # Get teacher predictions (soft targets)
    with torch.inference_mode():
        teacher_logits = teacher(images)
        if isinstance(teacher_logits, dict):
            teacher_logits = teacher_logits.get(
                "logits",
                teacher_logits.get("output", next(iter(teacher_logits.values()))),
            )

    # Get student predictions
    student.train()
    student_logits = student(images)
    if isinstance(student_logits, dict):
        student_logits = student_logits.get(
            "logits", student_logits.get("output", next(iter(student_logits.values())))
        )

    # Compute distillation loss
    loss = distill_loss(student_logits, teacher_logits, labels)
    print(f"  • Distillation loss: {loss.item():.4f}")

    # Break down loss components
    ce_loss = F.cross_entropy(student_logits, labels)
    student_soft = F.log_softmax(student_logits / 3.0, dim=1)
    teacher_soft = F.softmax(teacher_logits / 3.0, dim=1)
    kl_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * 9.0

    print(f"  • Hard target loss (CE): {ce_loss.item():.4f}")
    print(f"  • Soft target loss (KL): {kl_loss.item():.4f}")

    print("\nDistillation insights:")
    print("  • Teacher provides 'soft' targets with uncertainty information")
    print("  • Student learns from both ground truth and teacher")
    print("  • Temperature parameter controls softness of distributions")
    print("  • α balances hard vs soft target importance")

    print("\nTypical results:")
    print("  • Student trained from scratch: ~85% accuracy")
    print("  • Student with distillation: ~88% accuracy (3% gain)")
    print("  • Teacher accuracy: ~91%")
    print("  • Student retains 96% of teacher performance at 10x smaller size")


def example_4_ensemble_vs_distillation():
    """Example 4: Compare ensemble and distillation approaches."""
    print("\n" + "=" * 80)
    print("Example 4: Ensemble vs Distillation Comparison")
    print("=" * 80)

    comparison = {
        "Simple Ensemble": {
            "accuracy": "+2-3%",
            "inference_time": "N × single model",
            "memory": "N × single model",
            "training_cost": "Low (no training needed)",
            "best_for": "When inference cost is not critical",
        },
        "Weighted Ensemble": {
            "accuracy": "+2.5-4%",
            "inference_time": "N × single model + weight optimization",
            "memory": "N × single model",
            "training_cost": "Low (only weight training)",
            "best_for": "Slightly better than simple ensemble",
        },
        "Knowledge Distillation": {
            "accuracy": "+1-2%",
            "inference_time": "1 × student model (fast!)",
            "memory": "1 × student model (small!)",
            "training_cost": "High (full student training)",
            "best_for": "Production deployment, edge devices",
        },
    }

    print("\nMethod comparison:\n")
    print(f"{'Method':<25} {'Accuracy Gain':>15} {'Inference Time':>20} {'Best For'}")
    print("-" * 80)

    for method, stats in comparison.items():
        print(
            f"{method:<25} {stats['accuracy']:>15} {stats['inference_time']:>20} {stats['best_for']}"
        )

    print("\nDecision guide:")
    print("\n1. Use Simple Ensemble when:")
    print("   • You have multiple trained models")
    print("   • Inference time is not critical (offline processing)")
    print("   • Want quick accuracy boost without training")

    print("\n2. Use Weighted Ensemble when:")
    print("   • You have validation set to optimize weights")
    print("   • Models have varying quality")
    print("   • Can afford slightly more computation")

    print("\n3. Use Knowledge Distillation when:")
    print("   • Need fast inference (production, edge devices)")
    print("   • Have limited memory budget")
    print("   • Can afford training time")
    print("   • Have a strong teacher model")

    print("\n4. Hybrid approach:")
    print("   • Train ensemble as teacher")
    print("   • Distill into single student")
    print("   • Best of both worlds!")


def example_5_practical_tips():
    """Example 5: Practical tips for ensembles and distillation."""
    print("\n" + "=" * 80)
    print("Example 5: Practical Tips & Best Practices")
    print("=" * 80)

    print("\nEnsemble Best Practices:")
    print("\n1. Model Diversity:")
    print("   • Use different architectures (ResNet + EfficientNet + ViT)")
    print("   • Use different pretraining datasets (IN1k + IN21k)")
    print("   • Use different augmentation strategies")
    print("   • Use different random seeds")

    print("\n2. Ensemble Size:")
    print("   • 3-5 models: good trade-off")
    print("   • >5 models: diminishing returns")
    print("   • 2 models: minimal improvement")

    print("\n3. Weighted Ensemble Optimization:")
    print("   • Use held-out validation set")
    print("   • Try gradient-based optimization")
    print("   • Grid search for small ensembles")
    print("   • Constrain weights to sum to 1")

    print("\nDistillation Best Practices:")
    print("\n1. Temperature Selection:")
    print("   • T=1: hard targets (no distillation)")
    print("   • T=3-5: typical range for vision")
    print("   • T=10+: very soft, more regularization")
    print("   • Higher T for larger teacher-student gap")

    print("\n2. Alpha (α) Selection:")
    print("   • α=0.5: equal weight to hard and soft targets")
    print("   • α=0.7-0.9: emphasize soft targets (typical)")
    print("   • α=0.1-0.3: emphasize hard targets")
    print("   • Higher α when teacher is very strong")

    print("\n3. Student Architecture:")
    print("   • Use same family as teacher (easier)")
    print("   • Or use efficient architecture (MobileNet, TinyNet)")
    print("   • Student capacity should be ~0.1-0.3x teacher")
    print("   • Too small: limited learning capacity")
    print("   • Too large: defeats purpose of distillation")

    print("\n4. Training Strategy:")
    print("   • Pretrain student on same data as teacher")
    print("   • Use same augmentation as teacher training")
    print("   • Train longer than normal (2-3x epochs)")
    print("   • Use lower learning rate than from-scratch training")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Model Ensemble & Knowledge Distillation with HuggingFace Models")
    print("=" * 80)
    print("\nThis example demonstrates techniques for combining multiple models")
    print("and distilling knowledge into compact student models.\n")

    # Run examples
    try:
        example_1_simple_ensemble()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        example_2_weighted_ensemble()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        example_3_knowledge_distillation()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    try:
        example_4_ensemble_vs_distillation()
    except Exception as e:
        print(f"Example 4 failed: {e}")

    try:
        example_5_practical_tips()
    except Exception as e:
        print(f"Example 5 failed: {e}")

    print("\n" + "=" * 80)
    print("Key Takeaways")
    print("=" * 80)
    print("\n1. Ensembles boost accuracy by 2-4%")
    print("   • Simple averaging is often sufficient")
    print("   • Weighted ensembles can be optimized")
    print("   • Diversity is key for ensemble success")

    print("\n2. Knowledge distillation enables deployment")
    print("   • Compress large models into small ones")
    print("   • Retain 95-98% of teacher performance")
    print("   • 5-10x faster inference")

    print("\n3. Choose based on deployment constraints")
    print("   • Offline/cloud: use ensemble")
    print("   • Production/edge: use distillation")
    print("   • Hybrid: ensemble teacher → distilled student")

    print("\n4. Hyperparameters matter")
    print("   • Temperature (T): 3-5 for vision")
    print("   • Alpha (α): 0.7-0.9 for strong teachers")
    print("   • Student size: 0.1-0.3x teacher capacity")

    print("\nNext steps:")
    print("• Implement ensemble for your best models")
    print("• Try distillation for production deployment")
    print("• Experiment with different α and T values")
    print("• Measure accuracy-speed trade-offs")


if __name__ == "__main__":
    main()
