"""Example: Advanced Transfer Learning with HuggingFace Hub Models.

This example demonstrates sophisticated transfer learning strategies including:
- Progressive unfreezing (gradually unfreeze layers during training)
- Layer-wise learning rate decay (LLRD)
- Discriminative fine-tuning (different LR for different layers)
- Two-phase training (frozen backbone → unfrozen backbone)
- Comparison of pretraining datasets (IN1k vs IN21k vs IN22k)
- Backbone selection strategies

Usage:
    python examples/hf_transfer_learning.py
"""

from __future__ import annotations

import torch
from torch import nn

from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
)


def explore_pretraining_impact():
    """Compare models pretrained on different datasets."""
    print("=" * 80)
    print("Impact of Pretraining Dataset on Transfer Learning")
    print("=" * 80)

    print("\nPretraining Dataset Characteristics:\n")
    print(f"{'Dataset':<30} {'Images':>15} {'Classes':>10} {'Best For'}")
    print("-" * 80)

    dataset_info = [
        ("ImageNet-1k", "1.28M", "1,000", "General-purpose, fast training"),
        (
            "ImageNet-21k → 1k",
            "14M → 1.28M",
            "21k → 1k",
            "Complex domains, many classes",
        ),
        ("ImageNet-22k", "14M", "22,000", "Fine-grained recognition"),
        ("Semi-supervised", "940M", "1,000", "Limited labeled data scenarios"),
    ]

    for dataset, n_images, n_classes, best_for in dataset_info:
        print(f"{dataset:<30} {n_images:>15} {n_classes:>10} {best_for}")

    print("\nKey Insights:")
    print("  • IN21k/22k models: Better features for transfer learning")
    print("  • Semi-supervised: Robust features learned from diverse data")
    print("  • IN1k: Faster convergence, good baseline")
    print(
        "  • Choice depends on: target dataset size, domain similarity, compute budget"
    )


def example_1_basic_transfer_learning():
    """Example 1: Basic transfer learning (freeze → unfreeze)."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Transfer Learning (Freeze → Unfreeze)")
    print("=" * 80)

    # Phase 1: Train with frozen backbone
    print("\nPhase 1: Training classification head (backbone frozen)")
    print("-" * 80)

    model = ImageClassifier(
        backbone="hf-hub:timm/resnet18.a1_in1k",
        num_classes=10,
        freeze_backbone=True,  # Freeze backbone initially
    )

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("✓ Created model with frozen backbone")
    print(f"  • Trainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"  • Frozen backbone: {model.freeze_backbone}")

    # Setup data (CIFAR-10 as example)
    ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=64,
        num_workers=4,
    )

    # Train phase 1 (head only)
    AutoTrainer(
        max_epochs=5,
        accelerator="auto",
        log_every_n_steps=10,
    )

    print("\nTraining classification head for 5 epochs...")
    print("(Simulated - set max_epochs higher for real training)")

    # Phase 2: Unfreeze and fine-tune
    print("\nPhase 2: Fine-tuning entire model (backbone unfrozen)")
    print("-" * 80)

    model.unfreeze_backbone()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("✓ Unfroze backbone")
    print(f"  • Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Train phase 2 (full model, lower learning rate)
    AutoTrainer(
        max_epochs=10,
        accelerator="auto",
        log_every_n_steps=10,
    )

    print("\nFine-tuning full model for 10 epochs with lower LR...")
    print("(Simulated - adjust hyperparameters for real training)")

    print("\n✓ Two-phase training complete!")


def example_2_progressive_unfreezing():
    """Example 2: Progressive unfreezing (unfreeze layers gradually)."""
    print("\n" + "=" * 80)
    print("Example 2: Progressive Unfreezing")
    print("=" * 80)

    model = ImageClassifier(
        backbone="hf-hub:timm/resnet50.a1_in1k",
        num_classes=10,
        freeze_backbone=True,
    )

    print("✓ Created ResNet-50 model")
    print("\nProgressive unfreezing strategy:")
    print("  1. Train head only (5 epochs)")
    print("  2. Unfreeze layer4 + train (3 epochs)")
    print("  3. Unfreeze layer3 + train (3 epochs)")
    print("  4. Unfreeze layer2 + train (3 epochs)")
    print("  5. Unfreeze all + train (5 epochs)")

    # Helper function to unfreeze specific layers
    def unfreeze_layers(model, layer_names: list[str]):
        """Unfreeze specific layers by name."""
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True

    def count_trainable(model):
        """Count trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Stage 1: Head only
    print("\n" + "-" * 80)
    print("Stage 1: Training head only")
    trainable = count_trainable(model)
    print(f"  Trainable params: {trainable:,}")

    # Stage 2: Unfreeze layer4
    print("\n" + "-" * 80)
    print("Stage 2: Unfreeze layer4")
    unfreeze_layers(model, ["backbone.layer4", "head"])
    trainable = count_trainable(model)
    print(f"  Trainable params: {trainable:,}")

    # Stage 3: Unfreeze layer3
    print("\n" + "-" * 80)
    print("Stage 3: Unfreeze layer3")
    unfreeze_layers(model, ["backbone.layer3", "backbone.layer4", "head"])
    trainable = count_trainable(model)
    print(f"  Trainable params: {trainable:,}")

    # Stage 4: Unfreeze layer2
    print("\n" + "-" * 80)
    print("Stage 4: Unfreeze layer2")
    unfreeze_layers(
        model, ["backbone.layer2", "backbone.layer3", "backbone.layer4", "head"]
    )
    trainable = count_trainable(model)
    print(f"  Trainable params: {trainable:,}")

    # Stage 5: Unfreeze all
    print("\n" + "-" * 80)
    print("Stage 5: Unfreeze all layers")
    model.unfreeze_backbone()
    trainable = count_trainable(model)
    print(f"  Trainable params: {trainable:,}")

    print("\n✓ Progressive unfreezing schedule defined")
    print("\nBenefits:")
    print("  • Prevents catastrophic forgetting of pretrained features")
    print("  • Allows gradual adaptation to new domain")
    print("  • Often achieves better final performance")


def example_3_layer_wise_lr_decay():
    """Example 3: Layer-wise learning rate decay (LLRD)."""
    print("\n" + "=" * 80)
    print("Example 3: Layer-wise Learning Rate Decay (LLRD)")
    print("=" * 80)

    model = ImageClassifier(
        backbone="hf-hub:timm/resnet50.a1_in1k",
        num_classes=10,
        freeze_backbone=False,
    )

    print("✓ Created ResNet-50 model")
    print("\nLayer-wise LR Decay Strategy:")
    print("  • Earlier layers (closer to input): smaller LR")
    print("  • Later layers (closer to output): larger LR")
    print("  • Head: largest LR")

    def get_layer_wise_lr_groups(
        model: nn.Module,
        base_lr: float = 1e-4,
        decay_factor: float = 0.8,
    ) -> list[dict]:
        """
        Create parameter groups with decaying learning rates.

        Args:
            model: The model
            base_lr: Learning rate for the head
            decay_factor: LR multiplier for each earlier layer (< 1.0)
        """
        # Define layer groups (deepest to shallowest)
        layer_groups = [
            (["head"], "head", 1.0),  # Head gets base_lr
            (["backbone.layer4"], "layer4", decay_factor),
            (["backbone.layer3"], "layer3", decay_factor**2),
            (["backbone.layer2"], "layer2", decay_factor**3),
            (["backbone.layer1"], "layer1", decay_factor**4),
            (["backbone.conv1", "backbone.bn1"], "stem", decay_factor**5),
        ]

        param_groups = []
        assigned_params = set()

        print(
            f"\n{'Layer Group':<15} {'LR Multiplier':>15} {'Effective LR':>15} {'# Params':>12}"
        )
        print("-" * 80)

        for layer_names, group_name, lr_mult in layer_groups:
            params = []
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    if id(param) not in assigned_params:
                        params.append(param)
                        assigned_params.add(id(param))

            if params:
                effective_lr = base_lr * lr_mult
                n_params = sum(p.numel() for p in params)
                param_groups.append({"params": params, "lr": effective_lr})
                print(
                    f"{group_name:<15} {lr_mult:>15.4f} {effective_lr:>15.6f} {n_params:>12,}"
                )

        # Add any remaining parameters
        remaining_params = [
            p for p in model.parameters() if id(p) not in assigned_params
        ]
        if remaining_params:
            param_groups.append(
                {"params": remaining_params, "lr": base_lr * decay_factor**6}
            )
            print(
                f"{'other':<15} {decay_factor**6:>15.4f} {base_lr * decay_factor**6:>15.6f} {sum(p.numel() for p in remaining_params):>12,}"
            )

        return param_groups

    # Create parameter groups with LLRD
    param_groups = get_layer_wise_lr_groups(model, base_lr=1e-3, decay_factor=0.8)

    # Create optimizer with different LRs per group
    torch.optim.AdamW(param_groups, weight_decay=0.01)

    print(f"\n✓ Created optimizer with {len(param_groups)} parameter groups")
    print("\nBenefits of LLRD:")
    print("  • Earlier layers adapt slowly (preserve pretrained features)")
    print("  • Later layers adapt quickly (learn task-specific features)")
    print("  • Often improves final performance by 1-3%")
    print("  • Commonly used in NLP (BERT, GPT), now adopted in vision")


def example_4_discriminative_fine_tuning():
    """Example 4: Discriminative fine-tuning strategies."""
    print("\n" + "=" * 80)
    print("Example 4: Discriminative Fine-tuning Strategies")
    print("=" * 80)

    print("\nStrategy Comparison:\n")

    strategies = {
        "Standard Fine-tuning": {
            "description": "Same LR for all layers",
            "head_lr": 1e-3,
            "layer4_lr": 1e-3,
            "layer1_lr": 1e-3,
            "pros": "Simple, fast to converge",
            "cons": "May overwrite pretrained features",
        },
        "Frozen Backbone": {
            "description": "Only train head",
            "head_lr": 1e-3,
            "layer4_lr": 0.0,
            "layer1_lr": 0.0,
            "pros": "Fast, prevents overfitting",
            "cons": "Limited adaptation to new domain",
        },
        "LLRD": {
            "description": "Gradual LR decay",
            "head_lr": 1e-3,
            "layer4_lr": 8e-4,
            "layer1_lr": 3.3e-4,
            "pros": "Balanced adaptation",
            "cons": "More hyperparameters",
        },
        "Chain-thaw": {
            "description": "Progressive unfreezing",
            "head_lr": "1e-3 → 1e-4",
            "layer4_lr": "0 → 5e-4",
            "layer1_lr": "0 → 1e-4",
            "pros": "Stable, prevents catastrophic forgetting",
            "cons": "Longer training time",
        },
    }

    for strategy_name, config in strategies.items():
        print(f"{strategy_name}:")
        print(f"  Description: {config['description']}")
        print("  LR schedule:")
        print(f"    • Head:    {config['head_lr']}")
        print(f"    • Layer4:  {config['layer4_lr']}")
        print(f"    • Layer1:  {config['layer1_lr']}")
        print(f"  Pros:  {config['pros']}")
        print(f"  Cons:  {config['cons']}")
        print()

    print("Decision Guide:")
    print("  • Small dataset (<10k images): Frozen backbone or LLRD")
    print("  • Medium dataset (10k-100k): LLRD or Chain-thaw")
    print("  • Large dataset (>100k): Standard fine-tuning or LLRD")
    print("  • Domain shift (natural → medical): Chain-thaw or LLRD")
    print("  • Similar domain (IN1k → CIFAR): Standard fine-tuning")


def example_5_pretraining_comparison():
    """Example 5: Compare models pretrained on different datasets."""
    print("\n" + "=" * 80)
    print("Example 5: Pretraining Dataset Comparison")
    print("=" * 80)

    # Different pretraining variants of similar architectures
    models_to_compare = {
        "ViT-B/16 (IN1k)": "hf-hub:timm/vit_base_patch16_224.augreg_in1k",
        "ViT-B/16 (IN21k→1k)": "hf-hub:timm/vit_base_patch16_224.augreg_in21k_ft_in1k",
        "ConvNeXt-B (IN1k)": "hf-hub:timm/convnext_base.fb_in1k",
        "ConvNeXt-B (IN22k→1k)": "hf-hub:timm/convnext_base.fb_in22k_ft_in1k",
    }

    print("\nExpected Transfer Learning Performance:\n")
    print(f"{'Model':<25} {'Pretraining':>20} {'Expected Transfer Performance'}")
    print("-" * 80)

    for model_name, model_id in models_to_compare.items():
        if "in21k" in model_id or "in22k" in model_id:
            pretraining = "IN21k/22k"
            performance = "Best (stronger features)"
        else:
            pretraining = "IN1k"
            performance = "Good (standard baseline)"

        print(f"{model_name:<25} {pretraining:>20} {performance}")

    print("\nWhen to use each:")
    print("\nIN1k models:")
    print("  • ✓ Fast training (fewer parameters)")
    print("  • ✓ Standard baseline")
    print("  • ✓ Good for similar domains (natural images)")
    print("  • ✗ May need more fine-tuning for specialized domains")

    print("\nIN21k/22k models:")
    print("  • ✓ Superior transfer learning performance")
    print("  • ✓ Better for specialized domains (medical, satellite)")
    print("  • ✓ Better for fine-grained recognition")
    print("  • ✗ Slower training (more parameters)")
    print("  • ✗ May overfit on very small datasets")


def example_6_practical_training_pipeline():
    """Example 6: Complete practical transfer learning pipeline."""
    print("\n" + "=" * 80)
    print("Example 6: Practical Transfer Learning Pipeline")
    print("=" * 80)

    print("\nRecommended pipeline for a new dataset:")
    print("\n1. Baseline: Frozen backbone (2-5 epochs)")
    print("   • Quick sanity check")
    print("   • Establishes lower bound on performance")

    print("\n2. Standard fine-tuning (10-20 epochs)")
    print("   • Unfreeze all layers")
    print("   • Use moderate learning rate (1e-4)")
    print("   • Establishes upper bound with simple approach")

    print("\n3. LLRD fine-tuning (10-20 epochs)")
    print("   • Layer-wise LR decay (factor=0.8)")
    print("   • Base LR = 1e-3 (head), 3e-4 (layer1)")
    print("   • Often beats standard fine-tuning by 1-3%")

    print("\n4. Progressive unfreezing (if needed)")
    print("   • Use if LLRD underperforms or unstable")
    print("   • Especially helpful with small datasets")

    print("\n5. Hyperparameter optimization")
    print("   • Tune: base_lr, decay_factor, weight_decay")
    print("   • Use validation set for selection")

    print("\n" + "-" * 80)
    print("Example configuration:")

    example_config = """
    # Stage 1: Frozen backbone
    model = ImageClassifier(
        backbone="hf-hub:timm/convnext_tiny.fb_in22k_ft_in1k",
        num_classes=10,
        freeze_backbone=True,
    )
    optimizer = Adam(model.parameters(), lr=1e-3)
    train(model, epochs=5)

    # Stage 2: LLRD fine-tuning
    model.unfreeze_backbone()
    param_groups = get_layer_wise_lr_groups(
        model,
        base_lr=1e-3,
        decay_factor=0.8,
    )
    optimizer = AdamW(param_groups, weight_decay=0.01)
    train(model, epochs=20)
    """

    print(example_config)


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Advanced Transfer Learning with HuggingFace Hub Models")
    print("=" * 80)
    print("\nThis example demonstrates sophisticated transfer learning strategies")
    print("for achieving optimal performance with pretrained models.\n")

    # Run examples
    explore_pretraining_impact()
    example_1_basic_transfer_learning()
    example_2_progressive_unfreezing()
    example_3_layer_wise_lr_decay()
    example_4_discriminative_fine_tuning()
    example_5_pretraining_comparison()
    example_6_practical_training_pipeline()

    print("\n" + "=" * 80)
    print("Key Takeaways")
    print("=" * 80)
    print("\n1. Choose pretraining wisely:")
    print("   • IN21k/22k models transfer better but train slower")
    print("   • IN1k models are good baseline for natural images")

    print("\n2. Layer-wise learning rates are powerful:")
    print("   • LLRD often improves performance by 1-3%")
    print("   • Earlier layers need smaller learning rates")

    print("\n3. Progressive unfreezing prevents catastrophic forgetting:")
    print("   • Especially useful with small datasets")
    print("   • Allows gradual domain adaptation")

    print("\n4. Start simple, then optimize:")
    print("   • Begin with frozen backbone baseline")
    print("   • Try standard fine-tuning")
    print("   • Use LLRD for improvement")
    print("   • Progressive unfreezing as last resort")

    print("\n5. Dataset size matters:")
    print("   • Small (<10k): frozen or minimal fine-tuning")
    print("   • Medium (10k-100k): LLRD recommended")
    print("   • Large (>100k): standard fine-tuning works well")

    print("\nNext steps:")
    print("• Implement LLRD in your training pipeline")
    print("• Experiment with different decay factors (0.7-0.95)")
    print("• Try IN21k/22k models for your domain")
    print("• Monitor validation metrics to prevent overfitting")


if __name__ == "__main__":
    main()
