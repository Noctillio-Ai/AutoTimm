"""Example: Hyperparameter Tuning with Optuna and HuggingFace Models.

This example demonstrates automated hyperparameter optimization using Optuna:
- Basic hyperparameter search
- Multi-objective optimization (accuracy + speed)
- Pruning unpromising trials
- Architecture-specific search spaces
- Visualization of optimization history

Usage:
    python examples/hf_hyperparameter_tuning.py

Requirements:
    pip install optuna optuna-dashboard  # For hyperparameter optimization
"""

from __future__ import annotations

import time

import torch

from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
)

# Optional: Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.trial import Trial
    from optuna.visualization import (
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
    )

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠ Optuna not available. Install: pip install optuna")


def create_sample_datamodule():
    """Create sample data module for demonstration."""
    # Using CIFAR-10 as example
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=64,
        num_workers=4,
    )
    data.setup("fit")
    return data


def example_1_basic_optuna_search():
    """Example 1: Basic hyperparameter search with Optuna."""
    if not OPTUNA_AVAILABLE:
        print("\n⚠ Skipping Optuna examples (optuna not installed)")
        return

    print("=" * 80)
    print("Example 1: Basic Hyperparameter Search")
    print("=" * 80)

    def objective(trial: Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Validation accuracy (to maximize)
        """
        # Sample hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        # Model architecture
        backbone = trial.suggest_categorical(
            "backbone",
            [
                "hf-hub:timm/resnet18.a1_in1k",
                "hf-hub:timm/resnet34.a1_in1k",
                "hf-hub:timm/mobilenetv3_small_100.lamb_in1k",
            ],
        )

        # Create data module with trial batch size
        data = ImageDataModule(
            data_dir="./data",
            dataset_name="CIFAR10",
            image_size=224,
            batch_size=batch_size,
            num_workers=4,
        )
        data.setup("fit")

        # Create model with trial hyperparameters
        model = ImageClassifier(
            backbone=backbone,
            num_classes=10,
            lr=lr,
            weight_decay=weight_decay,
        )

        # Train for few epochs (quick trial)
        trainer = AutoTrainer(
            max_epochs=3,  # Short for fast search
            accelerator="auto",
            enable_checkpointing=False,
            logger=False,
        )

        trainer.fit(model, datamodule=data)

        # Return validation accuracy
        val_metrics = trainer.callback_metrics
        accuracy = val_metrics.get("val_accuracy", 0.0)

        return accuracy.item() if torch.is_tensor(accuracy) else float(accuracy)

    print("\nStarting hyperparameter search...")
    print("  • Searching: lr, weight_decay, batch_size, backbone")
    print("  • Objective: Maximize validation accuracy")
    print("  • Trials: 10 (set higher in practice, e.g., 50-100)")

    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name="hf_model_optimization",
    )

    # Run optimization (reduced trials for demo)
    study.optimize(objective, n_trials=10)

    # Print results
    print("\n" + "=" * 80)
    print("Optimization Results")
    print("=" * 80)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  • {key}: {value:.6f}")
        else:
            print(f"  • {key}: {value}")

    # Analyze parameter importance
    print("\n" + "-" * 80)
    print("Parameter Importance (top 3):")
    try:
        importances = optuna.importance.get_param_importances(study)
        for i, (param, importance) in enumerate(list(importances.items())[:3]):
            print(f"  {i+1}. {param}: {importance:.4f}")
    except Exception:
        print("  (Not enough trials to compute importances)")

    return study


def example_2_multi_objective_optimization():
    """Example 2: Multi-objective optimization (accuracy + speed)."""
    if not OPTUNA_AVAILABLE:
        return

    print("\n" + "=" * 80)
    print("Example 2: Multi-Objective Optimization")
    print("=" * 80)
    print("\nOptimizing for both accuracy AND inference speed")

    def multi_objective(trial: Trial) -> tuple[float, float]:
        """
        Multi-objective: maximize accuracy, minimize inference time.

        Returns:
            Tuple of (accuracy, -inference_time)
        """
        # Sample hyperparameters
        backbone = trial.suggest_categorical(
            "backbone",
            [
                "hf-hub:timm/resnet18.a1_in1k",
                "hf-hub:timm/efficientnet_b0.ra_in1k",
                "hf-hub:timm/mobilenetv3_small_100.lamb_in1k",
            ],
        )

        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        # Create model
        model = ImageClassifier(
            backbone=backbone,
            num_classes=10,
            lr=lr,
        )
        model.eval()

        # Measure inference time
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.inference_mode():
            # Warmup
            for _ in range(10):
                _ = model(dummy_input)

            # Measure
            start = time.time()
            for _ in range(50):
                _ = model(dummy_input)
            inference_time = (time.time() - start) / 50 * 1000  # ms

        # Quick training for accuracy (simulated here)
        # In practice, you'd train and get val_accuracy
        # For demo, use random accuracy
        simulated_accuracy = 0.75 + torch.rand(1).item() * 0.2

        # Return both objectives (accuracy, -time to maximize both)
        return simulated_accuracy, -inference_time

    print("\nCreating multi-objective study...")
    study = optuna.create_study(
        directions=["maximize", "maximize"],  # Both objectives to maximize
        study_name="accuracy_speed_tradeoff",
    )

    # Optimize (reduced trials for demo)
    study.optimize(multi_objective, n_trials=15)

    # Analyze Pareto front
    print("\n" + "=" * 80)
    print("Multi-Objective Results")
    print("=" * 80)
    print(f"\nCompleted {len(study.trials)} trials")
    print(f"Pareto-optimal solutions: {len(study.best_trials)}")

    print("\nPareto front (accuracy vs speed):")
    print(f"{'Trial':<8} {'Backbone':<50} {'Accuracy':>10} {'Time (ms)':>12}")
    print("-" * 80)

    for trial in study.best_trials[:5]:  # Show top 5
        backbone = trial.params.get("backbone", "N/A").replace("hf-hub:timm/", "")
        accuracy = trial.values[0]
        time_ms = -trial.values[1]
        print(f"{trial.number:<8} {backbone:<50} {accuracy:>10.4f} {time_ms:>12.2f}")

    print("\nTrade-off insights:")
    print("  • MobileNet: fastest but lower accuracy")
    print("  • ResNet: balanced speed and accuracy")
    print("  • EfficientNet: best accuracy but slower")


def example_3_pruning_unpromising_trials():
    """Example 3: Pruning unpromising trials early."""
    if not OPTUNA_AVAILABLE:
        return

    print("\n" + "=" * 80)
    print("Example 3: Pruning Unpromising Trials")
    print("=" * 80)

    print("\nPruning stops unpromising trials early to save compute")

    def objective_with_pruning(trial: Trial) -> float:
        """Objective with intermediate value reporting for pruning."""
        # Sample hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        backbone = trial.suggest_categorical(
            "backbone",
            ["hf-hub:timm/resnet18.a1_in1k", "hf-hub:timm/resnet34.a1_in1k"],
        )

        # Simulated training loop
        print(f"  Trial {trial.number}: {backbone.split('/')[-1]}, lr={lr:.6f}")

        for epoch in range(5):
            # Simulated validation accuracy (increasing with epochs)
            val_acc = 0.5 + epoch * 0.05 + torch.rand(1).item() * 0.1

            # Report intermediate value
            trial.report(val_acc, epoch)

            # Check if trial should be pruned
            if trial.should_prune():
                print(f"    → Pruned at epoch {epoch}")
                raise optuna.TrialPruned()

        return val_acc

    print("\nRunning optimization with MedianPruner...")
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,  # Don't prune first 3 trials
            n_warmup_steps=1,  # Start pruning after epoch 1
        ),
    )

    study.optimize(objective_with_pruning, n_trials=10)

    # Analyze pruning
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("\n" + "=" * 80)
    print("Pruning Results")
    print("=" * 80)
    print(f"\nCompleted trials: {len(completed_trials)}")
    print(f"Pruned trials: {len(pruned_trials)}")
    print(
        f"Compute saved: ~{len(pruned_trials) * 60:.0f}% (assuming avg pruning at epoch 2/5)"
    )

    print("\nBest result:")
    print(f"  Accuracy: {study.best_value:.4f}")
    print(f"  Parameters: {study.best_params}")


def example_4_architecture_search_spaces():
    """Example 4: Architecture-specific search spaces."""
    if not OPTUNA_AVAILABLE:
        return

    print("\n" + "=" * 80)
    print("Example 4: Architecture-Specific Search Spaces")
    print("=" * 80)

    print("\nDifferent model families need different hyperparameters:")

    print("\nCNN Search Space (ResNet, EfficientNet):")
    cnn_space = {
        "lr": "1e-4 to 3e-3 (higher LR)",
        "weight_decay": "1e-4 to 1e-2",
        "dropout": "0.0 to 0.3",
        "batch_size": "32, 64, 128 (larger OK)",
        "optimizer": "SGD with momentum or AdamW",
    }

    for key, value in cnn_space.items():
        print(f"  • {key}: {value}")

    print("\nVision Transformer Search Space:")
    vit_space = {
        "lr": "1e-5 to 5e-4 (lower LR!)",
        "weight_decay": "0.01 to 0.1 (higher)",
        "dropout": "0.0 to 0.1 (lower)",
        "batch_size": "16, 32 (smaller due to memory)",
        "optimizer": "AdamW (not SGD)",
        "warmup_epochs": "5 to 20 (important!)",
    }

    for key, value in vit_space.items():
        print(f"  • {key}: {value}")

    print("\nExample: Architecture-aware objective")
    example_code = '''
def architecture_aware_objective(trial: Trial) -> float:
    """Objective with architecture-specific search space."""

    # First choose architecture
    arch_family = trial.suggest_categorical("arch_family", ["cnn", "vit"])

    if arch_family == "cnn":
        backbone = trial.suggest_categorical(
            "backbone",
            [
                "hf-hub:timm/resnet34.a1_in1k",
                "hf-hub:timm/efficientnet_b1.ra_in1k",
            ],
        )
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    else:  # ViT
        backbone = trial.suggest_categorical(
            "backbone",
            [
                "hf-hub:timm/vit_small_patch16_224.augreg_in1k",
                "hf-hub:timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k",
            ],
        )
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)  # Lower LR!
        weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])  # Smaller!

    # Rest of training code...
    model = ImageClassifier(backbone=backbone, num_classes=10, lr=lr, weight_decay=weight_decay)
    # ... train and return accuracy
'''

    print(example_code)


def example_5_visualization():
    """Example 5: Visualizing optimization results."""
    if not OPTUNA_AVAILABLE:
        return

    print("\n" + "=" * 80)
    print("Example 5: Visualization & Analysis")
    print("=" * 80)

    print("\nOptuna provides built-in visualizations:")
    print("\n1. Optimization History:")
    print("   Shows how trials improve over time")

    print("\n2. Parameter Importances:")
    print("   Identifies which parameters matter most")

    print("\n3. Parallel Coordinate Plot:")
    print("   Shows parameter combinations and their results")

    print("\n4. Contour Plot:")
    print("   2D visualization of parameter interactions")

    example_code = """
import optuna

# After running study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Generate visualizations
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour,
)

# 1. Optimization history
fig1 = plot_optimization_history(study)
fig1.write_html("optimization_history.html")

# 2. Parameter importances
fig2 = plot_param_importances(study)
fig2.write_html("param_importances.html")

# 3. Parallel coordinate plot
fig3 = plot_parallel_coordinate(study)
fig3.write_html("parallel_coordinate.html")

# 4. Contour plot (for 2 parameters)
fig4 = plot_contour(study, params=["lr", "weight_decay"])
fig4.write_html("contour_plot.html")

# Save study for later analysis
import joblib
joblib.dump(study, "study.pkl")

# Load later
study = joblib.load("study.pkl")
"""

    print(example_code)

    print("\n✓ Dashboard (optional):")
    print("  • Run: optuna-dashboard sqlite:///optuna.db")
    print("  • Real-time monitoring in browser")
    print("  • Compare multiple studies")


def example_6_practical_tips():
    """Example 6: Practical tips for hyperparameter tuning."""
    print("\n" + "=" * 80)
    print("Example 6: Practical Tips & Best Practices")
    print("=" * 80)

    print("\n1. Search Space Design:")
    print("   • Start broad, then narrow around promising regions")
    print("   • Use log scale for learning rate and weight decay")
    print("   • Don't optimize too many parameters at once (5-7 max)")

    print("\n2. Computational Budget:")
    print("   • Small dataset: 50-100 trials")
    print("   • Medium dataset: 30-50 trials")
    print("   • Large dataset: 20-30 trials (use pruning!)")

    print("\n3. Training Strategy:")
    print("   • Use short epochs for trials (3-5 epochs)")
    print("   • Train best model longer afterward")
    print("   • Use validation set for optimization")
    print("   • Test set only for final evaluation")

    print("\n4. Parallel Execution:")
    print("   • Optuna supports distributed optimization")
    print("   • Use multiple GPUs/machines")
    print("   • Share results via database")

    example_parallel = """
# Parallel optimization with multiple workers
import optuna

# Create shared study (SQLite for demo, use PostgreSQL for production)
study = optuna.create_study(
    study_name="distributed_hpo",
    storage="sqlite:///optuna.db",
    load_if_exists=True,
    direction="maximize",
)

# Run on multiple machines/GPUs simultaneously
# Each worker runs:
study.optimize(objective, n_trials=50)
"""

    print(example_parallel)

    print("\n5. When to Use HPO:")
    print("   • New dataset/domain: Always!")
    print("   • Production deployment: Optimize for inference speed too")
    print("   • Academic research: Document search space and # trials")
    print("   • Proof-of-concept: Use defaults, HPO later")

    print("\n6. What to Optimize:")
    print("   Priority 1: Learning rate, weight decay")
    print("   Priority 2: Batch size, optimizer")
    print("   Priority 3: Architecture, dropout")
    print("   Priority 4: Augmentation, scheduler")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Hyperparameter Tuning with Optuna & HuggingFace Models")
    print("=" * 80)
    print("\nThis example demonstrates automated hyperparameter optimization")
    print("using Optuna for finding optimal training configurations.\n")

    if not OPTUNA_AVAILABLE:
        print("⚠ Optuna is not installed.")
        print("Install with: pip install optuna optuna-dashboard")
        print("\nShowing example code only...\n")

    # Run examples
    try:
        example_1_basic_optuna_search()
    except Exception as e:
        print(f"Example 1 skipped: {e}")

    try:
        example_2_multi_objective_optimization()
    except Exception as e:
        print(f"Example 2 skipped: {e}")

    try:
        example_3_pruning_unpromising_trials()
    except Exception as e:
        print(f"Example 3 skipped: {e}")

    try:
        example_4_architecture_search_spaces()
    except Exception as e:
        print(f"Example 4 skipped: {e}")

    try:
        example_5_visualization()
    except Exception as e:
        print(f"Example 5 skipped: {e}")

    try:
        example_6_practical_tips()
    except Exception as e:
        print(f"Example 6 skipped: {e}")

    print("\n" + "=" * 80)
    print("Key Takeaways")
    print("=" * 80)
    print("\n1. Optuna automates hyperparameter search")
    print("   • Define objective function")
    print("   • Specify search space")
    print("   • Run trials and get best parameters")

    print("\n2. Multi-objective optimization")
    print("   • Optimize accuracy AND speed")
    print("   • Find Pareto-optimal solutions")
    print("   • Choose based on deployment constraints")

    print("\n3. Pruning saves compute")
    print("   • Stop unpromising trials early")
    print("   • 30-50% compute savings typical")
    print("   • Use MedianPruner or HyperbandPruner")

    print("\n4. Architecture-specific search spaces")
    print("   • CNNs: higher LR, larger batches")
    print("   • ViTs: lower LR, smaller batches, warmup")

    print("\n5. Best practices:")
    print("   • Start with 20-50 trials")
    print("   • Use short training (3-5 epochs per trial)")
    print("   • Train best model longer afterward")
    print("   • Prioritize lr and weight_decay")
    print("   • Use visualization to understand results")

    print("\nNext steps:")
    print("• Install Optuna: pip install optuna")
    print("• Define your objective function")
    print("• Run optimization with 50-100 trials")
    print("• Visualize results and parameter importances")
    print("• Train final model with best hyperparameters")


if __name__ == "__main__":
    main()
