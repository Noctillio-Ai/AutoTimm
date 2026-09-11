"""Demonstrate automatic learning rate and batch size finding.

This example demonstrates:
- Using TunerConfig for automatic hyperparameter tuning
- Finding optimal learning rate before training
- Finding optimal batch size that fits in GPU memory
- Combining both tuners or using them independently
- MetricManager for metric configuration

Usage:
    python examples/auto_tuning.py
"""

from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    LoggingConfig,
    MetricConfig,
    MetricManager,
    TunerConfig,
)


def main():
    # Data - batch_size here is the initial value for tuning
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=32,  # Starting batch size (will be scaled up if auto_batch_size=True)
        num_workers=4,
    )

    # Configure metrics
    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
    ]

    # Create MetricManager
    metric_manager = MetricManager(configs=metric_configs, num_classes=10)

    # Model - lr here is the initial value (will be overridden by LR finder)
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        metrics=metric_manager,
        logging_config=LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=False,
        ),
        lr=1e-3,  # Initial LR - will be tuned
        scheduler="cosine",
    )

    # ========================================================================
    # Option 1: LR finding only (recommended for most cases)
    # ========================================================================
    print("=" * 60)
    print("Option 1: Learning Rate Finding")
    print("=" * 60)

    trainer_lr = AutoTrainer(
        max_epochs=10,
        accelerator="auto",
        logger=[
            LoggerConfig(backend="tensorboard", params={"save_dir": "logs/lr_tuning"}),
        ],
        tuner_config=TunerConfig(
            auto_lr=True,
            auto_batch_size=False,
            lr_find_kwargs={
                "min_lr": 1e-7,  # Minimum LR to test
                "max_lr": 1.0,  # Maximum LR to test
                "num_training": 100,  # Number of training steps
                "mode": "exponential",  # "exponential" or "linear"
                "early_stop_threshold": 4.0,  # Stop if loss > early_stop_threshold * best_loss
            },
        ),
        checkpoint_monitor="val/accuracy",
        checkpoint_mode="max",
    )

    # The fit() method will:
    # 1. Run LR finder
    # 2. Update model._lr with suggested value
    # 3. Start training with optimal LR
    # trainer_lr.fit(model, datamodule=data)

    # ========================================================================
    # Option 2: Batch size finding only
    # ========================================================================
    print("\n" + "=" * 60)
    print("Option 2: Batch Size Finding")
    print("=" * 60)

    # Reset model for fresh training
    model2 = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        metrics=metric_manager,
        lr=1e-3,
        scheduler="cosine",
    )

    trainer_bs = AutoTrainer(
        max_epochs=10,
        accelerator="auto",
        logger=[
            LoggerConfig(backend="tensorboard", params={"save_dir": "logs/bs_tuning"}),
        ],
        tuner_config=TunerConfig(
            auto_lr=False,
            auto_batch_size=True,
            batch_size_kwargs={
                "mode": "power",  # "power" (double until OOM) or "binsearch" (binary search)
                "steps_per_trial": 3,  # Steps to run per batch size trial
                "init_val": 16,  # Initial batch size to try
                "max_trials": 25,  # Maximum number of trials
            },
        ),
        checkpoint_monitor="val/accuracy",
        checkpoint_mode="max",
    )

    # trainer_bs.fit(model2, datamodule=data)

    # ========================================================================
    # Option 3: Both LR and batch size finding (full auto-tuning)
    # ========================================================================
    print("\n" + "=" * 60)
    print("Option 3: Full Auto-Tuning (LR + Batch Size)")
    print("=" * 60)

    model3 = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        metrics=metric_manager,
        logging_config=LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=False,
        ),
        lr=1e-3,
        scheduler="cosine",
    )

    trainer_full = AutoTrainer(
        max_epochs=10,
        accelerator="auto",
        logger=[
            LoggerConfig(
                backend="tensorboard", params={"save_dir": "logs/full_tuning"}
            ),
        ],
        tuner_config=TunerConfig(
            auto_lr=True,
            auto_batch_size=True,
            lr_find_kwargs={
                "min_lr": 1e-6,
                "max_lr": 1.0,
                "num_training": 100,
            },
            batch_size_kwargs={
                "mode": "power",
                "init_val": 16,
            },
        ),
        checkpoint_monitor="val/accuracy",
        checkpoint_mode="max",
    )

    # Run full auto-tuning (batch size first, then LR)
    print("\nRunning full auto-tuning...")
    print("Step 1: Finding optimal batch size...")
    print("Step 2: Finding optimal learning rate...")
    print("Step 3: Training with optimal hyperparameters...\n")

    trainer_full.fit(model3, datamodule=data)
    trainer_full.test(model3, datamodule=data)


if __name__ == "__main__":
    main()
