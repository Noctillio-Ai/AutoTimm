"""Demonstrate multi-GPU and distributed training.

This example demonstrates:
- Training on multiple GPUs using DDP (Distributed Data Parallel)
- Configuring accelerator and devices
- Using different distributed strategies
- Scaling batch size for multi-GPU training
- MetricManager for metric configuration

Usage:
    python examples/multi_gpu_training.py
"""

from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    LoggingConfig,
    MetricConfig,
    MetricManager,
)


def main():
    # Data - adjust batch_size for multi-GPU
    # Effective batch size = batch_size * num_gpus * accumulate_grad_batches
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=64,  # Per-GPU batch size
        num_workers=4,  # Per-GPU workers
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
        MetricConfig(
            name="f1",
            backend="torchmetrics",
            metric_class="F1Score",
            params={"task": "multiclass", "average": "macro"},
            stages=["val", "test"],
        ),
    ]

    # Create MetricManager
    metric_manager = MetricManager(configs=metric_configs, num_classes=10)

    # Model
    model = ImageClassifier(
        backbone="resnet50",
        num_classes=10,
        metrics=metric_manager,
        logging_config=LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=True,
        ),
        lr=1e-3,  # Base learning rate - may need to scale with num_gpus
        scheduler="cosine",
        label_smoothing=0.1,
    )

    # ========================================================================
    # Option 1: Automatic GPU detection (recommended)
    # ========================================================================
    print("=" * 60)
    print("Option 1: Automatic GPU detection")
    print("=" * 60)

    # accelerator="auto" and devices="auto" will:
    # - Use GPU if available, else CPU
    # - Use all available GPUs
    trainer_auto = AutoTrainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        logger=[
            LoggerConfig(backend="tensorboard", params={"save_dir": "logs/multi_gpu"}),
        ],
        checkpoint_monitor="val/accuracy",
        checkpoint_mode="max",
    )

    print(f"  Accelerator: {trainer_auto.accelerator}")
    print(f"  Devices: {trainer_auto.num_devices}")

    # ========================================================================
    # Option 2: Specify number of GPUs
    # ========================================================================
    print("\n" + "=" * 60)
    print("Option 2: Specify number of GPUs")
    print("=" * 60)

    # Use 2 specific GPUs
    trainer_2gpu = AutoTrainer(
        max_epochs=10,
        accelerator="gpu",
        devices=2,  # Use 2 GPUs
        logger=[
            LoggerConfig(backend="tensorboard", params={"save_dir": "logs/2gpu"}),
        ],
        checkpoint_monitor="val/accuracy",
        checkpoint_mode="max",
    )

    # Use specific GPU indices
    # trainer_specific = AutoTrainer(
    #     accelerator="gpu",
    #     devices=[0, 2],  # Use GPU 0 and GPU 2
    # )

    # ========================================================================
    # Option 3: Different distributed strategies
    # ========================================================================
    print("\n" + "=" * 60)
    print("Option 3: Distributed strategies")
    print("=" * 60)

    # DDP (Distributed Data Parallel) - default for multi-GPU
    trainer_ddp = AutoTrainer(
        max_epochs=10,
        accelerator="gpu",
        devices="auto",
        strategy="ddp",  # Default for multi-GPU
        logger=[
            LoggerConfig(backend="tensorboard", params={"save_dir": "logs/ddp"}),
        ],
        checkpoint_monitor="val/accuracy",
        checkpoint_mode="max",
    )

    # DDP with find_unused_parameters (for models with unused params)
    # trainer_ddp_unused = AutoTrainer(
    #     accelerator="gpu",
    #     devices="auto",
    #     strategy="ddp_find_unused_parameters_true",
    # )

    # DeepSpeed (if installed) - for very large models
    # trainer_deepspeed = AutoTrainer(
    #     accelerator="gpu",
    #     devices="auto",
    #     strategy="deepspeed_stage_2",
    #     precision="16-mixed",
    # )

    # FSDP (Fully Sharded Data Parallel) - for very large models
    # trainer_fsdp = AutoTrainer(
    #     accelerator="gpu",
    #     devices="auto",
    #     strategy="fsdp",
    #     precision="16-mixed",
    # )

    # ========================================================================
    # Option 4: Gradient accumulation for larger effective batch size
    # ========================================================================
    print("\n" + "=" * 60)
    print("Option 4: Gradient accumulation")
    print("=" * 60)

    # Effective batch size = 64 * 2 GPUs * 4 accumulation = 512
    trainer_accum = AutoTrainer(
        max_epochs=10,
        accelerator="gpu",
        devices=2,
        accumulate_grad_batches=4,  # Accumulate gradients over 4 batches
        gradient_clip_val=1.0,
        precision="bf16-mixed",  # Use mixed precision for memory efficiency
        logger=[
            LoggerConfig(backend="tensorboard", params={"save_dir": "logs/accum"}),
        ],
        checkpoint_monitor="val/accuracy",
        checkpoint_mode="max",
    )

    print("  Per-GPU batch size: 64")
    print("  Number of GPUs: 2")
    print("  Gradient accumulation: 4")
    print(f"  Effective batch size: {64 * 2 * 4}")

    # ========================================================================
    # Option 5: Mixed precision training
    # ========================================================================
    print("\n" + "=" * 60)
    print("Option 5: Mixed precision options")
    print("=" * 60)

    # BF16 mixed precision (recommended for Ampere+ GPUs)
    trainer_bf16 = AutoTrainer(
        max_epochs=10,
        accelerator="gpu",
        devices="auto",
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
    )

    # FP16 mixed precision (for older GPUs)
    trainer_fp16 = AutoTrainer(
        max_epochs=10,
        accelerator="gpu",
        devices="auto",
        precision="16-mixed",
        logger=False,
        enable_checkpointing=False,
    )

    print("  BF16: Best for Ampere+ GPUs (A100, RTX 30xx, RTX 40xx)")
    print("  FP16: Compatible with older GPUs (V100, RTX 20xx)")

    # ========================================================================
    # Run training (uncomment to run)
    # ========================================================================
    print("\n" + "=" * 60)
    print("Running multi-GPU training...")
    print("=" * 60)

    # Use automatic detection for actual training
    trainer_auto.fit(model, datamodule=data)
    trainer_auto.test(model, datamodule=data)


if __name__ == "__main__":
    main()
