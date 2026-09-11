# Logger Integration Issues

Problems with TensorBoard, WandB, and other logging backends.

## Weights & Biases (WandB) Issues

### Login Issues

```python
# 1. Login issues
import wandb
wandb.login(key="your_api_key")

# 2. Disable online sync for offline training
from autotimm import AutoTrainer, LoggerConfig

trainer = AutoTrainer(
    max_epochs=10,
    logger=[
        LoggerConfig(
            backend="wandb",
            params={"project": "my-project", "offline": True},
        )
    ],
)

# 3. Resume run
trainer = AutoTrainer(
    logger=[
        LoggerConfig(
            backend="wandb",
            params={"project": "my-project", "id": "run_id", "resume": "must"},
        )
    ],
)
```

## TensorBoard Issues

```python
import autotimm as at  # recommended alias
from autotimm import LoggerConfig, LoggerManager

# Specify custom log directory
logger_manager = LoggerManager(
    configs=[
        LoggerConfig(
            backend="tensorboard",
            params={"save_dir": "./custom_logs", "name": "run_1"},
        ),
    ]
)

# View logs
# tensorboard --logdir ./custom_logs

# If port is occupied
# tensorboard --logdir ./custom_logs --port 6007
```

## Related Issues

- [Installation](../environment/installation.md) - Missing logger dependencies
- [Reproducibility](reproducibility.md) - Logging reproducibility info
