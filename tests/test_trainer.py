"""Tests for the AutoTrainer class."""

import pytorch_lightning as pl

from autotimm.training.trainer import AutoTrainer, TunerConfig


def test_auto_trainer_returns_pl_trainer():
    trainer = AutoTrainer(max_epochs=1, logger=False, enable_checkpointing=False)
    assert isinstance(trainer, pl.Trainer)
    assert trainer.max_epochs == 1


def test_auto_trainer_checkpoint_callback_with_monitor():
    trainer = AutoTrainer(
        max_epochs=1,
        logger=False,
        checkpoint_monitor="val/accuracy",
        checkpoint_mode="max",
    )
    checkpoint_cbs = [
        cb for cb in trainer.callbacks if isinstance(cb, pl.callbacks.ModelCheckpoint)
    ]
    assert len(checkpoint_cbs) >= 1
    assert checkpoint_cbs[0].monitor == "val/accuracy"
    assert checkpoint_cbs[0].mode == "max"


def test_auto_trainer_no_custom_checkpoint_callback_without_monitor():
    trainer = AutoTrainer(
        max_epochs=1, logger=False, enable_checkpointing=True, checkpoint_monitor=None
    )
    # No checkpoint_monitor specified, so no custom checkpoint with monitor should be added
    # Lightning may still add a default one
    checkpoint_cbs = [
        cb for cb in trainer.callbacks if isinstance(cb, pl.callbacks.ModelCheckpoint)
    ]
    # Check that none of our custom checkpoints (with monitor) were added
    custom_cbs = [cb for cb in checkpoint_cbs if cb.monitor is not None]
    assert len(custom_cbs) == 0


def test_auto_trainer_no_checkpoint():
    trainer = AutoTrainer(max_epochs=1, logger=False, enable_checkpointing=False)
    assert isinstance(trainer, pl.Trainer)


class TestTunerConfig:
    """Tests for TunerConfig dataclass."""

    def test_default_config(self):
        """Test that default TunerConfig enables both auto_lr and auto_batch_size."""
        config = TunerConfig()
        assert config.auto_lr is True
        assert config.auto_batch_size is True
        # Default kwargs should be populated
        assert "min_lr" in config.lr_find_kwargs
        assert "max_lr" in config.lr_find_kwargs
        assert "mode" in config.batch_size_kwargs
        assert "init_val" in config.batch_size_kwargs

    def test_valid_config(self):
        config = TunerConfig(auto_lr=True, auto_batch_size=True)
        assert config.auto_lr is True
        assert config.auto_batch_size is True
        # With new defaults, kwargs should be populated
        assert isinstance(config.lr_find_kwargs, dict)
        assert isinstance(config.batch_size_kwargs, dict)

    def test_config_with_kwargs(self):
        config = TunerConfig(
            auto_lr=True,
            auto_batch_size=True,
            lr_find_kwargs={"min_lr": 1e-6, "max_lr": 1.0},
            batch_size_kwargs={"mode": "power", "init_val": 16},
        )
        assert config.lr_find_kwargs == {"min_lr": 1e-6, "max_lr": 1.0}
        assert config.batch_size_kwargs == {"mode": "power", "init_val": 16}

    def test_only_lr_finder(self):
        config = TunerConfig(auto_lr=True, auto_batch_size=False)
        assert config.auto_lr is True
        assert config.auto_batch_size is False

    def test_only_batch_size_finder(self):
        config = TunerConfig(auto_lr=False, auto_batch_size=True)
        assert config.auto_lr is False
        assert config.auto_batch_size is True


def test_auto_trainer_with_tuner_config():
    config = TunerConfig(
        auto_lr=True,
        auto_batch_size=False,
        lr_find_kwargs={"min_lr": 1e-5, "max_lr": 0.1},
    )
    trainer = AutoTrainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        tuner_config=config,
    )
    assert trainer.tuner_config is config
    assert trainer.tuner_config.auto_lr is True
    assert trainer.tuner_config.auto_batch_size is False


def test_auto_trainer_without_tuner_config():
    """Test that AutoTrainer creates default TunerConfig when none is provided."""
    import torch

    trainer = AutoTrainer(max_epochs=1, logger=False, enable_checkpointing=False)
    # Default behavior: auto-tuning enabled
    assert trainer.tuner_config is not None
    assert isinstance(trainer.tuner_config, TunerConfig)
    assert trainer.tuner_config.auto_lr is True
    # auto_batch_size is only enabled on GPU (CUDA) machines
    assert trainer.tuner_config.auto_batch_size is torch.cuda.is_available()


def test_auto_trainer_disable_tuning():
    """Test that AutoTrainer can disable tuning by passing False."""
    trainer = AutoTrainer(
        max_epochs=1, logger=False, enable_checkpointing=False, tuner_config=False
    )
    assert trainer.tuner_config is None


def test_auto_trainer_fast_dev_run_disables_tuning():
    """Test that fast_dev_run automatically disables auto-tuning."""
    trainer = AutoTrainer(max_epochs=1, logger=False, fast_dev_run=True)
    assert trainer.tuner_config is None
