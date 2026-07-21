"""
Tests for reproducibility features: seeding and deterministic mode.
"""

import pickle

import pytest
import torch
import numpy as np
from autotimm import AutoTrainer, seed_everything
from autotimm.core.utils import safe_torch_load
from autotimm.tasks.classification import ImageClassifier


class TestSeedEverything:
    """Test the seed_everything utility function."""

    def test_seed_everything_basic(self):
        """Test basic seeding functionality."""
        seed_everything(42)

        # Generate random numbers
        torch_rand1 = torch.rand(5)
        np_rand1 = np.random.rand(5)

        # Reset seed
        seed_everything(42)

        # Should get same random numbers
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)

        assert torch.allclose(torch_rand1, torch_rand2)
        assert np.allclose(np_rand1, np_rand2)

    def test_seed_everything_deterministic_mode(self):
        """Test deterministic mode setting."""
        seed_everything(42, deterministic=True)

        # Check deterministic flags are set
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

        # Non-deterministic mode
        seed_everything(42, deterministic=False)
        assert torch.backends.cudnn.benchmark is True

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        seed_everything(42)
        rand1 = torch.rand(10)

        seed_everything(123)
        rand2 = torch.rand(10)

        # Should be different
        assert not torch.allclose(rand1, rand2)


class TestSafeTorchLoad:
    """Tests for restricted checkpoint loading."""

    def test_retries_unsafe_load_only_after_unpickling_rejection(self, monkeypatch):
        calls = []

        def load_checkpoint(*args, **kwargs):
            calls.append(kwargs["weights_only"])
            if kwargs["weights_only"]:
                raise pickle.UnpicklingError("unsupported checkpoint metadata")
            return {"state_dict": {}}

        monkeypatch.setattr(torch, "load", load_checkpoint)

        assert safe_torch_load("legacy.ckpt") == {"state_dict": {}}
        assert calls == [True, False]

    def test_missing_file_is_not_retried_with_unsafe_unpickling(self, monkeypatch):
        calls = []

        def missing_file(*args, **kwargs):
            calls.append(kwargs["weights_only"])
            raise FileNotFoundError("checkpoint does not exist")

        monkeypatch.setattr(torch, "load", missing_file)

        with pytest.raises(FileNotFoundError, match="checkpoint does not exist"):
            safe_torch_load("missing.ckpt")

        assert calls == [True]


class TestModelLevelSeeding:
    """Test seeding at the model level."""

    def test_default_seed(self):
        """Test that models use default seed=None."""
        model1 = ImageClassifier(
            backbone="resnet18", num_classes=10, compile_model=False
        )
        initial_weights1 = model1.head.fc.weight.clone()

        # Create another model with default seed
        model2 = ImageClassifier(
            backbone="resnet18", num_classes=10, compile_model=False
        )
        initial_weights2 = model2.head.fc.weight.clone()

        # Should have different initialization
        assert not torch.allclose(initial_weights1, initial_weights2, rtol=1e-4)

    def test_custom_seed(self):
        """Test custom seed values."""
        model1 = ImageClassifier(
            backbone="resnet18", num_classes=10, seed=42, compile_model=False
        )
        weights1 = model1.head.fc.weight.clone()

        model2 = ImageClassifier(
            backbone="resnet18", num_classes=10, seed=42, compile_model=False
        )
        weights2 = model2.head.fc.weight.clone()

        # Same seed = same initialization
        assert torch.allclose(weights1, weights2, rtol=1e-4)

        # Different seed
        model3 = ImageClassifier(
            backbone="resnet18", num_classes=10, seed=123, compile_model=False
        )
        weights3 = model3.head.fc.weight.clone()

        # Should be different
        assert not torch.allclose(weights1, weights3, rtol=1e-4)

    def test_disable_seeding(self):
        """Test disabling seeding with seed=None."""
        model1 = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            seed=None,
            deterministic=False,
            compile_model=False,
        )
        weights1 = model1.head.fc.weight.clone()

        model2 = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            seed=None,
            deterministic=False,
            compile_model=False,
        )
        weights2 = model2.head.fc.weight.clone()

        # Without seeding, should get different initializations
        # (very unlikely to be the same)
        assert not torch.allclose(weights1, weights2, rtol=1e-4)

    def test_deterministic_parameter(self):
        """Test deterministic parameter."""
        # Create model with deterministic=True
        _model1 = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            seed=42,
            deterministic=True,
            compile_model=False,
        )

        # Deterministic mode should be enabled
        assert torch.backends.cudnn.deterministic is True

        # Create model with deterministic=False
        _model2 = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            seed=42,
            deterministic=False,
            compile_model=False,
        )

        # Benchmark should be enabled for faster training
        assert torch.backends.cudnn.benchmark is True


class TestTrainerLevelSeeding:
    """Test seeding at the trainer level."""

    def test_trainer_default_seed(self):
        """Test that AutoTrainer uses default seed=None."""
        # Just verify it doesn't crash - AutoTrainer handles seeding internally
        trainer = AutoTrainer(max_epochs=1, fast_dev_run=True)

        # Verify trainer was created successfully
        assert trainer is not None

    def test_trainer_custom_seed(self):
        """Test custom seed in AutoTrainer."""
        # Verify custom seed doesn't crash
        trainer = AutoTrainer(max_epochs=1, seed=123, fast_dev_run=True)

        assert trainer is not None

    def test_trainer_deterministic(self):
        """Test deterministic parameter in AutoTrainer."""
        trainer = AutoTrainer(
            max_epochs=1, seed=42, deterministic=True, fast_dev_run=True
        )

        # Verify trainer was created successfully
        assert trainer is not None

    def test_trainer_use_autotimm_seeding(self):
        """Test use_autotimm_seeding parameter."""
        # Default: use Lightning's seeding
        trainer1 = AutoTrainer(max_epochs=1, seed=42, fast_dev_run=True)
        assert trainer1 is not None

        # Use AutoTimm's custom seeding
        trainer2 = AutoTrainer(
            max_epochs=1, seed=42, use_autotimm_seeding=True, fast_dev_run=True
        )
        assert trainer2 is not None

    def test_trainer_disable_seeding(self):
        """Test disabling seeding in AutoTrainer."""
        trainer = AutoTrainer(max_epochs=1, seed=None, fast_dev_run=True)

        assert trainer is not None


class TestReproducibleTraining:
    """Test end-to-end reproducible training."""

    @pytest.fixture
    def simple_data(self):
        """Create simple dummy data for testing."""
        from torch.utils.data import TensorDataset, DataLoader

        # Create dummy data
        X = torch.randn(100, 3, 32, 32)
        y = torch.randint(0, 10, (100,))

        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=16)
        val_loader = DataLoader(dataset, batch_size=16)

        return train_loader, val_loader

    def test_reproducible_training_same_seed(self, simple_data):
        """Test that same seed produces same training results."""
        train_loader, val_loader = simple_data

        # First training run
        model1 = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            seed=42,
            deterministic=True,
            compile_model=False,
            metrics=None,
        )

        trainer1 = AutoTrainer(
            max_epochs=1,
            seed=42,
            deterministic=True,
            fast_dev_run=5,
            logger=False,
            enable_checkpointing=False,
        )

        trainer1.fit(model1, train_loader, val_loader)
        weights1 = model1.head.fc.weight.clone()

        # Second training run with same seed
        model2 = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            seed=42,
            deterministic=True,
            compile_model=False,
            metrics=None,
        )

        trainer2 = AutoTrainer(
            max_epochs=1,
            seed=42,
            deterministic=True,
            fast_dev_run=5,
            logger=False,
            enable_checkpointing=False,
        )

        trainer2.fit(model2, train_loader, val_loader)
        weights2 = model2.head.fc.weight.clone()

        # Weights should be very similar (allowing for small numerical differences)
        assert torch.allclose(weights1, weights2, rtol=1e-3, atol=1e-5)

    def test_different_seeds_produce_different_results(self, simple_data):
        """Test that different seeds produce different training results."""
        train_loader, val_loader = simple_data

        # Training with seed=42
        model1 = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            seed=42,
            compile_model=False,
            metrics=None,
        )

        trainer1 = AutoTrainer(
            max_epochs=1,
            seed=42,
            fast_dev_run=5,
            logger=False,
            enable_checkpointing=False,
        )

        trainer1.fit(model1, train_loader, val_loader)
        weights1 = model1.head.fc.weight.clone()

        # Training with seed=123
        model2 = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            seed=123,
            compile_model=False,
            metrics=None,
        )

        trainer2 = AutoTrainer(
            max_epochs=1,
            seed=123,
            fast_dev_run=5,
            logger=False,
            enable_checkpointing=False,
        )

        trainer2.fit(model2, train_loader, val_loader)
        weights2 = model2.head.fc.weight.clone()

        # Different seeds should produce different weights
        assert not torch.allclose(weights1, weights2, rtol=1e-3, atol=1e-5)
