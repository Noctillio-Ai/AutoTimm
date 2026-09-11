"""Test HuggingFace models compatibility with AutoTrainer.

This tests whether HF Hub timm models and HF transformers direct models work
with the AutoTrainer class and its specific features (TunerConfig, LoggerManager, etc.).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch

from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    MetricConfig,
)

try:
    from transformers import ViTConfig, ViTModel

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.fixture
def dummy_data():
    """Create dummy dataset for testing."""

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 32

        def __getitem__(self, idx):
            x = torch.randn(3, 224, 224)
            y = torch.randint(0, 10, (1,)).item()
            return x, y

    dataset = DummyDataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    return train_loader, val_loader


@pytest.fixture
def basic_metrics():
    """Create basic metrics for testing."""
    return [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass", "num_classes": 10},
            stages=["train", "val"],
            prog_bar=True,
        ),
    ]


class TestHFHubWithAutoTrainer:
    """Test HF Hub timm models with AutoTrainer."""

    def test_basic_training(self, dummy_data, basic_metrics):
        """Test basic training with AutoTrainer and HF Hub model."""
        train_loader, val_loader = dummy_data

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=basic_metrics,
            lr=1e-3,
        )

        # Use AutoTrainer (disable auto-tuning for tests)
        trainer = AutoTrainer(
            max_epochs=2,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            tuner_config=False,
        )

        # Train
        trainer.fit(model, train_loader, val_loader)

        assert trainer.current_epoch == 2

    def test_autotrainer_with_checkpointing(self, dummy_data, basic_metrics):
        """Test AutoTrainer checkpoint monitoring with HF Hub model."""
        train_loader, val_loader = dummy_data

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=basic_metrics,
            lr=1e-3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint callback
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=tmpdir,
                filename="best-{epoch:02d}",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                enable_version_counter=False,
            )

            trainer = AutoTrainer(
                max_epochs=2,
                accelerator="cpu",
                logger=False,
                callbacks=[checkpoint_callback],
                enable_checkpointing=True,  # Explicitly enable
                tuner_config=False,
            )

            trainer.fit(model, train_loader, val_loader)

            # Check checkpoint was created
            checkpoint_files = list(Path(tmpdir).glob("*.ckpt"))
            assert len(checkpoint_files) > 0, f"No checkpoints found in {tmpdir}"

    def test_autotrainer_with_early_stopping(self, dummy_data, basic_metrics):
        """Test AutoTrainer with early stopping and HF Hub model."""
        train_loader, val_loader = dummy_data

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=basic_metrics,
            lr=1e-3,
        )

        early_stop = pl.callbacks.EarlyStopping(
            monitor="val/loss",
            patience=10,
            mode="min",
        )

        trainer = AutoTrainer(
            max_epochs=2,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            callbacks=[early_stop],
            tuner_config=False,
        )

        trainer.fit(model, train_loader, val_loader)

        # Should complete without errors
        assert trainer.current_epoch <= 2

    def test_autotrainer_validation_testing(self, dummy_data, basic_metrics):
        """Test AutoTrainer validate/test with HF Hub model."""
        train_loader, val_loader = dummy_data

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=basic_metrics,
            lr=1e-3,
        )

        trainer = AutoTrainer(
            max_epochs=1,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            tuner_config=False,
        )

        # Train
        trainer.fit(model, train_loader, val_loader)

        # Validate
        val_results = trainer.validate(model, val_loader)
        assert isinstance(val_results, list)
        assert len(val_results) > 0

        # Test (ckpt_path=None because checkpointing is disabled)
        test_results = trainer.test(model, val_loader, ckpt_path=None)
        assert isinstance(test_results, list)
        assert len(test_results) > 0

    def test_different_hf_hub_backbones(self, dummy_data, basic_metrics):
        """Test AutoTrainer with different HF Hub backbones."""
        train_loader, val_loader = dummy_data

        backbones = [
            "hf-hub:timm/resnet18.a1_in1k",
            "hf-hub:timm/resnet34.a1_in1k",
        ]

        for backbone in backbones:
            model = ImageClassifier(
                backbone=backbone,
                num_classes=10,
                metrics=basic_metrics,
                lr=1e-3,
            )

            trainer = AutoTrainer(
                max_epochs=1,
                accelerator="cpu",
                logger=False,
                enable_checkpointing=False,
                tuner_config=False,
            )

            trainer.fit(model, train_loader, val_loader)

            assert trainer.current_epoch == 1


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestHFDirectWithAutoTrainer:
    """Test HF transformers direct models with AutoTrainer."""

    def test_vit_classifier_with_autotrainer(self, dummy_data):
        """Test ViT classifier with AutoTrainer."""
        train_loader, val_loader = dummy_data

        class ViTClassifier(pl.LightningModule):
            def __init__(self, num_classes=10):
                super().__init__()
                self.save_hyperparameters()

                config = ViTConfig(
                    hidden_size=384,
                    num_hidden_layers=4,
                    num_attention_heads=6,
                    intermediate_size=1536,
                    image_size=224,
                    patch_size=16,
                    num_channels=3,
                )

                self.vit = ViTModel(config)
                self.classifier = torch.nn.Linear(config.hidden_size, num_classes)
                self.criterion = torch.nn.CrossEntropyLoss()

            def forward(self, x):
                outputs = self.vit(pixel_values=x)
                return self.classifier(outputs.pooler_output)

            def training_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = self.criterion(logits, y)
                self.log("train/loss", loss, prog_bar=True)
                return loss

            def validation_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = self.criterion(logits, y)
                self.log("val/loss", loss, prog_bar=True)

            def configure_optimizers(self):
                return torch.optim.AdamW(self.parameters(), lr=1e-4)

        model = ViTClassifier(num_classes=10)

        trainer = AutoTrainer(
            max_epochs=2,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            tuner_config=False,
        )

        trainer.fit(model, train_loader, val_loader)

        assert trainer.current_epoch == 2

    def test_vit_with_checkpoint_save_load(self, dummy_data):
        """Test ViT with AutoTrainer checkpoint save/load."""
        train_loader, val_loader = dummy_data

        class ViTClassifier(pl.LightningModule):
            def __init__(self, num_classes=10):
                super().__init__()
                self.save_hyperparameters()

                config = ViTConfig(
                    hidden_size=384,
                    num_hidden_layers=2,
                    num_attention_heads=6,
                    intermediate_size=1536,
                    image_size=224,
                    patch_size=16,
                    num_channels=3,
                )

                self.vit = ViTModel(config)
                self.classifier = torch.nn.Linear(config.hidden_size, num_classes)
                self.criterion = torch.nn.CrossEntropyLoss()

            def forward(self, x):
                outputs = self.vit(pixel_values=x)
                return self.classifier(outputs.pooler_output)

            def training_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = self.criterion(logits, y)
                self.log("train/loss", loss)
                return loss

            def validation_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = self.criterion(logits, y)
                self.log("val/loss", loss)

            def configure_optimizers(self):
                return torch.optim.AdamW(self.parameters(), lr=1e-4)

        model = ViTClassifier(num_classes=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=tmpdir,
                filename="best",
                monitor="val/loss",
                mode="min",
            )

            trainer = AutoTrainer(
                max_epochs=2,
                accelerator="cpu",
                logger=False,
                callbacks=[checkpoint_callback],
                tuner_config=False,
            )

            trainer.fit(model, train_loader, val_loader)

            # Check checkpoint exists
            checkpoint_files = list(Path(tmpdir).glob("*.ckpt"))
            assert len(checkpoint_files) > 0

            # Load checkpoint
            checkpoint_path = checkpoint_files[0]
            loaded_model = ViTClassifier.load_from_checkpoint(checkpoint_path)

            assert loaded_model is not None


class TestAutoTrainerFeatures:
    """Test AutoTrainer-specific features with HF models."""

    def test_multiple_callbacks(self, dummy_data, basic_metrics):
        """Test AutoTrainer with multiple callbacks and HF Hub model."""
        train_loader, val_loader = dummy_data

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=basic_metrics,
            lr=1e-3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=tmpdir,
                filename="best",
                monitor="val/loss",
                mode="min",
            )

            early_stop = pl.callbacks.EarlyStopping(
                monitor="val/loss",
                patience=10,
                mode="min",
            )

            progress_bar = pl.callbacks.RichProgressBar()

            trainer = AutoTrainer(
                max_epochs=2,
                accelerator="cpu",
                logger=False,
                callbacks=[checkpoint_callback, early_stop, progress_bar],
                tuner_config=False,
            )

            trainer.fit(model, train_loader, val_loader)

            assert trainer.current_epoch <= 2

    def test_gradient_accumulation(self, dummy_data, basic_metrics):
        """Test AutoTrainer gradient accumulation with HF Hub model."""
        train_loader, val_loader = dummy_data

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=basic_metrics,
            lr=1e-3,
        )

        trainer = AutoTrainer(
            max_epochs=2,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            accumulate_grad_batches=2,
            tuner_config=False,
        )

        trainer.fit(model, train_loader, val_loader)

        assert trainer.current_epoch == 2

    def test_mixed_precision(self, dummy_data, basic_metrics):
        """Test AutoTrainer mixed precision with HF Hub model."""
        train_loader, val_loader = dummy_data

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=basic_metrics,
            lr=1e-3,
        )

        trainer = AutoTrainer(
            max_epochs=2,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            precision="16-mixed" if torch.cuda.is_available() else "32-true",
            tuner_config=False,
        )

        trainer.fit(model, train_loader, val_loader)

        assert trainer.current_epoch == 2

    def test_limit_batches(self, dummy_data, basic_metrics):
        """Test AutoTrainer with limited batches and HF Hub model."""
        train_loader, val_loader = dummy_data

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=basic_metrics,
            lr=1e-3,
        )

        trainer = AutoTrainer(
            max_epochs=1,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            limit_train_batches=2,
            limit_val_batches=2,
            tuner_config=False,
        )

        trainer.fit(model, train_loader, val_loader)

        assert trainer.current_epoch == 1


class TestAutoTrainerWithDataModule:
    """Test AutoTrainer with ImageDataModule and HF models."""

    def test_with_image_datamodule(self, basic_metrics):
        """Test AutoTrainer with ImageDataModule and HF Hub model."""

        # Create minimal dataset structure
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "dataset"
            train_dir = data_dir / "train"
            val_dir = data_dir / "val"

            # Create class directories
            for split_dir in [train_dir, val_dir]:
                for i in range(2):
                    class_dir = split_dir / f"class{i}"
                    class_dir.mkdir(parents=True, exist_ok=True)

                    # Create dummy images
                    for j in range(4):
                        img = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8)
                        from PIL import Image

                        pil_img = Image.fromarray(img.numpy())
                        pil_img.save(class_dir / f"img{j}.jpg")

            # Create datamodule - expects train/val subdirs in data_dir
            datamodule = ImageDataModule(
                data_dir=str(data_dir),
                image_size=224,
                batch_size=2,
                num_workers=0,
            )

            model = ImageClassifier(
                backbone="hf-hub:timm/resnet18.a1_in1k",
                num_classes=2,
                metrics=basic_metrics,
                lr=1e-3,
            )

            trainer = AutoTrainer(
                max_epochs=1,
                accelerator="cpu",
                logger=False,
                enable_checkpointing=False,
                tuner_config=False,
            )

            trainer.fit(model, datamodule=datamodule)

            assert trainer.current_epoch == 1
