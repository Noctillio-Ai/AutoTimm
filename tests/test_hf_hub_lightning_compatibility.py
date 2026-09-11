"""Test compatibility between HF Hub models and PyTorch Lightning."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import pytorch_lightning as pl

from autotimm import (
    ImageClassifier,
    MetricConfig,
    ObjectDetector,
    SemanticSegmentor,
)


class TestHFHubLightningIntegration:
    """Test that HF Hub models work with PyTorch Lightning features."""

    def test_lightning_module_creation(self):
        """Test that HF Hub models create valid Lightning modules."""
        metrics = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass", "num_classes": 10},
                stages=["train", "val"],
                prog_bar=True,
            ),
        ]

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=metrics,
            lr=1e-3,
        )

        # Verify it's a Lightning module
        assert isinstance(model, pl.LightningModule)
        assert hasattr(model, "training_step")
        assert hasattr(model, "validation_step")
        assert hasattr(model, "configure_optimizers")

    def test_forward_pass_with_hf_hub(self):
        """Test forward pass with HF Hub backbone."""
        metrics = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass", "num_classes": 10},
                stages=["train", "val"],
                prog_bar=True,
            ),
        ]

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=metrics,
        )

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)

        assert output.shape == (2, 10)
        assert not torch.isnan(output).any()

    def test_configure_optimizers(self):
        """Test that optimizer configuration works with HF Hub models."""
        metrics = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass", "num_classes": 10},
                stages=["train", "val"],
                prog_bar=True,
            ),
        ]

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=metrics,
            lr=1e-3,
            optimizer="adamw",
            scheduler=None,  # No scheduler to avoid needing trainer setup
        )

        # Configure optimizers (without scheduler, no trainer needed)
        optimizer_config = model.configure_optimizers()

        assert "optimizer" in optimizer_config
        assert isinstance(optimizer_config["optimizer"], torch.optim.Optimizer)

    @pytest.mark.slow
    def test_checkpoint_save_load(self):
        """Test that HF Hub models can be saved and loaded via checkpoints."""
        metrics = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass", "num_classes": 10},
                stages=["train", "val"],
                prog_bar=True,
            ),
        ]

        # Create model with HF Hub backbone
        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=metrics,
            lr=1e-3,
        )

        # Get initial parameters
        initial_params = {
            name: param.clone() for name, param in model.named_parameters()
        }

        # Create a temporary checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.ckpt"

            # Save checkpoint
            trainer = pl.Trainer(
                max_epochs=1,
                logger=False,
                enable_checkpointing=False,
                accelerator="cpu",
            )
            trainer.strategy.connect(model)
            trainer.save_checkpoint(checkpoint_path)

            # Load checkpoint into new model
            loaded_model = ImageClassifier.load_from_checkpoint(
                checkpoint_path,
                backbone="hf-hub:timm/resnet18.a1_in1k",
                metrics=metrics,
            )

            # Verify parameters match
            for name, param in loaded_model.named_parameters():
                assert torch.allclose(param, initial_params[name])

    @pytest.mark.slow
    def test_state_dict_compatibility(self):
        """Test state dict save/load with HF Hub models."""
        metrics = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass", "num_classes": 10},
                stages=["train", "val"],
                prog_bar=True,
            ),
        ]

        # Create model
        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=metrics,
        )

        # Save state dict
        state_dict = model.state_dict()

        # Create new model and load state
        new_model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=metrics,
        )
        new_model.load_state_dict(state_dict)

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output1 = model(x)
        output2 = new_model(x)

        assert torch.allclose(output1, output2, rtol=1e-5)

    def test_training_step(self):
        """Test training step execution with HF Hub model."""
        metrics = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass", "num_classes": 10},
                stages=["train", "val"],
                prog_bar=True,
            ),
        ]

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=metrics,
        )

        # Create dummy batch
        x = torch.randn(4, 3, 224, 224)
        y = torch.randint(0, 10, (4,))
        batch = (x, y)

        # Execute training step
        model.train()
        loss = model.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        assert loss > 0

    def test_validation_step(self):
        """Test validation step execution with HF Hub model."""
        metrics = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass", "num_classes": 10},
                stages=["train", "val"],
                prog_bar=True,
            ),
        ]

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=metrics,
        )

        # Create dummy batch
        x = torch.randn(4, 3, 224, 224)
        y = torch.randint(0, 10, (4,))
        batch = (x, y)

        # Execute validation step
        model.eval()
        with torch.no_grad():
            model.validation_step(batch, 0)

        # Should not raise any errors

    def test_device_placement(self):
        """Test that models can be moved to different devices."""
        metrics = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass", "num_classes": 10},
                stages=["train", "val"],
                prog_bar=True,
            ),
        ]

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=metrics,
        )

        # Move to CPU (already there, but test the operation)
        model.to("cpu")

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)

        assert output.device.type == "cpu"

        # If CUDA available, test GPU placement
        if torch.cuda.is_available():
            model.to("cuda")
            x = x.to("cuda")
            output = model(x)
            assert output.device.type == "cuda"

    @pytest.mark.slow
    def test_mixed_precision_compatibility(self):
        """Test that HF Hub models work with mixed precision training."""
        metrics = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass", "num_classes": 10},
                stages=["train", "val"],
                prog_bar=True,
            ),
        ]

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=metrics,
        )

        # Test with autocast
        x = torch.randn(2, 3, 224, 224)

        with torch.cuda.amp.autocast(enabled=True):
            output = model(x)

        assert output.shape == (2, 10)
        assert not torch.isnan(output).any()

    def test_gradient_computation(self):
        """Test that gradients flow correctly through HF Hub backbone."""
        metrics = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass", "num_classes": 10},
                stages=["train", "val"],
                prog_bar=True,
            ),
        ]

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=metrics,
        )

        # Forward pass
        x = torch.randn(2, 3, 224, 224)
        y = torch.randint(0, 10, (2,))
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                assert not torch.isnan(param.grad).any()

        assert has_gradients, "No gradients computed"


class TestHFHubAllTasks:
    """Test HF Hub models with all AutoTimm tasks."""

    def test_segmentation_task(self):
        """Test HF Hub model with semantic segmentation."""
        metrics = [
            MetricConfig(
                name="iou",
                backend="torchmetrics",
                metric_class="JaccardIndex",
                params={"task": "multiclass", "num_classes": 19, "average": "macro"},
                stages=["val"],
                prog_bar=True,
            ),
        ]

        model = SemanticSegmentor(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=19,
            head_type="fcn",
            metrics=metrics,
        )

        assert isinstance(model, pl.LightningModule)

        # Test forward pass
        x = torch.randn(1, 3, 256, 256)
        output = model(x)
        # Output shape will be downsampled from input based on backbone
        assert output.shape[0] == 1  # Batch size
        assert output.shape[1] == 19  # Number of classes
        assert output.ndim == 4  # (B, C, H, W)

    def test_detection_task(self):
        """Test HF Hub model with object detection."""
        # Use empty metrics to avoid pycocotools dependency in test
        model = ObjectDetector(
            backbone="hf-hub:timm/resnet50.a1_in1k",  # ResNet50 has standard feature structure
            num_classes=80,
            metrics=[],  # No metrics to avoid dependencies
        )

        assert isinstance(model, pl.LightningModule)

        # Verify model has expected components
        assert hasattr(model, "backbone")
        assert hasattr(model, "fpn")
        assert hasattr(model, "head")

        # Model should be callable (even if we don't test full inference here)
        assert callable(model)


class TestHFHubHyperparameters:
    """Test hyperparameter saving/loading with HF Hub models."""

    def test_hparams_saved_correctly(self):
        """Test that hyperparameters include HF Hub backbone info."""
        metrics = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass", "num_classes": 10},
                stages=["train", "val"],
                prog_bar=True,
            ),
        ]

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=metrics,
            lr=1e-3,
            optimizer="adamw",
        )

        # Check hparams
        assert "backbone_name" in model.hparams
        assert model.hparams["backbone_name"] == "hf-hub:timm/resnet18.a1_in1k"
        assert model.hparams["num_classes"] == 10
        assert model.hparams["lr"] == 1e-3
        assert model.hparams["optimizer"] == "adamw"

    @pytest.mark.slow
    def test_hparams_loaded_from_checkpoint(self):
        """Test that hyperparameters are restored from checkpoint."""
        metrics = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass", "num_classes": 10},
                stages=["train", "val"],
                prog_bar=True,
            ),
        ]

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=metrics,
            lr=2e-3,
            weight_decay=1e-5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.ckpt"

            trainer = pl.Trainer(
                max_epochs=1,
                logger=False,
                enable_checkpointing=False,
                accelerator="cpu",
            )
            trainer.strategy.connect(model)
            trainer.save_checkpoint(checkpoint_path)

            # Load and verify hparams
            loaded_model = ImageClassifier.load_from_checkpoint(
                checkpoint_path,
                backbone="hf-hub:timm/resnet18.a1_in1k",
                metrics=metrics,
            )

            assert (
                loaded_model.hparams["backbone_name"] == "hf-hub:timm/resnet18.a1_in1k"
            )
            assert loaded_model.hparams["lr"] == 2e-3
            assert loaded_model.hparams["weight_decay"] == 1e-5


class TestHFHubDistributedCompatibility:
    """Test HF Hub models with distributed training features."""

    def test_model_serialization(self):
        """Test that models can be serialized (required for DDP)."""
        import pickle

        metrics = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass", "num_classes": 10},
                stages=["train", "val"],
                prog_bar=True,
            ),
        ]

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=metrics,
        )

        # Serialize and deserialize
        serialized = pickle.dumps(model)
        deserialized = pickle.loads(serialized)

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output1 = model(x)
        output2 = deserialized(x)

        # Should produce same output
        assert torch.allclose(output1, output2, rtol=1e-5)

    def test_parameter_broadcast_compatibility(self):
        """Test that model parameters can be broadcast (for DDP init)."""
        metrics = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass", "num_classes": 10},
                stages=["train", "val"],
                prog_bar=True,
            ),
        ]

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=metrics,
        )

        # Verify all parameters are tensors (required for broadcast)
        for name, param in model.named_parameters():
            assert isinstance(param, torch.Tensor)
            assert param.is_contiguous() or param.is_sparse
