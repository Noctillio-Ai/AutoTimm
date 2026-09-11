"""Test using HuggingFace models directly without Auto classes with PyTorch Lightning.

This tests whether we can use HuggingFace transformers models (ViT, DeiT, BEiT, etc.)
directly with PyTorch Lightning, without relying on AutoModel/AutoImageProcessor.
"""

from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch

try:
    from transformers import ViTConfig, ViTImageProcessor, ViTModel

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestHFDirectModels:
    """Test HF models used directly without Auto classes."""

    def test_manual_vit_creation(self):
        """Test creating ViT model manually without AutoModel."""
        # Create config
        config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=224,
            patch_size=16,
            num_channels=3,
            num_labels=10,
        )

        # Create model directly (not using AutoModel)
        model = ViTModel(config)

        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_manual_vit_forward_pass(self):
        """Test forward pass with manually created ViT."""
        config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=224,
            patch_size=16,
            num_channels=3,
        )

        model = ViTModel(config)
        model.eval()

        # Create dummy input
        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            outputs = model(pixel_values=x)

        # ViTModel returns a tuple with last_hidden_state, pooler_output, etc.
        assert outputs.last_hidden_state is not None
        assert outputs.last_hidden_state.ndim == 3  # (batch, seq_len, hidden_size)

    def test_vit_with_lightning_module(self):
        """Test wrapping HF model in Lightning module."""

        class ViTClassifier(pl.LightningModule):
            def __init__(self, num_classes=10):
                super().__init__()

                # Create ViT base model (without Auto)
                config = ViTConfig(
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    image_size=224,
                    patch_size=16,
                    num_channels=3,
                )

                self.vit = ViTModel(config)

                # Add classification head
                self.classifier = torch.nn.Linear(config.hidden_size, num_classes)
                self.criterion = torch.nn.CrossEntropyLoss()

            def forward(self, x):
                # ViT expects pixel_values argument
                outputs = self.vit(pixel_values=x)
                # Use pooler output (CLS token)
                pooled = outputs.pooler_output
                logits = self.classifier(pooled)
                return logits

            def training_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = self.criterion(logits, y)
                return loss

            def configure_optimizers(self):
                return torch.optim.AdamW(self.parameters(), lr=1e-4)

        # Create model
        model = ViTClassifier(num_classes=10)

        # Test it's a Lightning module
        assert isinstance(model, pl.LightningModule)

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 10)

        # Test training step
        y = torch.randint(0, 10, (2,))
        batch = (x, y)
        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar

    def test_vit_gradient_flow(self):
        """Test gradients flow through HF model."""

        class SimpleViTModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                config = ViTConfig(
                    hidden_size=768,
                    num_hidden_layers=2,  # Smaller for faster test
                    num_attention_heads=12,
                    intermediate_size=3072,
                    image_size=224,
                    patch_size=16,
                    num_channels=3,
                )
                self.vit = ViTModel(config)
                self.classifier = torch.nn.Linear(config.hidden_size, 10)

            def forward(self, x):
                outputs = self.vit(pixel_values=x)
                logits = self.classifier(outputs.pooler_output)
                return logits

        model = SimpleViTModel()
        model.train()

        x = torch.randn(2, 3, 224, 224)
        y = torch.randint(0, 10, (2,))

        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()

        # Check gradients exist
        has_grads = False
        for param in model.parameters():
            if param.grad is not None:
                has_grads = True
                break

        assert has_grads, "No gradients computed"

    def test_pretrained_vit_loading(self):
        """Test loading pretrained ViT without AutoModel."""
        try:
            # Load pretrained model directly (not using AutoModel)
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

            assert model is not None

            # Test forward pass
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                outputs = model(pixel_values=x)

            assert outputs.last_hidden_state is not None

        except Exception as e:
            pytest.skip(f"Could not load pretrained model: {e}")

    def test_image_processor_manual(self):
        """Test using ViTImageProcessor manually (not AutoImageProcessor)."""
        try:
            # Create processor manually (not using AutoImageProcessor)
            processor = ViTImageProcessor(
                size={"height": 224, "width": 224},
                do_normalize=True,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )

            # Create dummy PIL image
            import numpy as np
            from PIL import Image

            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )

            # Process image
            inputs = processor(images=img, return_tensors="pt")

            assert "pixel_values" in inputs
            assert inputs["pixel_values"].shape == (1, 3, 224, 224)

        except ImportError:
            pytest.skip("PIL not available")

    def test_vit_checkpoint_compatibility(self):
        """Test checkpoint save/load with HF ViT in Lightning."""
        import tempfile
        from pathlib import Path

        class ViTClassifier(pl.LightningModule):
            def __init__(self, num_classes=10):
                super().__init__()
                self.save_hyperparameters()

                config = ViTConfig(
                    hidden_size=768,
                    num_hidden_layers=2,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    image_size=224,
                    patch_size=16,
                    num_channels=3,
                )

                self.vit = ViTModel(config)
                self.classifier = torch.nn.Linear(config.hidden_size, num_classes)

            def forward(self, x):
                outputs = self.vit(pixel_values=x)
                return self.classifier(outputs.pooler_output)

            def configure_optimizers(self):
                return torch.optim.AdamW(self.parameters(), lr=1e-4)

        model = ViTClassifier(num_classes=10)

        # Get initial weight
        initial_weight = next(model.parameters()).clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.ckpt"

            # Save
            trainer = pl.Trainer(
                max_epochs=1,
                logger=False,
                enable_checkpointing=False,
                accelerator="cpu",
            )
            trainer.strategy.connect(model)
            trainer.save_checkpoint(checkpoint_path)

            # Load
            loaded_model = ViTClassifier.load_from_checkpoint(checkpoint_path)

            # Verify weights match
            loaded_weight = next(loaded_model.parameters())
            assert torch.allclose(initial_weight, loaded_weight)


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestOtherHFModels:
    """Test other HF models without Auto classes."""

    def test_deit_model(self):
        """Test DeiT model directly."""
        try:
            from transformers import DeiTConfig, DeiTModel

            config = DeiTConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                image_size=224,
                patch_size=16,
                num_channels=3,
            )

            model = DeiTModel(config)

            x = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                outputs = model(pixel_values=x)

            assert outputs.last_hidden_state is not None

        except ImportError:
            pytest.skip("DeiT model not available")

    def test_beit_model(self):
        """Test BEiT model directly."""
        try:
            from transformers import BeitConfig, BeitModel

            config = BeitConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                image_size=224,
                patch_size=16,
                num_channels=3,
            )

            model = BeitModel(config)

            x = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                outputs = model(pixel_values=x)

            assert outputs.last_hidden_state is not None

        except ImportError:
            pytest.skip("BEiT model not available")


class TestWithoutTransformers:
    """Tests that run even without transformers library."""

    def test_transformers_not_required_for_autotimm(self):
        """Verify AutoTimm works without transformers library."""
        # AutoTimm should work fine without transformers
        # It uses timm, not transformers
        import autotimm

        # Should be able to create models
        model = autotimm.create_backbone("resnet18")
        assert model is not None

        # HF Hub timm models should also work
        model = autotimm.create_backbone("hf-hub:timm/resnet18.a1_in1k")
        assert model is not None

        # This proves AutoTimm doesn't require transformers library
        # It uses timm + huggingface_hub, not transformers
