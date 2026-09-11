"""Tests for segmentation heads."""

import torch

from autotimm.heads import ASPP, DeepLabV3PlusHead, FCNHead, MaskHead


class TestASPP:
    """Test ASPP module."""

    def test_aspp_forward(self):
        """Test forward pass."""
        aspp = ASPP(in_channels=2048, out_channels=256, dilation_rates=(6, 12, 18))
        x = torch.randn(2, 2048, 32, 32)

        output = aspp(x)

        assert output.shape == (2, 256, 32, 32)

    def test_aspp_different_dilation_rates(self):
        """Test with different dilation rates."""
        aspp = ASPP(in_channels=512, out_channels=128, dilation_rates=(3, 6, 9))
        x = torch.randn(2, 512, 16, 16)

        output = aspp(x)

        assert output.shape == (2, 128, 16, 16)

    def test_aspp_gradient_flow(self):
        """Test gradient flow."""
        aspp = ASPP(in_channels=256, out_channels=128)
        aspp.eval()  # Set to eval mode to avoid BatchNorm issues with batch_size=1
        x = torch.randn(1, 256, 16, 16, requires_grad=True)

        output = aspp(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestDeepLabV3PlusHead:
    """Test DeepLabV3+ head."""

    def test_deeplabv3plus_forward(self):
        """Test forward pass."""
        in_channels_list = [64, 128, 256, 512]  # C2, C3, C4, C5
        head = DeepLabV3PlusHead(
            in_channels_list=in_channels_list,
            num_classes=19,
            aspp_out_channels=256,
            decoder_channels=256,
        )

        # Create feature maps at different scales
        features = [
            torch.randn(2, 64, 128, 128),  # C2
            torch.randn(2, 128, 64, 64),  # C3
            torch.randn(2, 256, 32, 32),  # C4
            torch.randn(2, 512, 16, 16),  # C5
        ]

        output = head(features)

        # Output should be at 1/4 resolution (C2 resolution)
        assert output.shape == (2, 19, 128, 128)

    def test_deeplabv3plus_different_classes(self):
        """Test with different number of classes."""
        in_channels_list = [64, 128, 256, 512]
        head = DeepLabV3PlusHead(
            in_channels_list=in_channels_list,
            num_classes=21,
        )
        head.eval()  # Set to eval mode

        features = [
            torch.randn(1, 64, 64, 64),
            torch.randn(1, 128, 32, 32),
            torch.randn(1, 256, 16, 16),
            torch.randn(1, 512, 8, 8),
        ]

        output = head(features)

        assert output.shape == (1, 21, 64, 64)

    def test_deeplabv3plus_gradient_flow(self):
        """Test gradient flow."""
        in_channels_list = [64, 128, 256, 512]
        head = DeepLabV3PlusHead(in_channels_list=in_channels_list, num_classes=10)
        head.train()  # Ensure in training mode
        # Use batch_size=2 to avoid BatchNorm issues in training mode

        features = [
            torch.randn(2, 64, 32, 32, requires_grad=True),
            torch.randn(2, 128, 16, 16, requires_grad=True),
            torch.randn(2, 256, 8, 8, requires_grad=True),
            torch.randn(2, 512, 4, 4, requires_grad=True),
        ]

        output = head(features)
        loss = output.sum()
        loss.backward()

        # Only check gradients for features that are actually used by DeepLabV3+
        # DeepLabV3+ uses features[0] (low-level) and features[-1] (high-level)
        assert features[0].grad is not None, "Low-level features should have gradients"
        assert (
            features[-1].grad is not None
        ), "High-level features should have gradients"


class TestFCNHead:
    """Test FCN head."""

    def test_fcn_forward(self):
        """Test forward pass."""
        head = FCNHead(in_channels=512, num_classes=21, intermediate_channels=256)

        # Only uses last feature
        features = [torch.randn(2, 512, 32, 32)]

        output = head(features)

        assert output.shape == (2, 21, 32, 32)

    def test_fcn_multiple_features(self):
        """Test with multiple features (uses only last one)."""
        head = FCNHead(in_channels=512, num_classes=10)

        features = [
            torch.randn(1, 64, 128, 128),
            torch.randn(1, 128, 64, 64),
            torch.randn(1, 256, 32, 32),
            torch.randn(1, 512, 16, 16),  # Only this is used
        ]

        output = head(features)

        assert output.shape == (1, 10, 16, 16)

    def test_fcn_gradient_flow(self):
        """Test gradient flow."""
        head = FCNHead(in_channels=256, num_classes=5)
        features = [torch.randn(1, 256, 32, 32, requires_grad=True)]

        output = head(features)
        loss = output.sum()
        loss.backward()

        assert features[0].grad is not None


class TestMaskHead:
    """Test Mask head."""

    def test_mask_head_forward(self):
        """Test forward pass."""
        head = MaskHead(
            in_channels=256,
            num_classes=80,
            hidden_channels=256,
            num_convs=4,
            mask_size=28,
        )

        # ROI-aligned features
        roi_features = torch.randn(10, 256, 14, 14)

        output = head(roi_features)

        # After deconv, output is 2x the input size
        assert output.shape == (10, 80, 28, 28)

    def test_mask_head_different_sizes(self):
        """Test with different input sizes."""
        head = MaskHead(in_channels=128, num_classes=20, mask_size=14)

        roi_features = torch.randn(5, 128, 7, 7)

        output = head(roi_features)

        assert output.shape == (5, 20, 14, 14)

    def test_mask_head_gradient_flow(self):
        """Test gradient flow."""
        head = MaskHead(in_channels=256, num_classes=10)
        roi_features = torch.randn(3, 256, 14, 14, requires_grad=True)

        output = head(roi_features)
        loss = output.sum()
        loss.backward()

        assert roi_features.grad is not None


class TestHeadsIntegration:
    """Integration tests for segmentation heads."""

    def test_heads_with_timm_features(self):
        """Test heads with feature shapes from typical timm models."""
        # ResNet-like feature channels
        in_channels_list = [256, 512, 1024, 2048]

        # DeepLabV3+
        deeplabv3plus = DeepLabV3PlusHead(
            in_channels_list=in_channels_list,
            num_classes=19,
        )

        features_resnet = [
            torch.randn(2, 256, 128, 128),
            torch.randn(2, 512, 64, 64),
            torch.randn(2, 1024, 32, 32),
            torch.randn(2, 2048, 16, 16),
        ]

        output = deeplabv3plus(features_resnet)
        assert output.shape == (2, 19, 128, 128)

        # FCN
        fcn = FCNHead(in_channels=2048, num_classes=19)
        output_fcn = fcn([features_resnet[-1]])
        assert output_fcn.shape == (2, 19, 16, 16)

    def test_heads_memory_efficient(self):
        """Test that heads don't use excessive memory."""
        import gc

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        in_channels_list = [64, 128, 256, 512]
        head = DeepLabV3PlusHead(in_channels_list=in_channels_list, num_classes=10)
        head.eval()  # Set to eval mode

        features = [
            torch.randn(1, 64, 128, 128),
            torch.randn(1, 128, 64, 64),
            torch.randn(1, 256, 32, 32),
            torch.randn(1, 512, 16, 16),
        ]

        output = head(features)

        # Just verify it completes without OOM
        assert output is not None

    def test_heads_batch_size_independence(self):
        """Test that heads work with different batch sizes."""
        in_channels_list = [64, 128, 256, 512]

        for batch_size in [1, 2, 4, 8]:
            head = DeepLabV3PlusHead(in_channels_list=in_channels_list, num_classes=5)
            # Use eval mode for batch_size=1 due to BatchNorm
            if batch_size == 1:
                head.eval()

            features = [
                torch.randn(batch_size, 64, 32, 32),
                torch.randn(batch_size, 128, 16, 16),
                torch.randn(batch_size, 256, 8, 8),
                torch.randn(batch_size, 512, 4, 4),
            ]

            output = head(features)
            assert output.shape == (batch_size, 5, 32, 32)
