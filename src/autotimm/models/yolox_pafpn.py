"""YOLOX Path Aggregation Feature Pyramid Network (PAFPN).

Based on the official YOLOX implementation:
https://github.com/Megvii-BaseDetection/YOLOX
"""

from __future__ import annotations

import torch
from torch import nn

from autotimm.models.csp_darknet import BaseConv, CSPLayer, DWConv


class YOLOXPAFPN(nn.Module):
    """YOLOX Path Aggregation Feature Pyramid Network.

    Combines features from backbone using:
    1. Top-down pathway (FPN-style)
    2. Bottom-up pathway (PAN-style)
    3. CSP blocks for feature fusion

    Args:
        depth: Depth multiplier (controls number of bottleneck blocks)
        width: Width multiplier (controls channel dimensions)
        in_features: Input feature names (default: ["dark3", "dark4", "dark5"])
        in_channels: Input channel dimensions for each feature level
        depthwise: Use depthwise convolutions
        act: Activation function name

    Example:
        >>> # YOLOX-s configuration
        >>> pafpn = YOLOXPAFPN(depth=0.33, width=0.50)
        >>> features = {
        ...     "dark3": torch.randn(1, 128, 80, 80),
        ...     "dark4": torch.randn(1, 256, 40, 40),
        ...     "dark5": torch.randn(1, 512, 20, 20),
        ... }
        >>> outputs = pafpn(features)
        >>> print([o.shape for o in outputs])
        # [torch.Size([1, 128, 80, 80]), torch.Size([1, 256, 40, 40]), torch.Size([1, 512, 20, 20])]
    """

    def __init__(
        self,
        depth: float = 1.0,
        width: float = 1.0,
        in_features: tuple[str, ...] = ("dark3", "dark4", "dark5"),
        in_channels: tuple[int, ...] = (256, 512, 1024),
        depthwise: bool = False,
        act: str = "silu",
    ):
        super().__init__()
        self.in_features = in_features
        Conv = DWConv if depthwise else BaseConv

        # in_channels already represent actual backbone output channels (scaled)
        # All PAFPN outputs will have uniform channels (smallest level)
        self.out_channels = in_channels[0]
        base_depth = max(round(depth * 3), 1)

        # Top-down FPN pathway
        # Reduce channels from C5
        self.lateral_conv0 = BaseConv(in_channels[2], in_channels[1], 1, 1, act=act)
        # C3 + C4 fusion
        self.C3_p4 = CSPLayer(
            2 * in_channels[1],
            in_channels[1],
            round(base_depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # Reduce channels from C4
        self.reduce_conv1 = BaseConv(in_channels[1], in_channels[0], 1, 1, act=act)
        # C2 + C3 fusion
        self.C3_p3 = CSPLayer(
            2 * in_channels[0],
            in_channels[0],
            round(base_depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # Bottom-up PAN pathway
        # P3 -> P4
        self.bu_conv2 = Conv(in_channels[0], in_channels[0], 3, 2, act=act)
        # C3 + C4 fusion
        self.C3_n3 = CSPLayer(
            in_channels[0] + in_channels[1],
            in_channels[1],
            round(base_depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # P4 -> P5
        self.bu_conv1 = Conv(in_channels[1], in_channels[1], 3, 2, act=act)
        # C4 + C5 fusion
        self.C3_n4 = CSPLayer(
            in_channels[1] + in_channels[2],
            in_channels[2],
            round(base_depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Output projection layers to ensure uniform channels
        # Project N3 and N4 to have same channels as P3_out
        self.out_proj_n3 = BaseConv(in_channels[1], in_channels[0], 1, 1, act=act)
        self.out_proj_n4 = BaseConv(in_channels[2], in_channels[0], 1, 1, act=act)

    def forward(self, features: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass through PAFPN.

        Args:
            features: Dictionary with keys from in_features
                - dark3: [B, C3, H/8, W/8]
                - dark4: [B, C4, H/16, W/16]
                - dark5: [B, C5, H/32, W/32]

        Returns:
            List of 3 feature maps [P3, P4, P5] with uniform channels
        """
        # Extract input features
        [feat_c3, feat_c4, feat_c5] = [features[f] for f in self.in_features]

        # Top-down FPN pathway
        # C5 -> P5
        P5 = self.lateral_conv0(feat_c5)  # 1x1 conv to reduce channels
        P5_upsample = self.upsample(P5)
        # C4 + P5_up -> P4
        P5_upsample = torch.cat([P5_upsample, feat_c4], dim=1)
        P4 = self.C3_p4(P5_upsample)  # CSP fusion

        # P4 -> P4_reduced
        P4_reduced = self.reduce_conv1(P4)
        P4_upsample = self.upsample(P4_reduced)
        # C3 + P4_up -> P3
        P4_upsample = torch.cat([P4_upsample, feat_c3], dim=1)
        P3_out = self.C3_p3(P4_upsample)  # CSP fusion

        # Bottom-up PAN pathway
        # P3 -> P3_down
        P3_downsample = self.bu_conv2(P3_out)
        # P3_down + P4 -> N3
        P3_downsample = torch.cat([P3_downsample, P4], dim=1)
        N3 = self.C3_n3(P3_downsample)  # CSP fusion

        # N3 -> N3_down
        N3_downsample = self.bu_conv1(N3)
        # N3_down + feat_c5 -> N4 (use original feat_c5, not reduced P5)
        N3_downsample = torch.cat([N3_downsample, feat_c5], dim=1)
        N4 = self.C3_n4(N3_downsample)  # CSP fusion

        # Project all outputs to uniform channels
        N3_out = self.out_proj_n3(N3)
        N4_out = self.out_proj_n4(N4)

        # Return [P3_out, N3_out, N4_out] - 3 feature levels with uniform channels
        return [P3_out, N3_out, N4_out]


def build_yolox_pafpn(model_name: str = "yolox-s") -> YOLOXPAFPN:
    """Build YOLOXPAFPN with predefined configurations.

    Args:
        model_name: Model variant name

    Returns:
        YOLOXPAFPN neck
    """
    configs = {
        "yolox-nano": {
            "depth": 0.33,
            "width": 0.25,
            "in_channels": (64, 128, 256),
        },
        "yolox-tiny": {
            "depth": 0.33,
            "width": 0.375,
            "in_channels": (96, 192, 384),
        },
        "yolox-s": {
            "depth": 0.33,
            "width": 0.50,
            "in_channels": (128, 256, 512),
        },
        "yolox-m": {
            "depth": 0.67,
            "width": 0.75,
            "in_channels": (192, 384, 768),
        },
        "yolox-l": {
            "depth": 1.0,
            "width": 1.0,
            "in_channels": (256, 512, 1024),
        },
        "yolox-x": {
            "depth": 1.33,
            "width": 1.25,
            "in_channels": (320, 640, 1280),
        },
    }

    if model_name not in configs:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(configs.keys())}"
        )

    config = configs[model_name]
    return YOLOXPAFPN(**config)
