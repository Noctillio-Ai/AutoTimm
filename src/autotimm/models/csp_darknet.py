"""CSPDarknet backbone for YOLOX.

Based on the official YOLOX implementation:
https://github.com/Megvii-BaseDetection/YOLOX

CSPDarknet is the backbone used in YOLOX with Cross Stage Partial connections.
"""

from __future__ import annotations

import torch
from torch import nn


class SiLU(nn.Module):
    """SiLU activation (Swish) - inplace for memory efficiency."""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name: str = "silu") -> nn.Module:
    """Get activation function by name."""
    if name == "silu":
        return SiLU()
    elif name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "lrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    else:
        raise ValueError(f"Unsupported activation: {name}")


class BaseConv(nn.Module):
    """Basic convolution block: Conv2d + BatchNorm + Activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int,
        groups: int = 1,
        bias: bool = False,
        act: str = "silu",
    ):
        super().__init__()
        # Same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int = 1,
        act: str = "silu",
    ):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, act=act)

    def forward(self, x):
        return self.pconv(self.dconv(x))


class Focus(nn.Module):
    """Focus module to reduce spatial dimension while increasing channels.

    Extracts patches in a space-to-depth manner:
    W, H, C -> W/2, H/2, 4C
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int = 1,
        stride: int = 1,
        act: str = "silu",
    ):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # Extract 4 patches: top-left, top-right, bottom-left, bottom-right
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            [patch_top_left, patch_bot_left, patch_top_right, patch_bot_right], dim=1
        )
        return self.conv(x)


class Bottleneck(nn.Module):
    """Standard bottleneck block with optional shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        depthwise: bool = False,
        act: str = "silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class SPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, ...] = (5, 9, 13),
        activation: str = "silu",
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer with multiple bottleneck blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
        depthwise: bool = False,
        act: str = "silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels,
                hidden_channels,
                shortcut,
                1.0,
                depthwise,
                act=act,
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class CSPDarknet(nn.Module):
    """CSPDarknet backbone for YOLOX.

    Architecture:
        Stem -> Stage1 -> Stage2 -> Stage3 -> Stage4 -> Stage5

    Output: Multi-scale features from Stage3, Stage4, Stage5 (C3, C4, C5)
    Strides: 8, 16, 32

    Args:
        dep_mul: Depth multiplier (controls number of bottleneck blocks)
        wid_mul: Width multiplier (controls channel dimensions)
        out_features: Output feature names (default: ["dark3", "dark4", "dark5"])
        depthwise: Use depthwise convolutions
        act: Activation function name

    Example:
        >>> # YOLOX-s configuration
        >>> backbone = CSPDarknet(dep_mul=0.33, wid_mul=0.50)
        >>> x = torch.randn(1, 3, 640, 640)
        >>> features = backbone(x)
        >>> print([f.shape for f in features])
        # [torch.Size([1, 128, 80, 80]), torch.Size([1, 256, 40, 40]), torch.Size([1, 512, 20, 20])]
    """

    def __init__(
        self,
        dep_mul: float = 1.0,
        wid_mul: float = 1.0,
        out_features: tuple[str, ...] = ("dark3", "dark4", "dark5"),
        depthwise: bool = False,
        act: str = "silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of CSPDarknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # Stem: Focus layer
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # Stage 1: Dark2 (stride 4 -> stride 8)
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # Stage 2: Dark3 (stride 8 -> stride 16)
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # Stage 3: Dark4 (stride 16 -> stride 32)
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # Stage 4: Dark5 (stride 32 -> stride 32, with SPP)
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass returning multi-scale features.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Dictionary of features with keys from out_features
        """
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x

        x = self.dark2(x)
        outputs["dark2"] = x

        x = self.dark3(x)
        outputs["dark3"] = x

        x = self.dark4(x)
        outputs["dark4"] = x

        x = self.dark5(x)
        outputs["dark5"] = x

        return {k: v for k, v in outputs.items() if k in self.out_features}


def build_csp_darknet(model_name: str = "yolox-s") -> CSPDarknet:
    """Build CSPDarknet with predefined configurations.

    Args:
        model_name: Model variant name

    Returns:
        CSPDarknet backbone
    """
    configs = {
        "yolox-nano": {"dep_mul": 0.33, "wid_mul": 0.25},
        "yolox-tiny": {"dep_mul": 0.33, "wid_mul": 0.375},
        "yolox-s": {"dep_mul": 0.33, "wid_mul": 0.50},
        "yolox-m": {"dep_mul": 0.67, "wid_mul": 0.75},
        "yolox-l": {"dep_mul": 1.0, "wid_mul": 1.0},
        "yolox-x": {"dep_mul": 1.33, "wid_mul": 1.25},
    }

    if model_name not in configs:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(configs.keys())}"
        )

    config = configs[model_name]
    return CSPDarknet(**config)
