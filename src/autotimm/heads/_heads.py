"""Task-specific heads."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """Linear classification head with optional dropout.

    Parameters:
        in_features: Dimensionality of the backbone output.
        num_classes: Number of target classes.
        dropout: Dropout probability before the final linear layer.
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(x))


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale object detection.

    Takes feature maps from a backbone at multiple scales (C2-C5) and produces
    a feature pyramid (P3-P7) with top-down pathway and lateral connections.

    Parameters:
        in_channels_list: List of input channel counts from backbone features.
            Typically [256, 512, 1024, 2048] for ResNet-50.
        out_channels: Number of output channels for all pyramid levels.
        num_extra_levels: Number of extra levels to add beyond the backbone
            features. Default 2 adds P6 and P7 from P5.
        use_depthwise: Whether to use depthwise separable convolutions.
    """

    def __init__(
        self,
        in_channels_list: Sequence[int],
        out_channels: int = 256,
        num_extra_levels: int = 2,
        use_depthwise: bool = False,
    ):
        super().__init__()
        self.in_channels_list = list(in_channels_list)
        self.out_channels = out_channels
        self.num_extra_levels = num_extra_levels

        # Lateral (1x1) connections
        self.lateral_convs = nn.ModuleList()
        for in_channels in self.in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )

        # Top-down pathway (3x3) convolutions
        self.output_convs = nn.ModuleList()
        conv_fn = _depthwise_conv if use_depthwise else _standard_conv
        for _ in range(len(self.in_channels_list)):
            self.output_convs.append(conv_fn(out_channels, out_channels))

        # Extra levels (P6, P7) from P5
        self.extra_convs = nn.ModuleList()
        for i in range(num_extra_levels):
            in_ch = out_channels if i == 0 else out_channels
            self.extra_convs.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=3, stride=2, padding=1)
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass through FPN.

        Args:
            features: List of feature maps from backbone [C2, C3, C4, C5] or similar.

        Returns:
            List of pyramid features [P3, P4, P5, P6, P7] with uniform channels.
        """
        assert len(features) == len(self.in_channels_list), (
            f"Expected {len(self.in_channels_list)} features, got {len(features)}"
        )

        # Build top-down pathway
        laterals = [
            lateral_conv(f) for lateral_conv, f in zip(self.lateral_convs, features)
        ]

        # Top-down fusion (from highest level to lowest)
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )

        # Apply 3x3 convs to remove aliasing
        outputs = [
            output_conv(lateral)
            for output_conv, lateral in zip(self.output_convs, laterals)
        ]

        # Add extra levels from the last output
        last_feat = outputs[-1]
        for extra_conv in self.extra_convs:
            last_feat = F.relu(extra_conv(last_feat))
            outputs.append(last_feat)

        return outputs


def _standard_conv(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


def _depthwise_conv(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
        ),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
    )


class ScaleModule(nn.Module):
    """Learnable scalar for FCOS centerness/regression scaling."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class DetectionHead(nn.Module):
    """FCOS-style detection head with classification, regression, and centerness branches.

    This is an anchor-free detection head that predicts:
    - Class scores for each spatial location
    - Bounding box regression (distances to left, top, right, bottom)
    - Centerness scores to downweight predictions far from object centers

    Parameters:
        in_channels: Number of input channels from FPN.
        num_classes: Number of object classes (excluding background).
        num_convs: Number of conv layers in each branch before prediction.
        prior_prob: Prior probability for focal loss initialization.
        use_group_norm: Whether to use GroupNorm after conv layers.
        num_groups: Number of groups for GroupNorm.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
        num_convs: int = 4,
        prior_prob: float = 0.01,
        use_group_norm: bool = True,
        num_groups: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Shared conv layers for classification branch
        cls_convs = []
        for _ in range(num_convs):
            cls_convs.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            )
            if use_group_norm:
                cls_convs.append(nn.GroupNorm(num_groups, in_channels))
            cls_convs.append(nn.ReLU(inplace=True))
        self.cls_convs = nn.Sequential(*cls_convs)

        # Shared conv layers for regression branch
        reg_convs = []
        for _ in range(num_convs):
            reg_convs.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            )
            if use_group_norm:
                reg_convs.append(nn.GroupNorm(num_groups, in_channels))
            reg_convs.append(nn.ReLU(inplace=True))
        self.reg_convs = nn.Sequential(*reg_convs)

        # Prediction layers
        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.centerness_pred = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        # Per-level learnable scales for regression
        self.scales = nn.ModuleList([ScaleModule(1.0) for _ in range(5)])

        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob: float):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize classification bias for focal loss
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)

    def forward(
        self, features: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass through detection head.

        Args:
            features: List of FPN features [P3, P4, P5, P6, P7].

        Returns:
            Tuple of (cls_outputs, reg_outputs, centerness_outputs), each a list
            of tensors with shapes:
            - cls: [B, num_classes, H, W]
            - reg: [B, 4, H, W] (left, top, right, bottom distances)
            - centerness: [B, 1, H, W]
        """
        cls_outputs = []
        reg_outputs = []
        centerness_outputs = []

        for i, feat in enumerate(features):
            cls_feat = self.cls_convs(feat)
            reg_feat = self.reg_convs(feat)

            # Classification prediction
            cls_out = self.cls_pred(cls_feat)

            # Regression prediction (with per-level scaling)
            scale_idx = min(i, len(self.scales) - 1)
            reg_out = self.scales[scale_idx](self.reg_pred(reg_feat))
            reg_out = F.relu(reg_out)  # Distances must be positive

            # Centerness prediction (from regression branch)
            centerness_out = self.centerness_pred(reg_feat)

            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)
            centerness_outputs.append(centerness_out)

        return cls_outputs, reg_outputs, centerness_outputs

    def forward_single(
        self, feat: torch.Tensor, scale_idx: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a single feature level.

        Useful for inference or when processing levels independently.
        """
        cls_feat = self.cls_convs(feat)
        reg_feat = self.reg_convs(feat)

        cls_out = self.cls_pred(cls_feat)
        scale_idx = min(scale_idx, len(self.scales) - 1)
        reg_out = F.relu(self.scales[scale_idx](self.reg_pred(reg_feat)))
        centerness_out = self.centerness_pred(reg_feat)

        return cls_out, reg_out, centerness_out


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module for DeepLabV3+.

    Applies parallel atrous convolutions with different dilation rates to capture
    multi-scale context.

    Parameters:
        in_channels: Number of input channels from backbone.
        out_channels: Number of output channels.
        dilation_rates: List of dilation rates for parallel branches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        dilation_rates: Sequence[int] = (6, 12, 18),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1x1 convolution
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Parallel atrous convolutions
        self.atrous_convs = nn.ModuleList()
        for rate in dilation_rates:
            self.atrous_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Projection layer
        total_channels = out_channels * (len(dilation_rates) + 2)
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ASPP.

        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            Output feature map [B, out_channels, H, W]
        """
        size = x.shape[-2:]

        # Apply all branches
        x1 = self.conv1x1(x)
        x_atrous = [conv(x) for conv in self.atrous_convs]
        x_global = F.interpolate(
            self.global_pool(x), size=size, mode="bilinear", align_corners=False
        )

        # Concatenate all branches
        x = torch.cat([x1] + x_atrous + [x_global], dim=1)

        # Project to output channels
        return self.project(x)


class DeepLabV3PlusHead(nn.Module):
    """DeepLabV3+ segmentation head with ASPP and decoder.

    Combines high-level semantic features (C5) with low-level spatial features (C2)
    for accurate segmentation.

    Parameters:
        in_channels_list: List of input channel counts from backbone [C2, C3, C4, C5].
        num_classes: Number of segmentation classes.
        aspp_out_channels: Number of ASPP output channels.
        decoder_channels: Number of decoder output channels.
        dilation_rates: Dilation rates for ASPP.
    """

    def __init__(
        self,
        in_channels_list: Sequence[int],
        num_classes: int,
        aspp_out_channels: int = 256,
        decoder_channels: int = 256,
        dilation_rates: Sequence[int] = (6, 12, 18),
    ):
        super().__init__()
        self.num_classes = num_classes

        # ASPP module on high-level features (C5)
        self.aspp = ASPP(
            in_channels=in_channels_list[-1],
            out_channels=aspp_out_channels,
            dilation_rates=dilation_rates,
        )

        # Low-level feature projection (C2)
        self.low_level_projection = nn.Sequential(
            nn.Conv2d(in_channels_list[0], 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(
                aspp_out_channels + 48,
                decoder_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                decoder_channels,
                decoder_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )

        # Final classifier
        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through DeepLabV3+ head.

        Args:
            features: List of backbone features [C2, C3, C4, C5]

        Returns:
            Segmentation logits [B, num_classes, H/4, W/4]
        """
        low_level_feat = features[0]  # C2
        high_level_feat = features[-1]  # C5

        # ASPP on high-level features
        x = self.aspp(high_level_feat)

        # Upsample to match low-level feature size
        x = F.interpolate(
            x, size=low_level_feat.shape[-2:], mode="bilinear", align_corners=False
        )

        # Project low-level features
        low_level = self.low_level_projection(low_level_feat)

        # Concatenate and decode
        x = torch.cat([x, low_level], dim=1)
        x = self.decoder(x)

        # Final classification
        x = self.classifier(x)

        return x


class FCNHead(nn.Module):
    """Simple Fully Convolutional Network head for segmentation.

    A simpler baseline compared to DeepLabV3+.

    Parameters:
        in_channels: Number of input channels from backbone.
        num_classes: Number of segmentation classes.
        intermediate_channels: Number of intermediate channels.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        intermediate_channels: int = 512,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                intermediate_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(intermediate_channels, num_classes, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through FCN head.

        Args:
            features: List of backbone features (uses last one)

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        x = features[-1]  # Use highest-level feature
        return self.conv(x)


class MaskHead(nn.Module):
    """Mask prediction head for instance segmentation.

    Takes ROI-aligned features and predicts per-instance binary masks.

    Parameters:
        in_channels: Number of input channels from ROI align.
        num_classes: Number of object classes.
        hidden_channels: Number of hidden layer channels.
        num_convs: Number of convolutional layers before deconv.
        mask_size: Output mask resolution.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
        hidden_channels: int = 256,
        num_convs: int = 4,
        mask_size: int = 28,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.mask_size = mask_size

        # Convolutional layers
        convs = []
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else hidden_channels
            convs.extend(
                [
                    nn.Conv2d(in_ch, hidden_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
        self.convs = nn.Sequential(*convs)

        # Deconvolution to increase resolution
        self.deconv = nn.ConvTranspose2d(
            hidden_channels,
            hidden_channels,
            kernel_size=2,
            stride=2,
        )

        # Mask prediction
        self.mask_pred = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, roi_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through mask head.

        Args:
            roi_features: ROI-aligned features [N, C, H, W]

        Returns:
            Mask logits [N, num_classes, mask_size, mask_size]
        """
        x = self.convs(roi_features)
        x = F.relu(self.deconv(x))
        x = self.mask_pred(x)
        return x


class YOLOXHead(nn.Module):
    """YOLOX-style detection head with decoupled classification and regression branches.

    YOLOX is an anchor-free detection head that improves upon FCOS with:
    - Decoupled head: Separate feature processing for classification and regression
    - No centerness prediction (uses objectness implicitly in classification)
    - SiLU activation for better performance
    - Optional depthwise separable convolutions for efficiency

    Key differences from FCOS DetectionHead:
    - Classification and regression branches don't share features
    - Uses SiLU activation instead of ReLU
    - No centerness branch
    - Can use depthwise convolutions for mobile deployment

    Parameters:
        in_channels: Number of input channels from FPN.
        num_classes: Number of object classes (excluding background).
        num_convs: Number of conv layers in each branch before prediction.
        prior_prob: Prior probability for focal loss initialization.
        use_group_norm: Whether to use GroupNorm after conv layers.
        num_groups: Number of groups for GroupNorm.
        use_depthwise: Whether to use depthwise separable convolutions.
        activation: Activation function ("silu" or "relu").

    Example:
        >>> head = YOLOXHead(in_channels=256, num_classes=80)
        >>> fpn_features = [torch.randn(2, 256, h, w) for h, w in [(80, 80), (40, 40), (20, 20)]]
        >>> cls_outputs, reg_outputs = head(fpn_features)
        >>> print(cls_outputs[0].shape)  # [2, 80, 80, 80]
        >>> print(reg_outputs[0].shape)  # [2, 4, 80, 80]
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
        num_convs: int = 2,
        prior_prob: float = 0.01,
        use_group_norm: bool = True,
        num_groups: int = 32,
        use_depthwise: bool = False,
        activation: str = "silu",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.use_depthwise = use_depthwise

        # Select activation function
        if activation == "silu":
            act_fn = nn.SiLU(inplace=True)
        elif activation == "relu":
            act_fn = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Stem conv to process FPN features
        self.stem_conv = self._make_conv_block(
            in_channels, in_channels, use_group_norm, num_groups, act_fn
        )

        # Decoupled classification branch
        cls_convs = []
        for _ in range(num_convs):
            cls_convs.append(
                self._make_conv_block(
                    in_channels, in_channels, use_group_norm, num_groups, act_fn
                )
            )
        self.cls_convs = nn.Sequential(*cls_convs)

        # Decoupled regression branch
        reg_convs = []
        for _ in range(num_convs):
            reg_convs.append(
                self._make_conv_block(
                    in_channels, in_channels, use_group_norm, num_groups, act_fn
                )
            )
        self.reg_convs = nn.Sequential(*reg_convs)

        # Prediction layers
        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=1)

        # Per-level learnable scales for regression
        self.scales = nn.ModuleList([ScaleModule(1.0) for _ in range(5)])

        self._init_weights(prior_prob)

    def _make_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        use_group_norm: bool,
        num_groups: int,
        act_fn: nn.Module,
    ) -> nn.Sequential:
        """Create a convolutional block with optional depthwise conv, norm, and activation."""
        layers = []

        if self.use_depthwise:
            # Depthwise separable convolution
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        padding=1,
                        groups=in_channels,
                    ),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                ]
            )
        else:
            # Standard convolution
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )

        if use_group_norm:
            layers.append(nn.GroupNorm(num_groups, out_channels))

        layers.append(act_fn)

        return nn.Sequential(*layers)

    def _init_weights(self, prior_prob: float):
        """Initialize weights with special handling for classification bias."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize classification bias for focal loss
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)

    def forward(
        self, features: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass through YOLOX detection head.

        Args:
            features: List of FPN features [P3, P4, P5, P6, P7].

        Returns:
            Tuple of (cls_outputs, reg_outputs), each a list of tensors with shapes:
            - cls: [B, num_classes, H, W]
            - reg: [B, 4, H, W] (left, top, right, bottom distances)

        Note:
            Unlike FCOS, YOLOX does not produce centerness outputs.
        """
        cls_outputs = []
        reg_outputs = []

        for i, feat in enumerate(features):
            # Process through stem
            x = self.stem_conv(feat)

            # Decoupled branches process features independently
            cls_feat = self.cls_convs(x)
            reg_feat = self.reg_convs(x)

            # Classification prediction
            cls_out = self.cls_pred(cls_feat)

            # Regression prediction (with per-level scaling)
            scale_idx = min(i, len(self.scales) - 1)
            reg_out = self.scales[scale_idx](self.reg_pred(reg_feat))

            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)

        return cls_outputs, reg_outputs
