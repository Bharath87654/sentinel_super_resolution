"""
Multispectral Residual Network with Channel Attention for 2x Super-Resolution (RCAN-Lite / MSR-ResNet)

Key Features:
- Accepts 4-channel input tensors (B02, B03, B04, B08).
- Channel Attention (CA / Squeeze-and-Excitation) module exploits inter-band spectral correlations.
- Residual learning with both local and long skip connections.
- PixelShuffle 2x upsampling for efficient spatial expansion without checkerboard artifacts.
- Global Skip Connection: predicts high-frequency residual content on top of a 2x bicubic baseline.

Inputs:  (B, 4, 128, 128) PyTorch Float32 Tensor
Outputs: (B, 4, 256, 256) PyTorch Float32 Tensor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation Channel Attention module for 4 multispectral bands."""

    def __init__(self, channels: int = 64, reduction: int = 16):
        super().__init__()
        reduced_channels = max(4, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResidualCABlock(nn.Module):
    """Residual Block with Channel Attention (RCAB)."""

    def __init__(self, channels: int = 64, reduction: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            ChannelAttention(channels, reduction=reduction),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        return x + res


class MSRResNet2x(nn.Module):
    """
    Multispectral Residual Network with Channel Attention for 2x SR (RCAN-Lite / MSR-ResNet).

    Args:
        in_channels: Number of input channels (4 for Sentinel-2 B02, B03, B04, B08)
        out_channels: Number of output channels (4)
        num_features: Number of feature maps in residual backbone (default: 64)
        num_blocks: Number of Residual CA blocks (default: 6)
        reduction: Channel attention reduction ratio (default: 16)
        scale_factor: Super-resolution scale factor (default: 2)
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        num_features: int = 64,
        num_blocks: int = 6,
        reduction: int = 16,
        scale_factor: int = 2,
    ):
        super().__init__()

        self.scale_factor = scale_factor

        # 1. Shallow Feature Extraction
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1, bias=True)

        # 2. Deep Feature Extraction (Residual Group of RCABs)
        blocks = [ResidualCABlock(num_features, reduction=reduction) for _ in range(num_blocks)]
        self.body = nn.Sequential(*blocks)
        self.body_tail = nn.Conv2d(num_features, num_features, 3, padding=1, bias=True)

        # 3. Upsampling Module (2x PixelShuffle)
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor**2), 3, padding=1, bias=True),
            nn.PixelShuffle(scale_factor),
        )

        # 4. Reconstruction Layer
        self.tail = nn.Conv2d(num_features, out_channels, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shallow features
        feat = self.head(x)

        # Deep features with long skip connection
        res = self.body(feat)
        res = self.body_tail(res)
        feat = feat + res

        # 2x PixelShuffle Upsampling
        feat = self.upsampler(feat)

        # High-frequency residual prediction
        out_residual = self.tail(feat)

        # Global Skip Connection (Bicubic 2x upsampled input baseline)
        bicubic_base = F.interpolate(
            x, scale_factor=self.scale_factor, mode="bicubic", align_corners=False
        )

        out = out_residual + bicubic_base
        return out


# Alias for convenience
RCANLite2x = MSRResNet2x
