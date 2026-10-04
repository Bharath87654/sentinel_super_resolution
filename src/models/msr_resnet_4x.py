"""
MSRResNet4x — Multispectral Residual Network for 4× Super-Resolution

Scientific Scope
────────────────
Training task:  Synthetic 40m-equivalent LR4x (4,64,64)  →  10m HR (4,256,256).
Deployment:     Real 10m Sentinel-2 input → 2.5m-scale inferred output.
No claim:       No sub-10m ground-truth validation is performed or implied.
                Synthetic benchmark PSNR/SSIM does not constitute validation
                of genuine 2.5m spatial content in the output.

Architecture Overview
──────────────────────
1. Shallow feature extraction — Conv2d(4 → 64)
2. Deep feature extraction   — 6× Residual Channel-Attention Blocks (RCAB)
3. Body tail + long skip connection
4. 4× Upsampling via TWO sequential 2× PixelShuffle stages:
     Stage 1: Conv2d(64 → 256, 3) → PixelShuffle(2) → 128×128 features
     Stage 2: Conv2d(64 → 256, 3) → PixelShuffle(2) → 256×256 features
5. Reconstruction — Conv2d(64 → 4)
6. Global 4× bicubic skip connection — F.interpolate(..., scale_factor=4, bicubic)
7. Output = learned residual + bicubic skip

Why Two-Stage PixelShuffle?
────────────────────────────
A single 4× PixelShuffle would require Conv2d(64, 64×16=1024, 3) — a 16×
channel expansion that wastes VRAM and introduces checkerboard artifacts.
Two sequential 2× stages each expand by only 4× (64 → 256), are
empirically more stable, and match the approach used in RCAN/EDSR.

Relationship to MSRResNet2x
────────────────────────────
- ChannelAttention and ResidualCABlock are identical to msr_resnet.py.
- The only structural difference is the two-stage upsampler.
- Hyperparameters (num_features=64, num_blocks=6) are intentionally
  kept identical to the validated 2× model for direct comparison.

Inputs:   (B, 4,  64,  64) float32 — normalised LR4x patches
Outputs:  (B, 4, 256, 256) float32 — normalised SR prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules (identical to msr_resnet.py — no cross-import to keep isolation)
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention for 4-band multispectral data.
    Identical to the 2× model implementation.
    """

    def __init__(self, channels: int = 64, reduction: int = 16):
        super().__init__()
        reduced = max(4, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du  = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv_du(self.avg_pool(x))


class ResidualCABlock(nn.Module):
    """
    Residual Block with Channel Attention (RCAB).
    Identical to the 2× model implementation.
    """

    def __init__(self, channels: int = 64, reduction: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            ChannelAttention(channels, reduction=reduction),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class MSRResNet4x(nn.Module):
    """
    Multispectral Residual Network with Channel Attention — 4× Scale Factor.

    Parameters
    ----------
    in_channels  : int  — spectral input channels (4 for B02,B03,B04,B08)
    out_channels : int  — spectral output channels (4)
    num_features : int  — feature map depth throughout the backbone (64)
    num_blocks   : int  — number of RCAB blocks (6)
    reduction    : int  — channel attention reduction ratio (16)
    scale_factor : int  — spatial upsampling factor (must be 4)
    """

    def __init__(
        self,
        in_channels:  int = 4,
        out_channels: int = 4,
        num_features: int = 64,
        num_blocks:   int = 6,
        reduction:    int = 16,
        scale_factor: int = 4,
    ):
        super().__init__()

        if scale_factor != 4:
            raise ValueError(
                f"MSRResNet4x requires scale_factor=4, got {scale_factor}."
            )

        self.scale_factor = scale_factor

        # 1. Shallow feature extraction
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1, bias=True)

        # 2. Deep feature extraction — N× RCAB
        self.body = nn.Sequential(
            *[ResidualCABlock(num_features, reduction=reduction)
              for _ in range(num_blocks)]
        )
        self.body_tail = nn.Conv2d(num_features, num_features, 3, padding=1, bias=True)

        # 3. Two-stage 4× upsampling via sequential 2× PixelShuffle
        #    Stage 1: (B, 64, H,  W ) → (B, 64, 2H, 2W )
        #    Stage 2: (B, 64, 2H, 2W) → (B, 64, 4H, 4W )
        #    Each stage: Conv2d(64 → 256) then PixelShuffle(2).
        #    256 = 64 × 2² (the channel count needed by PixelShuffle for 2×).
        self.upsample_stage1 = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
        )
        self.upsample_stage2 = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
        )

        # 4. Reconstruction
        self.tail = nn.Conv2d(num_features, out_channels, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shallow features
        feat = self.head(x)                      # (B, 64, H, W)

        # Deep features + long residual skip
        res  = self.body(feat)
        res  = self.body_tail(res)
        feat = feat + res                        # (B, 64, H, W)

        # 4× upsampling: two sequential 2× PixelShuffle stages
        feat = self.upsample_stage1(feat)        # (B, 64, 2H, 2W)
        feat = self.upsample_stage2(feat)        # (B, 64, 4H, 4W)

        # Learned residual component
        residual = self.tail(feat)               # (B, 4, 4H, 4W)

        # Global 4× bicubic skip connection
        bicubic_base = F.interpolate(
            x, scale_factor=self.scale_factor,
            mode="bicubic", align_corners=False,
        )                                        # (B, 4, 4H, 4W)

        return residual + bicubic_base
