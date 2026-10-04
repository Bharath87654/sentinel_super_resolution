"""
Super-Resolution Evaluation Metrics Module

Includes:
- De-normalization transformation (norm -> reflectance domain).
- Unclipped Physical Reflectance MAE calculation.
- Clipped PSNR (dB) calculation (data_range = 1.0).
- Pure PyTorch 4-channel averaged SSIM calculation (data_range = 1.0).
- Pixel boundary out-of-bounds percentage reporting.
"""

import math
import torch
import torch.nn.functional as F


def denormalize(
    tensor_norm: torch.Tensor, train_mean: torch.Tensor, train_std: torch.Tensor
) -> torch.Tensor:
    """
    Inverse Z-score normalization: reflectance = norm * std + mean
    """
    mean = train_mean.to(tensor_norm.device)
    std = train_std.to(tensor_norm.device)
    return (tensor_norm * std) + mean


def calculate_psnr(
    pred_clip: torch.Tensor, target_clip: torch.Tensor, data_range: float = 1.0
) -> float:
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) in dB on clipped tensors.
    """
    mse = F.mse_loss(pred_clip, target_clip).item()
    if mse == 0:
        return 100.0
    return 10.0 * math.log10((data_range**2) / mse)


def _gaussian_window(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Generates a 1D Gaussian kernel."""
    gauss = torch.exp(
        -torch.tensor(
            [(x - window_size // 2) ** 2 / (2 * sigma**2) for x in range(window_size)]
        )
    )
    return gauss / gauss.sum()


def _create_3d_window(
    window_size: int, channel: int, device: torch.device
) -> torch.Tensor:
    """Creates a 2D Gaussian window kernel for SSIM calculation."""
    _1d = _gaussian_window(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channel, 1, window_size, window_size).contiguous()
    return window.to(device)


def calculate_ssim(
    pred_clip: torch.Tensor, target_clip: torch.Tensor, data_range: float = 1.0
) -> float:
    """
    Calculates SSIM independently for all 4 spectral channels and returns average.
    Inputs: (B, 4, H, W) PyTorch Tensors
    """
    window_size = 11
    channels = pred_clip.shape[1]
    device = pred_clip.device
    window = _create_3d_window(window_size, channels, device)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = F.conv2d(pred_clip, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target_clip, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(
            pred_clip * pred_clip, window, padding=window_size // 2, groups=channels
        )
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(
            target_clip * target_clip,
            window,
            padding=window_size // 2,
            groups=channels,
        )
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(
            pred_clip * target_clip, window, padding=window_size // 2, groups=channels
        )
        - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    # Average over channels and spatial dimensions
    return ssim_map.mean().item()


def evaluate_batch_metrics(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    train_mean: torch.Tensor,
    train_std: torch.Tensor,
):
    """
    Computes validation evaluation metrics:
    1. Inverse-normalizes to physical reflectance space.
    2. Calculates Unclipped Reflectance MAE.
    3. Calculates Clipped PSNR (dB) and SSIM (data_range=1.0).
    4. Computes percentage of target pixels outside [0, 1].
    """
    # 1. Inverse Normalization
    pred_ref = denormalize(pred_norm, train_mean, train_std)
    target_ref = denormalize(target_norm, train_mean, train_std)

    # 2. Unclipped Reflectance MAE
    reflectance_mae = F.l1_loss(pred_ref, target_ref).item()

    # 3. Out-of-bounds pixel percentage in target
    out_of_bounds_pixels = (target_ref < 0.0) | (target_ref > 1.0)
    out_of_bounds_pct = (
        out_of_bounds_pixels.float().mean().item() * 100.0
    )

    # 4. Clipped Metrics for PSNR & SSIM
    pred_clip = torch.clamp(pred_ref, 0.0, 1.0)
    target_clip = torch.clamp(target_ref, 0.0, 1.0)

    psnr_db = calculate_psnr(pred_clip, target_clip, data_range=1.0)
    ssim_val = calculate_ssim(pred_clip, target_clip, data_range=1.0)

    return {
        "reflectance_mae": reflectance_mae,
        "psnr_db": psnr_db,
        "ssim": ssim_val,
        "out_of_bounds_pct": out_of_bounds_pct,
    }
