"""
Test-Set Visual Comparison Generator

Selects 2 deterministic test samples per region (first + midpoint index).
For each sample, renders a 3-column panel:
  Col 1: LR bicubic 2x upsampled
  Col 2: Model SR prediction
  Col 3: HR ground truth

Display channel: B04-B03-B02 natural-color false-color RGB proxy.
Contrast stretch: 2nd–98th percentile computed from HR ground truth only.
Same stretch applied to all three columns — no independent normalization.

Each subplot annotated with PSNR (dB) and SSIM vs. HR ground truth.

Saves to: outputs/test_visuals/{region}_sample_{idx}.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/Windows compatibility
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Resolve project root and register src subdirs on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / "src" / "data"))
sys.path.append(str(project_root / "src" / "models"))
sys.path.append(str(project_root / "src" / "training"))

from metrics import denormalize, calculate_psnr, calculate_ssim
from msr_resnet import MSRResNet2x
from sentinel_sr_dataset import SentinelSRDataset

REGIONS = ["forest", "urban", "agriculture", "water"]
SAMPLES_PER_REGION = 2

# B04, B03, B02 correspond to channel indices 2, 1, 0 in [B02,B03,B04,B08] ordering
RGB_INDICES = [2, 1, 0]  # B04=red, B03=green, B02=blue
STRETCH_PLOW = 2.0
STRETCH_PHIGH = 98.0


def _tensor_to_rgb_numpy(patch_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (4, H, W) float32 tensor in physical reflectance space
    to a (H, W, 3) float32 numpy array using B04-B03-B02 channel order.
    Values are NOT clipped or normalized here — caller applies the stretch.
    """
    arr = patch_tensor.cpu().numpy()  # (4, H, W)
    rgb = arr[RGB_INDICES, :, :]     # (3, H, W): B04, B03, B02
    return rgb.transpose(1, 2, 0)    # (H, W, 3)


def _apply_stretch(
    image: np.ndarray, low: float, high: float
) -> np.ndarray:
    """
    Apply linear percentile stretch using pre-computed low/high values.
    Clips the output to [0, 1].
    """
    stretched = (image - low) / (high - low + 1e-8)
    return np.clip(stretched, 0.0, 1.0)


def _compute_stretch_limits(hr_rgb: np.ndarray):
    """
    Compute 2nd–98th percentile stretch limits from HR ground-truth RGB only.
    Returns (low, high) scalars derived from the full 3-channel HR patch.
    """
    low = float(np.percentile(hr_rgb, STRETCH_PLOW))
    high = float(np.percentile(hr_rgb, STRETCH_PHIGH))
    return low, high


def _psnr_ssim_single(
    pred: torch.Tensor,
    target: torch.Tensor,
    train_mean: torch.Tensor,
    train_std: torch.Tensor,
):
    """
    Compute PSNR and SSIM for a single (1, 4, H, W) pair.
    Denormalize → clip [0,1] → metric.
    """
    pred_ref = denormalize(pred, train_mean, train_std)
    target_ref = denormalize(target, train_mean, train_std)
    pred_clip = torch.clamp(pred_ref, 0.0, 1.0)
    target_clip = torch.clamp(target_ref, 0.0, 1.0)
    psnr = calculate_psnr(pred_clip, target_clip, data_range=1.0)
    ssim = calculate_ssim(pred_clip, target_clip, data_range=1.0)
    return psnr, ssim


def visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    checkpoint_path = project_root / "outputs" / "checkpoints" / "checkpoint_best.pth"
    output_dir = project_root / "outputs" / "test_visuals"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # -------------------------------------------------------------------------
    # Load Checkpoint & Model
    # -------------------------------------------------------------------------
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_config = ckpt["model_config"]
    train_mean = torch.tensor(ckpt["train_mean"], dtype=torch.float32).view(4, 1, 1)
    train_std = torch.tensor(ckpt["train_std"], dtype=torch.float32).view(4, 1, 1)

    model = MSRResNet2x(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from checkpoint (epoch {ckpt['epoch']}) on {device}")

    # -------------------------------------------------------------------------
    # Test Dataset (normalize=True — same as training/evaluation)
    # -------------------------------------------------------------------------
    test_dataset = SentinelSRDataset(split="test", normalize=True)
    df = test_dataset.df  # Fixed-order metadata DataFrame (676 rows)

    # -------------------------------------------------------------------------
    # Determine Deterministic Sample Indices per Region
    # -------------------------------------------------------------------------
    viz_indices = {}  # {region: [idx_a, idx_b]}
    for region in REGIONS:
        region_mask = df["region"] == region
        region_positions = df.index[region_mask].tolist()  # positions in 0..675 range
        assert len(region_positions) >= 2, (
            f"Region '{region}' has fewer than 2 test samples."
        )
        idx_first = region_positions[0]
        idx_mid = region_positions[len(region_positions) // 2]
        viz_indices[region] = [idx_first, idx_mid]
        print(f"  {region}: sample indices [{idx_first}, {idx_mid}] "
              f"(filenames: {df.iloc[idx_first]['filename']}, "
              f"{df.iloc[idx_mid]['filename']})")

    # -------------------------------------------------------------------------
    # Generate Visualizations
    # -------------------------------------------------------------------------
    print("\nGenerating visual comparisons...")

    for region in REGIONS:
        for sample_num, dataset_idx in enumerate(viz_indices[region]):
            meta_row = df.iloc[dataset_idx]
            filename = meta_row["filename"]

            # Load normalized tensors from dataset
            lr_norm, hr_norm = test_dataset[dataset_idx]
            lr_norm = lr_norm.unsqueeze(0).to(device)   # (1, 4, 128, 128)
            hr_norm = hr_norm.unsqueeze(0).to(device)   # (1, 4, 256, 256)

            with torch.no_grad():
                # Model SR prediction
                model_pred_norm = model(lr_norm)       # (1, 4, 256, 256)

                # Bicubic 2x baseline
                bicubic_pred_norm = F.interpolate(
                    lr_norm, scale_factor=2, mode="bicubic", align_corners=False
                )                                       # (1, 4, 256, 256)

            # --- Denormalize all three to physical reflectance space ---
            model_ref = denormalize(model_pred_norm[0], train_mean, train_std)
            bicubic_ref = denormalize(bicubic_pred_norm[0], train_mean, train_std)
            hr_ref = denormalize(hr_norm[0], train_mean, train_std)

            # --- Per-sample metrics ---
            m_psnr, m_ssim = _psnr_ssim_single(
                model_pred_norm, hr_norm, train_mean, train_std
            )
            b_psnr, b_ssim = _psnr_ssim_single(
                bicubic_pred_norm, hr_norm, train_mean, train_std
            )

            # --- RGB extraction (B04-B03-B02, no normalization yet) ---
            bicubic_rgb = _tensor_to_rgb_numpy(bicubic_ref)   # (256, 256, 3)
            model_rgb = _tensor_to_rgb_numpy(model_ref)       # (256, 256, 3)
            hr_rgb = _tensor_to_rgb_numpy(hr_ref)             # (256, 256, 3)

            # --- Compute stretch from HR ground truth only ---
            stretch_low, stretch_high = _compute_stretch_limits(hr_rgb)

            # --- Apply identical stretch to all three ---
            bicubic_display = _apply_stretch(bicubic_rgb, stretch_low, stretch_high)
            model_display = _apply_stretch(model_rgb, stretch_low, stretch_high)
            hr_display = _apply_stretch(hr_rgb, stretch_low, stretch_high)

            # --- Build 3-column panel ---
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(
                f"Region: {region.capitalize()}  |  File: {filename}\n"
                f"Stretch: [{stretch_low:.4f}, {stretch_high:.4f}] "
                f"(2nd–98th %ile of HR GT)  |  B04-B03-B02 RGB",
                fontsize=10, y=1.02,
            )

            panels = [
                (bicubic_display, f"Bicubic 2× Baseline\nPSNR: {b_psnr:.2f} dB  |  SSIM: {b_ssim:.4f}"),
                (model_display,   f"Model SR (MSRResNet2x)\nPSNR: {m_psnr:.2f} dB  |  SSIM: {m_ssim:.4f}"),
                (hr_display,      "HR Ground Truth\n(Reference)"),
            ]

            for ax, (img, title) in zip(axes, panels):
                ax.imshow(img, interpolation="nearest")
                ax.set_title(title, fontsize=9, pad=6)
                ax.axis("off")

            plt.tight_layout()

            save_path = output_dir / f"{region}_sample_{sample_num}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {save_path.name}  "
                  f"(Model PSNR: {m_psnr:.2f} dB vs Bicubic: {b_psnr:.2f} dB)")

    print(f"\nAll {len(REGIONS) * SAMPLES_PER_REGION} visual comparisons saved to: {output_dir}")


if __name__ == "__main__":
    visualize()
