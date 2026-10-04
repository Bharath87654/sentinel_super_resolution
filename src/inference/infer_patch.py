"""
Sentinel-2 Super-Resolution Inference Script

Runs a trained MSRResNet2x model on a single LR patch (.npy) and produces:

OUTPUTS:
  <stem>_sr_reflectance.npy  — float32, raw denormalized SR prediction (4,256,256).
                                Authoritative scientific output. No clipping applied.
  <stem>_sr_uint16.npy       — uint16 storage representation.
                                Derived via: round(reflectance * 10000).clip(0, 65535).
                                Clip is a dtype-safety guard only — not scientific clipping.
                                Clipped value count is explicitly reported.
  <stem>_sr_rgb.png          — B04-B03-B02 false-color RGB of SR prediction.
  <stem>_comparison.png      — 2-panel (bicubic vs SR) or 3-panel (bicubic vs SR vs HR).

PREPROCESSING (exactly matches training):
  Step 1: float32 = uint16_input / 10000.0
  Step 2: normalized = (float32 - train_mean) / train_std
  (train_mean, train_std, model_config, scale_factor all loaded from checkpoint)

USAGE:
  python src/inference/infer_patch.py \\
      --input  data/processed/lr/forest/patch_0001.npy \\
      --output_dir outputs/inference/

  python src/inference/infer_patch.py \\
      --input  data/processed/lr/forest/patch_0001.npy \\
      --output_dir outputs/inference/ \\
      --hr data/processed/hr/forest/patch_0001.npy
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Register src subdirectories so we can reuse existing modules
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(_project_root / "src" / "models"))
sys.path.append(str(_project_root / "src" / "training"))

from msr_resnet import MSRResNet2x
from metrics import denormalize, calculate_psnr, calculate_ssim

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
EXPECTED_LR_SHAPE = (4, 128, 128)
EXPECTED_HR_SHAPE = (4, 256, 256)
CHANNELS = ["B02", "B03", "B04", "B08"]
RGB_CHANNEL_INDICES = [2, 1, 0]   # B04=R, B03=G, B02=B in [B02,B03,B04,B08]
STRETCH_PLOW  = 2.0
STRETCH_PHIGH = 98.0

DEFAULT_CHECKPOINT = str(
    _project_root / "outputs" / "checkpoints" / "checkpoint_best.pth"
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _load_npy_validate(path: Path, expected_shape: tuple, label: str) -> np.ndarray:
    """Load a .npy file and validate its shape and numeric dtype."""
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    arr = np.load(path)
    if arr.shape != expected_shape:
        raise ValueError(
            f"{label}: expected shape {expected_shape}, got {arr.shape}"
        )
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(
            f"{label}: expected a numeric dtype, got {arr.dtype}"
        )
    return arr


def _print_array_stats(arr: np.ndarray, label: str, unit: str = ""):
    """Print min / max / mean statistics for a numpy array."""
    suffix = f" {unit}" if unit else ""
    print(f"    {label:<28} min={arr.min():.6f}{suffix}  "
          f"max={arr.max():.6f}{suffix}  mean={arr.mean():.6f}{suffix}")


def _to_rgb_hwc(patch: np.ndarray) -> np.ndarray:
    """
    Convert (4, H, W) float32 reflectance array to (H, W, 3) using
    B04-B03-B02 channel order (natural-color false-color proxy).
    No stretching applied here.
    """
    rgb = patch[RGB_CHANNEL_INDICES, :, :]   # (3, H, W)
    return rgb.transpose(1, 2, 0)            # (H, W, 3)


def _compute_stretch(rgb_hwc: np.ndarray):
    """Compute per-panel-independent 2nd–98th percentile limits from an RGB array."""
    low  = float(np.percentile(rgb_hwc, STRETCH_PLOW))
    high = float(np.percentile(rgb_hwc, STRETCH_PHIGH))
    return low, high


def _apply_stretch(rgb_hwc: np.ndarray, low: float, high: float) -> np.ndarray:
    """Apply a linear percentile stretch and clip the result to [0, 1]."""
    stretched = (rgb_hwc - low) / (high - low + 1e-8)
    return np.clip(stretched, 0.0, 1.0)


def _save_uint16(
    reflectance: np.ndarray,
    save_path: Path,
) -> dict:
    """
    Convert float32 reflectance (4, H, W) to uint16 using the project convention:
        uint16_value = round(reflectance * 10000).clip(0, 65535)

    The clip to [0, 65535] is a dtype-safety guard required by the uint16 type.
    It is NOT a scientific clipping of model predictions.
    All clipped values are counted and reported.

    Returns a dict with clip statistics.
    """
    scaled = reflectance * 10000.0
    rounded = np.round(scaled)

    # Count values that fall outside the representable uint16 range
    below_zero   = int(np.sum(rounded < 0))
    above_65535  = int(np.sum(rounded > 65535))
    total_clipped = below_zero + above_65535
    total_pixels  = rounded.size
    clipped_pct   = 100.0 * total_clipped / total_pixels

    clipped = np.clip(rounded, 0, 65535).astype(np.uint16)
    np.save(save_path, clipped)

    return {
        "clipped_below_zero":   below_zero,
        "clipped_above_65535":  above_65535,
        "total_clipped":        total_clipped,
        "total_pixels":         total_pixels,
        "clipped_pct":          clipped_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Inference Function
# ─────────────────────────────────────────────────────────────────────────────

def infer(
    input_path: Path,
    output_dir: Path,
    hr_path: Path | None,
    checkpoint_path: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    print("=" * 70)
    print("MSRResNet2x INFERENCE — Sentinel-2 Super-Resolution")
    print("=" * 70)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device:               {device}")
    if device.type == "cuda":
        print(f"  GPU:                  {torch.cuda.get_device_name(0)}")

    # ── Load Checkpoint ───────────────────────────────────────────────────────
    print(f"\n  Checkpoint:           {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    model_config   = ckpt["model_config"]
    train_mean_list = ckpt["train_mean"]   # list of 4 floats [B02,B03,B04,B08]
    train_std_list  = ckpt["train_std"]
    scale_factor_norm = ckpt["scale_factor"]  # 10000
    ckpt_epoch = ckpt.get("epoch", "?")

    # Shape: (4,1,1) for broadcasting over (4,H,W)
    train_mean = torch.tensor(train_mean_list, dtype=torch.float32).view(4, 1, 1).to(device)
    train_std  = torch.tensor(train_std_list,  dtype=torch.float32).view(4, 1, 1).to(device)

    print(f"  Checkpoint epoch:     {ckpt_epoch}")
    print(f"  Model config:         {model_config}")
    print(f"  Norm scale_factor:    {scale_factor_norm}")
    print(f"  Train mean (B02..B08): {[round(v,6) for v in train_mean_list]}")
    print(f"  Train std  (B02..B08): {[round(v,6) for v in train_std_list]}")

    # ── Load Model ────────────────────────────────────────────────────────────
    model = MSRResNet2x(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:           {total_params:,}")

    # ── Load & Validate LR Input ──────────────────────────────────────────────
    print(f"\n  Input LR patch:       {input_path}")
    lr_np = _load_npy_validate(input_path, EXPECTED_LR_SHAPE, "LR input")
    print(f"  Input shape:          {lr_np.shape}  dtype={lr_np.dtype}")

    # Raw uint16 statistics
    lr_float_raw = lr_np.astype(np.float32)
    print("\n  Input Reflectance Statistics (uint16 / 10000):")
    lr_ref_raw = lr_float_raw / float(scale_factor_norm)
    for c, ch in enumerate(CHANNELS):
        _print_array_stats(lr_ref_raw[c], ch, "")

    # ── Load & Validate Optional HR ───────────────────────────────────────────
    hr_np = None
    if hr_path is not None:
        print(f"\n  HR ground-truth:      {hr_path}")
        hr_np = _load_npy_validate(hr_path, EXPECTED_HR_SHAPE, "HR ground truth")
        print(f"  HR shape:             {hr_np.shape}  dtype={hr_np.dtype}")

    # ── Preprocessing — exactly matching training ──────────────────────────────
    # Step 1: uint16 → float32 reflectance (matches sentinel_sr_dataset.py line 113)
    lr_ref_tensor = torch.from_numpy(lr_np).to(torch.float32) / float(scale_factor_norm)
    # Step 2: Channel-wise Z-score normalization (matches sentinel_sr_dataset.py line 118)
    lr_norm_tensor = (lr_ref_tensor - train_mean.cpu()) / train_std.cpu()
    lr_norm_tensor = lr_norm_tensor.unsqueeze(0).to(device)  # (1,4,128,128)

    # Bicubic baseline (same normalized input, no preprocessing difference)
    bicubic_norm = F.interpolate(
        lr_norm_tensor, scale_factor=2, mode="bicubic", align_corners=False
    )  # (1,4,256,256)

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\n  Running inference...")
    t0 = time.perf_counter()
    with torch.no_grad():
        sr_norm = model(lr_norm_tensor)   # (1,4,256,256) normalized
    t1 = time.perf_counter()
    inference_ms = (t1 - t0) * 1000.0
    print(f"  Inference time:       {inference_ms:.2f} ms")

    # ── Inverse Normalization → Physical Reflectance ───────────────────────────
    # Uses denormalize() from metrics.py: reflectance = norm * std + mean
    sr_ref_tensor    = denormalize(sr_norm[0].cpu(),      train_mean.cpu(), train_std.cpu())
    bicubic_ref_tensor = denormalize(bicubic_norm[0].cpu(), train_mean.cpu(), train_std.cpu())

    sr_ref_np = sr_ref_tensor.numpy()           # (4,256,256) float32
    bicubic_ref_np = bicubic_ref_tensor.numpy() # (4,256,256) float32

    print(f"\n  Output SR shape:      {sr_ref_np.shape}  dtype={sr_ref_np.dtype}")
    print("\n  SR Reflectance Statistics (raw, unclipped):")
    for c, ch in enumerate(CHANNELS):
        _print_array_stats(sr_ref_np[c], ch, "")

    # ── Save Authoritative float32 SR Output ──────────────────────────────────
    sr_ref_path = output_dir / f"{stem}_sr_reflectance.npy"
    np.save(sr_ref_path, sr_ref_np)
    print(f"\n  Saved float32 SR reflectance (authoritative):  {sr_ref_path}")
    print(f"  (Raw model prediction — no clipping applied)")

    # ── Save uint16 Storage Representation ────────────────────────────────────
    sr_uint16_path = output_dir / f"{stem}_sr_uint16.npy"
    clip_stats = _save_uint16(sr_ref_np, sr_uint16_path)
    print(f"\n  Saved uint16 SR output (storage representation):  {sr_uint16_path}")
    print(f"  uint16 Clipping Report:")
    print(f"    Total pixels:          {clip_stats['total_pixels']:,}")
    print(f"    Clipped below 0:       {clip_stats['clipped_below_zero']:,}")
    print(f"    Clipped above 65535:   {clip_stats['clipped_above_65535']:,}")
    print(f"    Total clipped:         {clip_stats['total_clipped']:,}  "
          f"({clip_stats['clipped_pct']:.4f}% of pixels)")
    if clip_stats["total_clipped"] > 0:
        print(f"  NOTE: Clipping is a uint16 dtype-safety guard only.")
        print(f"        The authoritative prediction is {sr_ref_path.name}")

    # ── Ground-Truth Metrics (only if --hr provided) ──────────────────────────
    if hr_np is not None:
        # Normalize HR with the same training statistics for L1 in normalized space
        hr_ref_tensor = torch.from_numpy(hr_np).to(torch.float32) / float(scale_factor_norm)
        hr_norm_tensor = (hr_ref_tensor - train_mean.cpu()) / train_std.cpu()

        # Clipped reflectance for PSNR/SSIM — matching metrics.py convention
        sr_clip  = torch.clamp(sr_ref_tensor, 0.0, 1.0).unsqueeze(0)
        bi_clip  = torch.clamp(bicubic_ref_tensor, 0.0, 1.0).unsqueeze(0)
        hr_clip  = torch.clamp(hr_ref_tensor, 0.0, 1.0).unsqueeze(0)

        model_psnr  = calculate_psnr(sr_clip,  hr_clip,  data_range=1.0)
        model_ssim  = calculate_ssim(sr_clip,  hr_clip,  data_range=1.0)
        bicubic_psnr = calculate_psnr(bi_clip, hr_clip,  data_range=1.0)
        bicubic_ssim = calculate_ssim(bi_clip, hr_clip,  data_range=1.0)

        print(f"\n  Ground-Truth Metrics (clipped [0,1] reflectance, data_range=1.0):")
        print(f"    {'Metric':<20} {'Model SR':>12} {'Bicubic':>12} {'Delta':>12}")
        print(f"    {'-'*58}")
        print(f"    {'PSNR (dB)':<20} {model_psnr:>12.4f} {bicubic_psnr:>12.4f} "
              f"{model_psnr - bicubic_psnr:>+12.4f}")
        print(f"    {'SSIM':<20} {model_ssim:>12.6f} {bicubic_ssim:>12.6f} "
              f"{model_ssim - bicubic_ssim:>+12.6f}")
    else:
        model_psnr   = None
        model_ssim   = None
        bicubic_psnr = None
        bicubic_ssim = None

    # ── Display Stretch ────────────────────────────────────────────────────────
    # Stretch is derived from HR if provided, otherwise from SR only.
    # SR-derived stretch is used solely for visualization — not an evaluation metric.
    sr_rgb     = _to_rgb_hwc(sr_ref_np)       # (256,256,3)
    bicubic_rgb = _to_rgb_hwc(bicubic_ref_np)  # (256,256,3)

    if hr_np is not None:
        hr_ref_np = hr_np.astype(np.float32) / float(scale_factor_norm)
        hr_rgb    = _to_rgb_hwc(hr_ref_np)    # (256,256,3)
        stretch_source = "HR ground truth"
        stretch_low, stretch_high = _compute_stretch(hr_rgb)
    else:
        hr_rgb = None
        stretch_source = "SR prediction (visualization only — not an evaluation metric)"
        stretch_low, stretch_high = _compute_stretch(sr_rgb)

    print(f"\n  Display stretch source: {stretch_source}")
    print(f"  Stretch limits:         [{stretch_low:.6f}, {stretch_high:.6f}]  "
          f"(2nd–98th percentile of {('HR' if hr_np is not None else 'SR')} patch)")

    # Apply identical stretch to all panels
    bicubic_display = _apply_stretch(bicubic_rgb, stretch_low, stretch_high)
    sr_display      = _apply_stretch(sr_rgb,      stretch_low, stretch_high)
    if hr_rgb is not None:
        hr_display  = _apply_stretch(hr_rgb,      stretch_low, stretch_high)

    # ── Save SR-only RGB PNG ──────────────────────────────────────────────────
    sr_rgb_path = output_dir / f"{stem}_sr_rgb.png"
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(sr_display, interpolation="nearest")
    ax.set_title(f"SR Model Output\nB04-B03-B02  |  {stem}", fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(sr_rgb_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved SR RGB PNG:              {sr_rgb_path}")

    # ── Save Comparison PNG ───────────────────────────────────────────────────
    comparison_path = output_dir / f"{stem}_comparison.png"

    if hr_rgb is not None:
        ncols = 3
        panels = [
            (bicubic_display,
             f"Bicubic 2× Baseline"
             + (f"\nPSNR: {bicubic_psnr:.2f} dB  SSIM: {bicubic_ssim:.4f}"
                if bicubic_psnr is not None else "")),
            (sr_display,
             f"MSRResNet2x SR"
             + (f"\nPSNR: {model_psnr:.2f} dB  SSIM: {model_ssim:.4f}"
                if model_psnr is not None else "")),
            (hr_display, "HR Ground Truth\n(Reference)"),
        ]
        fig_w = 15
    else:
        ncols = 2
        panels = [
            (bicubic_display, "Bicubic 2× Baseline"),
            (sr_display,      "MSRResNet2x SR Prediction"),
        ]
        fig_w = 10

    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 5))
    if ncols == 2:
        axes = list(axes)

    stretch_label = (
        "Stretch from HR GT" if hr_np is not None
        else "Stretch from SR output (visualization only)"
    )
    fig.suptitle(
        f"Sentinel-2 SR Comparison  |  B04-B03-B02  |  {stem}\n"
        f"{stretch_label}: [{stretch_low:.4f}, {stretch_high:.4f}]",
        fontsize=10, y=1.02,
    )

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=9, pad=6)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison PNG:          {comparison_path}")

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)
    print(f"  Input:                {input_path}")
    print(f"  SR float32 (auth.):   {sr_ref_path}")
    print(f"  SR uint16 (compat.):  {sr_uint16_path}  "
          f"[{clip_stats['total_clipped']} px clipped]")
    print(f"  SR RGB PNG:           {sr_rgb_path}")
    print(f"  Comparison PNG:       {comparison_path}")
    print(f"  Inference time:       {inference_ms:.2f} ms")
    if model_psnr is not None:
        print(f"  Model PSNR:           {model_psnr:.4f} dB  "
              f"(vs bicubic {bicubic_psnr:.4f} dB)")
        print(f"  Model SSIM:           {model_ssim:.6f}  "
              f"(vs bicubic {bicubic_ssim:.6f})")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Sentinel-2 SR Inference with MSRResNet2x",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference only (no HR reference):
  python src/inference/infer_patch.py \\
      --input data/processed/lr/forest/patch_0001.npy \\
      --output_dir outputs/inference/

  # With HR reference — PSNR/SSIM computed, HR-derived display stretch:
  python src/inference/infer_patch.py \\
      --input data/processed/lr/forest/patch_0001.npy \\
      --output_dir outputs/inference/ \\
      --hr data/processed/hr/forest/patch_0001.npy
        """,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input LR patch .npy file. Required shape: (4, 128, 128).",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to save all output files.",
    )
    parser.add_argument(
        "--hr", default=None,
        help=(
            "Optional path to HR ground-truth .npy file (4, 256, 256). "
            "If provided: PSNR/SSIM are computed and display stretch is "
            "derived from HR. If absent: no ground-truth metrics are reported "
            "and stretch is derived from SR output (visualization only)."
        ),
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help=f"Path to checkpoint .pth file. Default: {DEFAULT_CHECKPOINT}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    infer(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        hr_path=Path(args.hr) if args.hr is not None else None,
        checkpoint_path=Path(args.checkpoint),
    )
