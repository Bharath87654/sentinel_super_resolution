"""
4× Deployment Inference Pipeline — 10m Sentinel-2 → 2.5m-scale

Scientific Scope
────────────────
This script executes DEPLOYMENT INFERENCE.
Input:  REAL 10m Sentinel-2 four-band imagery (B02, B03, B04, B08).
Output: 2.5m-scale INFERRED product.

IMPORTANT: The output is a model-inferred product. It does NOT represent
true 2.5m ground-truth imagery. The model was supervised exclusively on
a synthetic 40m-equivalent LR4x → 10m HR benchmark. Any fine spatial
detail at the 2.5m scale is extrapolated by the model based on that
training assumption.

Outputs generated per input:
1. <stem>_sr_reflectance.npy  — Authoritative float32 physical reflectance.
2. <stem>_sr_uint16.npy       — Storage-clipped representation (* 10000).
3. <stem>_sr_rgb.png          — RGB preview (B04-B03-B02).
4. <stem>_comparison.png      — (Optional) if an HR reference is provided.

Isolation Guarantee
────────────────────
Does not modify any training, evaluation, or dataset files.
Does not alter the raw input file.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ─── Path Registration ────────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(_project_root / "src" / "models"))
sys.path.append(str(_project_root / "src" / "training"))

from msr_resnet_4x import MSRResNet4x
from metrics import denormalize


# ─── Configuration & Paths ────────────────────────────────────────────────────
CHECKPOINT_PATH = _project_root / "outputs" / "checkpoints" / "checkpoint_best_4x.pth"
NORM_STATS_PATH = _project_root / "data" / "processed" / "normalization_stats_4x.json"
CHANNELS        = ["B02", "B03", "B04", "B08"]
RGB_INDICES     = [2, 1, 0]  # R=B04, G=B03, B=B02
EXPECTED_LR     = (4, 64, 64)
EXPECTED_HR     = (4, 256, 256)


def _apply_stretch(arr_hwc: np.ndarray, stretch_lo: float, stretch_hi: float) -> np.ndarray:
    """Apply linear stretch and clip to [0,1]."""
    return np.clip((arr_hwc - stretch_lo) / (stretch_hi - stretch_lo + 1e-8), 0.0, 1.0)


def _to_rgb_hwc(patch_chw: np.ndarray) -> np.ndarray:
    """Extract B04-B03-B02 and transpose to (H, W, 3)."""
    return patch_chw[RGB_INDICES, :, :].transpose(1, 2, 0)


def infer_4x():
    parser = argparse.ArgumentParser(
        description="4× Sentinel-2 Super-Resolution Inference"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the input Sentinel-2 .npy patch (4,64,64)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/inference_4x",
        help="Directory to save the inferred products",
    )
    parser.add_argument(
        "--hr_reference", type=str, default=None,
        help="(Optional) Path to a corresponding HR .npy file for visual comparison",
    )
    args = parser.parse_args()

    # ─── Verify Core Project Dependencies ─────────────────────────────────────
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint missing: {CHECKPOINT_PATH}")
    if not NORM_STATS_PATH.exists():
        raise FileNotFoundError(f"Normalization stats missing: {NORM_STATS_PATH}")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file missing: {input_path}")
    if input_path.suffix != ".npy":
        raise ValueError(f"Input must be a .npy file. Got: {input_path.suffix}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    # ─── Load Input Data ──────────────────────────────────────────────────────
    print("=" * 68)
    print("4× DEPLOYMENT INFERENCE: 10m Sentinel-2 → 2.5m-scale")
    print("=" * 68)

    lr_np = np.load(input_path)

    print(f"Input file   : {input_path.name}")
    print(f"Input shape  : {lr_np.shape}")
    print(f"Input dtype  : {lr_np.dtype}")

    if lr_np.shape != EXPECTED_LR:
        raise ValueError(f"Invalid input shape. Expected {EXPECTED_LR}, got {lr_np.shape}")

    # Determine reflectance
    if lr_np.dtype == np.uint16:
        lr_ref = lr_np.astype(np.float32) / 10000.0
    elif lr_np.dtype == np.float32:
        lr_ref = lr_np
        print("  Note: Processing float32 input directly as reflectance.")
    else:
        raise ValueError(f"Unsupported dtype {lr_np.dtype}. Use uint16 or float32.")

    if np.isnan(lr_ref).any() or np.isinf(lr_ref).any():
        raise ValueError("Input contains NaN or Inf values. Aborting.")

    # ─── Normalization ────────────────────────────────────────────────────────
    with open(NORM_STATS_PATH, "r") as f:
        norm_json = json.load(f)

    canonical = norm_json["canonical_normalization"]
    train_mean = torch.tensor([canonical["mean"][ch] for ch in CHANNELS], dtype=torch.float32).view(4, 1, 1)
    train_std  = torch.tensor([canonical["std"][ch]  for ch in CHANNELS], dtype=torch.float32).view(4, 1, 1)

    lr_tensor = torch.from_numpy(lr_ref)
    lr_norm   = (lr_tensor - train_mean) / train_std
    lr_norm   = lr_norm.unsqueeze(0)   # (1, 4, 64, 64)

    # ─── Load Model ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model_config = ckpt["model_config"]
    model_epoch  = ckpt.get("epoch", "?")

    model = MSRResNet4x(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model ckpt   : {CHECKPOINT_PATH.name}")
    print(f"Model epoch  : {model_epoch}")
    print(f"Device       : {device}")

    # ─── Inference ────────────────────────────────────────────────────────────
    lr_norm = lr_norm.to(device)

    t_start = time.perf_counter()
    with torch.no_grad():
        sr_norm = model(lr_norm)
    t_inf = time.perf_counter() - t_start

    assert sr_norm.shape == (1, *EXPECTED_HR), f"Model output shape error: {sr_norm.shape}"
    if torch.isnan(sr_norm).any() or torch.isinf(sr_norm).any():
        raise RuntimeError("Model output contains NaN or Inf values.")

    # ─── Denormalize to Physical Reflectance ──────────────────────────────────
    sr_ref = denormalize(sr_norm[0].cpu(), train_mean, train_std).numpy()
    assert sr_ref.shape == EXPECTED_HR

    # Output stats
    ref_min  = float(sr_ref.min())
    ref_max  = float(sr_ref.max())
    ref_mean = float(sr_ref.mean())

    pixels_below_0 = int(np.sum(sr_ref < 0.0))
    pixels_above_1 = int(np.sum(sr_ref > 1.0))
    pixels_above_65k = int(np.sum(sr_ref * 10000.0 > 65535))
    total_pixels   = sr_ref.size

    pct_below_0 = (pixels_below_0 / total_pixels) * 100
    pct_above_1 = (pixels_above_1 / total_pixels) * 100
    pct_above_65k = (pixels_above_65k / total_pixels) * 100

    pixels_clipped_uint16 = pixels_below_0 + pixels_above_65k
    pct_clipped_uint16 = (pixels_clipped_uint16 / total_pixels) * 100

    peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0

    print(f"\nInference time : {t_inf:.4f} seconds")
    if device.type == "cuda":
        print(f"Peak VRAM      : {peak_vram:.2f} MB")

    print(f"\nOutput shape   : {sr_ref.shape}")
    print(f"Reflectance min: {ref_min:.6f}")
    print(f"Reflectance max: {ref_max:.6f}")
    print(f"Reflectance avg: {ref_mean:.6f}")

    print("\nReflectance range boundaries (informative):")
    print(f"  Pixels < 0.0 : {pixels_below_0} ({pct_below_0:.4f}%)")
    print(f"  Pixels > 1.0 : {pixels_above_1} ({pct_above_1:.4f}%)")
    print(f"  > 65535/10k  : {pixels_above_65k} ({pct_above_65k:.4f}%)")

    print(f"\nTotal uint16-clipped pixels: {pixels_clipped_uint16}")
    print(f"Total uint16-clipped percentage: {pct_clipped_uint16:.4f}%")

    # ─── Save Outputs ─────────────────────────────────────────────────────────
    # 1. Authoritative Float32 Reflectance
    out_ref_path = output_dir / f"{stem}_sr_reflectance.npy"
    np.save(out_ref_path, sr_ref)

    # Verification: Ensure float32 reloading works
    _reload_check = np.load(out_ref_path)
    assert _reload_check.shape == EXPECTED_HR, "Float32 reload verification failed."
    assert _reload_check.dtype == np.float32, "Float32 dtype verification failed."

    print(f"\nSaved scientific reflectance: {out_ref_path.name}  (float32)")

    # 2. Storage-Clipped uint16
    out_u16_path = output_dir / f"{stem}_sr_uint16.npy"
    sr_u16 = np.clip(np.round(sr_ref * 10000.0), 0, 65535).astype(np.uint16)
    np.save(out_u16_path, sr_u16)
    print(f"Saved storage uint16        : {out_u16_path.name}  (clipped 0-65535)")

    # Verification: Ensure uint16 shapes/dtypes
    assert sr_u16.shape == EXPECTED_HR, "Uint16 shape verification failed."
    assert sr_u16.dtype == np.uint16, "Uint16 dtype verification failed."

    # 3. RGB Preview
    out_rgb_path = output_dir / f"{stem}_sr_rgb.png"
    sr_rgb_hwc = _to_rgb_hwc(sr_ref)

    # Standalone preview stretch (from SR output directly)
    standalone_lo = float(np.percentile(sr_rgb_hwc, 2.0))
    standalone_hi = float(np.percentile(sr_rgb_hwc, 98.0))
    sr_rgb_disp = _apply_stretch(sr_rgb_hwc, standalone_lo, standalone_hi)

    plt.figure(figsize=(8, 8))
    plt.imshow(sr_rgb_disp, interpolation="nearest")
    plt.title(f"4× Inferred SR (2.5m-scale) | {stem}\nRGB (B04-B03-B02)")
    plt.axis("off")
    plt.savefig(out_rgb_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved RGB preview           : {out_rgb_path.name}")

    # ─── Optional HR Comparison ───────────────────────────────────────────────
    if args.hr_reference:
        hr_path = Path(args.hr_reference)
        if not hr_path.exists():
            print(f"\nWARNING: HR reference missing at {hr_path}, skipping comparison.")
        else:
            hr_np = np.load(hr_path)
            if hr_np.shape != EXPECTED_HR:
                print(f"\nWARNING: HR reference shape {hr_np.shape} invalid. Skipping.")
            else:
                if hr_np.dtype == np.uint16:
                    hr_ref = hr_np.astype(np.float32) / 10000.0
                else:
                    hr_ref = hr_np

                hr_rgb_hwc = _to_rgb_hwc(hr_ref)

                # Compute 4x Bicubic
                bicubic_ref = F.interpolate(
                    torch.from_numpy(lr_ref).unsqueeze(0),
                    scale_factor=4,
                    mode="bicubic",
                    align_corners=False
                ).squeeze(0).numpy()
                bicubic_rgb_hwc = _to_rgb_hwc(bicubic_ref)

                # Canonical Stretch from HR Reference
                hr_lo = float(np.percentile(hr_rgb_hwc, 2.0))
                hr_hi = float(np.percentile(hr_rgb_hwc, 98.0))

                bicubic_rgb_disp = _apply_stretch(bicubic_rgb_hwc, hr_lo, hr_hi)
                comp_sr_rgb_disp = _apply_stretch(sr_rgb_hwc,      hr_lo, hr_hi)
                hr_rgb_disp      = _apply_stretch(hr_rgb_hwc,      hr_lo, hr_hi)

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle(f"4× Inference Comparison | {stem}", fontsize=12)

                for ax, img, title in zip(
                    axes,
                    [bicubic_rgb_disp, comp_sr_rgb_disp, hr_rgb_disp],
                    ["4× Bicubic Baseline", "MSRResNet4x Inferred (2.5m-scale)", "HR Ground Truth (10m)"]
                ):
                    ax.imshow(img, interpolation="nearest")
                    ax.set_title(title)
                    ax.axis("off")

                comp_path = output_dir / f"{stem}_comparison.png"
                plt.tight_layout()
                plt.savefig(comp_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Saved comparison preview    : {comp_path.name}")

    # ─── Final Scientific Note ────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("SCIENTIFIC SCOPE REMINDER")
    print("=" * 68)
    print("Output is a 2.5m-scale model-inferred product, not ground-truth")
    print("2.5m imagery. Any sub-10m spatial detail has been synthesized")
    print("based on the 40m-equivalent LR4x → 10m HR training benchmark.")
    print("=" * 68)


if __name__ == "__main__":
    infer_4x()
