"""
4× Sentinel-2 SR — Quality Heatmaps and Consistency Metrics

Scientific Scope
────────────────
Generates scientifically rigorous quality heatmaps for the 4× SR system.

MODE 1: Synthetic Test Benchmark
Uses the held-out TEST split (LR4x -> HR) where genuine ground-truth
is available. Generates true error maps, error improvement, NDVI error,
and Spectral Angle Mapper (SAM) error.

MODE 2: Real Deployment
For a given 10m input patch without ground truth. Generates radiometric
consistency residuals, NDVI consistency, and SR vs Bicubic difference maps.
THESE ARE NOT ERROR MAPS. They measure internal consistency and model
disagreement, not ground-truth accuracy.

TTA Uncertainty Proxy:
Computes the per-pixel standard deviation across 6 spatial test-time
augmentations. This is an uncertainty proxy, NOT calibrated uncertainty.

Isolation Guarantee
────────────────────
Does not modify any training, evaluation, inference, or dataset files.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ─── Path Registration ────────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(_project_root / "src" / "data"))
sys.path.append(str(_project_root / "src" / "models"))
sys.path.append(str(_project_root / "src" / "training"))

from metrics import denormalize, calculate_psnr, calculate_ssim
from msr_resnet_4x import MSRResNet4x
from sentinel_sr_dataset_4x import SentinelSR4xDataset

# ─── Configuration ────────────────────────────────────────────────────────────
CHECKPOINT_PATH = _project_root / "outputs" / "checkpoints" / "checkpoint_best_4x.pth"
NORM_STATS_PATH = _project_root / "data" / "processed" / "normalization_stats_4x.json"
OUTPUT_BASE     = _project_root / "outputs" / "quality_maps_4x"
METRICS_JSON    = OUTPUT_BASE / "quality_metrics.json"

CHANNELS    = ["B02", "B03", "B04", "B08"]
REGIONS     = ["forest", "urban", "agriculture", "water"]
RGB_INDICES = [2, 1, 0]  # R=B04, G=B03, B=B02
NDVI_RED    = 2          # B04 index
NDVI_NIR    = 3          # B08 index


# ─── Display & Math Utilities ─────────────────────────────────────────────────

def _compute_ndvi(refl_chw: np.ndarray) -> np.ndarray:
    """Compute NDVI = (NIR - Red) / (NIR + Red) from physical reflectance."""
    red = refl_chw[NDVI_RED]
    nir = refl_chw[NDVI_NIR]
    return (nir - red) / (nir + red + 1e-8)


def _compute_sam(refl_a: np.ndarray, refl_b: np.ndarray) -> np.ndarray:
    """Compute Spectral Angle Mapper (SAM) in degrees between two (4,H,W) arrays."""
    # Dot product along channel axis
    dot = np.sum(refl_a * refl_b, axis=0)
    norm_a = np.linalg.norm(refl_a, axis=0)
    norm_b = np.linalg.norm(refl_b, axis=0)

    # Clip for numerical stability
    cos_theta = np.clip(dot / (norm_a * norm_b + 1e-8), -1.0, 1.0)
    sam_rad = np.arccos(cos_theta)
    return sam_rad * (180.0 / math.pi)


def _save_heatmap(data_hw: np.ndarray, save_path: Path, title: str, cmap: str,
                  vmin: float = None, vmax: float = None, label: str = ""):
    """Generic heatmap saver."""
    plt.figure(figsize=(7, 6))
    if vmin is None: vmin = float(np.min(data_hw))
    if vmax is None: vmax = float(np.max(data_hw))

    # Ensure symmetric diverging colormaps are centered if requested
    if cmap in ["RdBu", "RdBu_r", "bwr"] and (vmin < 0 and vmax > 0):
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

    im = plt.imshow(data_hw, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.title(title, fontsize=10, pad=10)
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04, label=label)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_4band_map(data_chw: np.ndarray, save_path: Path, title: str, cmap: str,
                    global_vmax: float, label: str = ""):
    """4-panel band-wise heatmap (absolute values mapped 0 to vmax)."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(title, fontsize=12, y=1.02)

    for i, band in enumerate(CHANNELS):
        ax = axes[i]
        im = ax.imshow(data_chw[i], cmap=cmap, vmin=0.0, vmax=global_vmax, interpolation="nearest")
        ax.set_title(f"{band}", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_signed_4band_map(data_chw: np.ndarray, save_path: Path, title: str, cmap: str,
                           global_abs_max: float, label: str = ""):
    """4-panel band-wise heatmap for signed values symmetrically centered around zero."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(title, fontsize=12, y=1.02)

    for i, band in enumerate(CHANNELS):
        ax = axes[i]
        im = ax.imshow(data_chw[i], cmap=cmap, vmin=-global_abs_max, vmax=global_abs_max, interpolation="nearest")
        ax.set_title(f"{band}", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Core TTA Logic ───────────────────────────────────────────────────────────

def run_tta_proxy(model, lr_norm_t: torch.Tensor) -> np.ndarray:
    """
    Test-Time Augmentation Disagreement Proxy.
    Returns the per-pixel std dev (4,H,W) across 6 spatial augmentations.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        # 1. Identity
        preds.append(model(lr_norm_t))

        # 2. Horizontal Flip
        x_hf = torch.flip(lr_norm_t, [3])
        y_hf = model(x_hf)
        preds.append(torch.flip(y_hf, [3]))

        # 3. Vertical Flip
        x_vf = torch.flip(lr_norm_t, [2])
        y_vf = model(x_vf)
        preds.append(torch.flip(y_vf, [2]))

        # 4. Rot 90
        x_r90 = torch.rot90(lr_norm_t, 1, [2, 3])
        y_r90 = model(x_r90)
        preds.append(torch.rot90(y_r90, -1, [2, 3]))

        # 5. Rot 180
        x_r180 = torch.rot90(lr_norm_t, 2, [2, 3])
        y_r180 = model(x_r180)
        preds.append(torch.rot90(y_r180, -2, [2, 3]))

        # 6. Rot 270
        x_r270 = torch.rot90(lr_norm_t, 3, [2, 3])
        y_r270 = model(x_r270)
        preds.append(torch.rot90(y_r270, -3, [2, 3]))

    stacked = torch.cat(preds, dim=0)       # (6, 4, 256, 256)
    std_dev = torch.std(stacked, dim=0)     # (4, 256, 256)
    return std_dev.cpu().numpy()


# ─── Main Execution ───────────────────────────────────────────────────────────

def generate_quality_maps():
    parser = argparse.ArgumentParser(description="4× SR Quality Heatmaps")
    parser.add_argument("--real_input", type=str, default=None,
                        help="Optional: Run Mode 2 only on a specific real 10m .npy patch")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # ─── Load Model & Normalization ───────────────────────────────────────────
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model = MSRResNet4x(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(NORM_STATS_PATH) as f:
        norm_json = json.load(f)
    can = norm_json["canonical_normalization"]
    train_mean = torch.tensor([can["mean"][ch] for ch in CHANNELS], dtype=torch.float32).view(4,1,1)
    train_std  = torch.tensor([can["std"][ch]  for ch in CHANNELS], dtype=torch.float32).view(4,1,1)

    all_metrics = []

    # =========================================================================
    # OPTION A: Specific Real Deployment Patch (Mode 2 Only)
    # =========================================================================
    if args.real_input:
        in_path = Path(args.real_input)
        stem = in_path.stem
        out_dir = OUTPUT_BASE / f"real_{stem}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing Real Deployment Patch: {stem}")

        lr_np = np.load(in_path)

        # Validation checks
        if lr_np.shape != (4, 64, 64):
            raise ValueError(f"Invalid shape. Expected (4, 64, 64), got {lr_np.shape}")

        if lr_np.dtype not in [np.uint16, np.float32]:
            raise ValueError(f"Invalid dtype. Expected uint16 or float32, got {lr_np.dtype}")

        if np.isnan(lr_np).any() or np.isinf(lr_np).any():
            raise ValueError("Deployment input contains NaN or Inf values.")

        if lr_np.dtype == np.uint16:
            lr_ref = lr_np.astype(np.float32) / 10000.0
        else:
            lr_ref = lr_np

        lr_norm_t = ((torch.from_numpy(lr_ref) - train_mean) / train_std).unsqueeze(0).to(device)

        with torch.no_grad():
            sr_norm_t = model(lr_norm_t)
            bic_norm_t = F.interpolate(lr_norm_t, scale_factor=4, mode="bicubic", align_corners=False)

        sr_ref  = denormalize(sr_norm_t[0].cpu(),  train_mean, train_std).numpy()
        bic_ref = denormalize(bic_norm_t[0].cpu(), train_mean, train_std).numpy()

        # 1. SR vs Bicubic Difference
        sr_bic_diff = np.abs(sr_ref - bic_ref)
        _save_4band_map(sr_bic_diff, out_dir / "sr_vs_bicubic_difference.png",
                        "SR − Bicubic Difference (Reflectance)", "hot",
                        global_vmax=float(np.percentile(sr_bic_diff, 99.5)), label="Δ Refl.")

        # 2. Downsampled Radiometric Consistency
        sr_t_cpu = torch.from_numpy(sr_ref).unsqueeze(0) # (1,4,256,256)
        sr_10m_t = F.avg_pool2d(sr_t_cpu, kernel_size=4, stride=4) # (1,4,64,64)
        sr_10m_ref = sr_10m_t[0].numpy()

        rad_consistency = np.abs(sr_10m_ref - lr_ref)
        _save_4band_map(rad_consistency, out_dir / "radiometric_consistency_residual.png",
                        "10m Radiometric Consistency Residual", "hot",
                        global_vmax=float(np.percentile(rad_consistency, 99.5)), label="Δ Refl.")

        # 3. NDVI Consistency
        ndvi_lr = _compute_ndvi(lr_ref)
        ndvi_sr_10m = _compute_ndvi(sr_10m_ref)
        ndvi_consist = ndvi_sr_10m - ndvi_lr

        ndvi_abs_max = float(np.percentile(np.abs(ndvi_consist), 99.5))
        ndvi_vmin = -max(ndvi_abs_max, 1e-6)
        ndvi_vmax = max(ndvi_abs_max, 1e-6)

        _save_heatmap(ndvi_consist, out_dir / "ndvi_consistency_residual.png",
                      "NDVI Consistency Residual (Downsampled SR − Input)", "RdBu",
                      vmin=ndvi_vmin, vmax=ndvi_vmax, label="Δ NDVI")

        # 3b. NDVI Consistency Display (Nearest-neighbor upscaled to 2.5m grid for visualization)
        ndvi_consist_t = torch.from_numpy(ndvi_consist).unsqueeze(0).unsqueeze(0) # (1, 1, 64, 64)
        ndvi_consist_display = F.interpolate(ndvi_consist_t, scale_factor=4, mode="nearest").squeeze().numpy() # (256, 256)

        _save_heatmap(ndvi_consist_display, out_dir / "ndvi_consistency_residual_display.png",
                      "NDVI Consistency Residual (10m)\n(Display enlarged from 10m grid; no new spatial detail added.)",
                      "RdBu", vmin=ndvi_vmin, vmax=ndvi_vmax, label="Δ NDVI")

        # 3c. 2.5m-scale SR NDVI
        ndvi_sr_2p5m = _compute_ndvi(sr_ref)
        _save_heatmap(ndvi_sr_2p5m, out_dir / "sr_ndvi_2p5m.png",
                      "SR NDVI (2.5m-scale inferred)", "RdYlGn",
                      vmin=-1.0, vmax=1.0, label="NDVI")

        # 4. TTA Disagreement
        tta_std_norm = run_tta_proxy(model, lr_norm_t)
        tta_std_ref  = tta_std_norm * train_std.numpy()
        _save_4band_map(tta_std_ref, out_dir / "tta_disagreement_proxy.png",
                        "Uncertainty proxy — Test-Time Augmentation Disagreement (σ)", "viridis",
                        global_vmax=float(np.percentile(tta_std_ref, 99.5)), label="Std Dev (Refl.)")

        # Save deployment metadata JSON
        deployment_metrics = {
            "mode": "real_deployment",
            "input_filename": in_path.name,
            "input_shape": list(lr_np.shape),
            "input_dtype": str(lr_np.dtype),
            "mean_tta_disagreement": float(tta_std_ref.mean()),
            "mean_radiometric_consistency_residual": float(rad_consistency.mean()),
            "mean_absolute_ndvi_consistency": float(np.abs(ndvi_consist).mean()),
            "scientific_scope": (
                "Deployment output is a 2.5m-scale model-inferred product, "
                "not ground-truth 2.5m imagery. Residuals measure internal "
                "consistency and model disagreement, not ground-truth accuracy."
            ),
            "tta_is_uncertainty_proxy": True,
            "tta_disclaimer": "This is a disagreement-based proxy and is not calibrated probabilistic uncertainty.",
            "ground_truth_available": False,
        }
        with open(out_dir / "quality_metrics.json", "w") as f:
            json.dump(deployment_metrics, f, indent=2)

        print(f"  Saved deployment consistency metrics to {out_dir}")
        return

    # =========================================================================
    # OPTION B: Synthetic Test Benchmark (Mode 1 + Mode 2)
    # =========================================================================
    test_ds = SentinelSR4xDataset(split="test", normalize=True)
    df = test_ds.df

    # Deterministic selection (first + mid per region)
    selected = {}
    for region in REGIONS:
        pos = np.flatnonzero(df["region"] == region).tolist()
        selected[region] = [pos[0], pos[len(pos) // 2]]

    print("=" * 68)
    print("GENERATING SCIENTIFIC QUALITY HEATMAPS")
    print("=" * 68)

    for region in REGIONS:
        for ds_idx in selected[region]:
            row = df.iloc[ds_idx]
            filename = row["filename"]
            stem = Path(filename).stem

            out_dir = OUTPUT_BASE / f"synthetic_{region}_{stem}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nProcessing Benchmark Sample: {region} | {stem}")

            lr_norm, hr_norm = test_ds[ds_idx]
            lr_norm_t = lr_norm.unsqueeze(0).to(device)
            hr_norm_t = hr_norm.unsqueeze(0).to(device)

            with torch.no_grad():
                sr_norm_t = model(lr_norm_t)
                bic_norm_t = F.interpolate(lr_norm_t, scale_factor=4, mode="bicubic", align_corners=False)

            # Metrics
            p_clip = torch.clamp(denormalize(sr_norm_t[0].cpu(), train_mean, train_std), 0, 1).unsqueeze(0)
            t_clip = torch.clamp(denormalize(hr_norm_t[0].cpu(), train_mean, train_std), 0, 1).unsqueeze(0)
            b_clip = torch.clamp(denormalize(bic_norm_t[0].cpu(), train_mean, train_std), 0, 1).unsqueeze(0)

            model_psnr = float(calculate_psnr(p_clip, t_clip, 1.0))
            model_ssim = float(calculate_ssim(p_clip, t_clip, 1.0))
            bic_psnr   = float(calculate_psnr(b_clip, t_clip, 1.0))
            bic_ssim   = float(calculate_ssim(b_clip, t_clip, 1.0))

            # Unclipped Reflectances
            sr_ref  = denormalize(sr_norm_t[0].cpu(), train_mean, train_std).numpy()
            hr_ref  = denormalize(hr_norm_t[0].cpu(), train_mean, train_std).numpy()
            bic_ref = denormalize(bic_norm_t[0].cpu(), train_mean, train_std).numpy()
            lr_ref  = denormalize(lr_norm.cpu(), train_mean, train_std).numpy()

            # ─── MODE 1: SYNTHETIC BENCHMARK (GROUND-TRUTH AVAILABLE) ─────────

            # 1 & 2. Absolute Error Maps
            sr_err  = np.abs(sr_ref - hr_ref)
            bic_err = np.abs(bic_ref - hr_ref)
            global_err_vmax = float(np.percentile(bic_err, 99.5))

            _save_4band_map(sr_err, out_dir / "reconstruction_error.png",
                            "MSRResNet4x Absolute Reconstruction Error", "hot",
                            global_vmax=global_err_vmax, label="|SR − HR|")

            _save_4band_map(bic_err, out_dir / "bicubic_error.png",
                            "Bicubic Absolute Reconstruction Error", "hot",
                            global_vmax=global_err_vmax, label="|Bicubic − HR|")

            # 3. Error Improvement Map (|Bicubic - HR| - |SR - HR|)
            # Positive = SR is better
            err_improv = bic_err - sr_err
            abs_max_improv = float(np.percentile(np.abs(err_improv), 99.5))
            _save_signed_4band_map(err_improv, out_dir / "error_improvement.png",
                                   "Error Improvement (Positive = SR Lower Error)", "RdBu_r",
                                   global_abs_max=abs_max_improv, label="Error Improvement")

            # 4. NDVI Difference (SR - HR)
            ndvi_sr = _compute_ndvi(sr_ref)
            ndvi_hr = _compute_ndvi(hr_ref)
            ndvi_diff = ndvi_sr - ndvi_hr
            _save_heatmap(ndvi_diff, out_dir / "ndvi_difference.png",
                          "NDVI Difference (SR − HR)", "RdBu",
                          vmin=-0.15, vmax=0.15, label="Δ NDVI")

            # 5. Spectral Angle Mapper (SAM)
            sam_map = _compute_sam(sr_ref, hr_ref)
            _save_heatmap(sam_map, out_dir / "spectral_angle.png",
                          "Spectral Angle Mapper (SAM) Error", "inferno",
                          vmin=0.0, vmax=float(np.percentile(sam_map, 99)), label="SAM (degrees)")

            # ─── MODE 2: DEPLOYMENT METRICS ───────────────────────────────────

            # 6. SR vs Bicubic Difference
            sr_bic_diff = np.abs(sr_ref - bic_ref)
            _save_4band_map(sr_bic_diff, out_dir / "sr_vs_bicubic_difference.png",
                            "SR − Bicubic Difference (Reflectance)", "hot",
                            global_vmax=float(np.percentile(sr_bic_diff, 99.5)), label="Δ Refl.")

            # 7. Radiometric Consistency Residual
            sr_t_cpu = torch.from_numpy(sr_ref).unsqueeze(0)
            sr_10m_t = F.avg_pool2d(sr_t_cpu, kernel_size=4, stride=4)
            sr_10m_ref = sr_10m_t[0].numpy()

            rad_consistency = np.abs(sr_10m_ref - lr_ref)
            _save_4band_map(rad_consistency, out_dir / "radiometric_consistency_residual.png",
                            "10m Radiometric Consistency Residual", "hot",
                            global_vmax=float(np.percentile(rad_consistency, 99.5)), label="Δ Refl.")

            # 8. NDVI Consistency
            ndvi_lr = _compute_ndvi(lr_ref)
            ndvi_sr_10m = _compute_ndvi(sr_10m_ref)
            ndvi_consist = ndvi_sr_10m - ndvi_lr

            ndvi_abs_max = float(np.percentile(np.abs(ndvi_consist), 99.5))
            ndvi_vmin = -max(ndvi_abs_max, 1e-6)
            ndvi_vmax = max(ndvi_abs_max, 1e-6)

            _save_heatmap(ndvi_consist, out_dir / "ndvi_consistency_residual.png",
                          "NDVI Consistency Residual (Downsampled SR − Input)", "RdBu",
                          vmin=ndvi_vmin, vmax=ndvi_vmax, label="Δ NDVI")

            # 8b. NDVI Consistency Display (Nearest-neighbor upscaled to 2.5m grid for visualization)
            ndvi_consist_t = torch.from_numpy(ndvi_consist).unsqueeze(0).unsqueeze(0) # (1, 1, 64, 64)
            ndvi_consist_display = F.interpolate(ndvi_consist_t, scale_factor=4, mode="nearest").squeeze().numpy() # (256, 256)

            _save_heatmap(ndvi_consist_display, out_dir / "ndvi_consistency_residual_display.png",
                          "NDVI Consistency Residual (10m)\n(Display enlarged from 10m grid; no new spatial detail added.)",
                          "RdBu", vmin=ndvi_vmin, vmax=ndvi_vmax, label="Δ NDVI")

            # 8c. 2.5m-scale SR NDVI
            ndvi_sr_2p5m = _compute_ndvi(sr_ref)
            _save_heatmap(ndvi_sr_2p5m, out_dir / "sr_ndvi_2p5m.png",
                          "SR NDVI (2.5m-scale inferred)", "RdYlGn",
                          vmin=-1.0, vmax=1.0, label="NDVI")

            # 9. TTA Disagreement Proxy
            tta_std_norm = run_tta_proxy(model, lr_norm_t)
            tta_std_ref  = tta_std_norm * train_std.numpy()
            _save_4band_map(tta_std_ref, out_dir / "tta_disagreement_proxy.png",
                            "Uncertainty proxy — Test-Time Augmentation Disagreement (σ)", "viridis",
                            global_vmax=float(np.percentile(tta_std_ref, 99.5)), label="Std Dev (Refl.)")

            # ─── Logging ──────────────────────────────────────────────────────
            sample_metrics = {
                "filename": filename,
                "region": region,
                "ground_truth_benchmark": {
                    "model_psnr": model_psnr,
                    "model_ssim": model_ssim,
                    "bicubic_psnr": bic_psnr,
                    "bicubic_ssim": bic_ssim,
                    "mean_absolute_reconstruction_error": float(sr_err.mean()),
                    "mean_absolute_ndvi_difference": float(np.abs(ndvi_diff).mean()),
                    "mean_sam_degrees": float(sam_map.mean()),
                },
                "deployment_consistency": {
                    "mean_tta_disagreement": float(tta_std_ref.mean()),
                    "mean_radiometric_consistency_residual": float(rad_consistency.mean()),
                },
                "tta_disclaimer": "This is a disagreement-based proxy and is not calibrated probabilistic uncertainty."
            }
            all_metrics.append(sample_metrics)
            print("  -> Quality maps generated and saved.")

    # Write summary JSON
    with open(METRICS_JSON, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("\n" + "=" * 68)
    print(f"Saved quality metrics summary to: {METRICS_JSON.name}")
    print("=" * 68)


if __name__ == "__main__":
    generate_quality_maps()
