"""
4× Sentinel-2 SR — Qualitative Visual Comparison Pipeline

Scientific Scope
────────────────
These visuals demonstrate the synthetic 40m-equivalent LR4x → 10m HR
benchmark for the trained MSRResNet4x model. They must NOT be presented
as proof of genuine 2.5m spatial reconstruction. Region labels (forest,
urban, agriculture, water) are dataset acquisition categories, not
scientifically validated land-cover classes.

Visual Pipeline per Sample
───────────────────────────
  LR4x → 4× Bicubic  ┐
  LR4x → MSRResNet4x ├─→ RGB comparison   (outputs/test_visuals_4x/rgb/)
  HR Reference        ┘
                         Absolute-difference map vs HR
                         Band-wise comparison           (outputs/test_visuals_4x/bands/)
                         Difference maps                (outputs/test_visuals_4x/differences/)

RGB Convention (strict, no exception):
  R = B04  (channel index 2 in [B02,B03,B04,B08])
  G = B03  (channel index 1)
  B = B02  (channel index 0)
  B08 is NOT used in RGB panels.

Display stretch:
  2nd–98th percentile computed from HR ground-truth only.
  Identical stretch applied to Bicubic, SR, and HR panels.
  No independent per-panel normalization.

Isolation Guarantee
────────────────────
Reads  : outputs/checkpoints/checkpoint_best_4x.pth
         data/processed/ (via dataset — read only)
         data/processed/normalization_stats_4x.json

Writes : outputs/test_visuals_4x/rgb/
         outputs/test_visuals_4x/bands/
         outputs/test_visuals_4x/differences/
         outputs/test_visuals_4x/selected_samples.csv

Modifies: nothing.
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F

# ─── Path registration ────────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(_project_root / "src" / "data"))
sys.path.append(str(_project_root / "src" / "models"))
sys.path.append(str(_project_root / "src" / "training"))

from metrics import denormalize, calculate_psnr, calculate_ssim
from msr_resnet_4x import MSRResNet4x
from sentinel_sr_dataset_4x import SentinelSR4xDataset

# ─── Constants ────────────────────────────────────────────────────────────────
REGIONS           = ["forest", "urban", "agriculture", "water"]
SAMPLES_PER_REGION = 2
CHANNELS          = ["B02", "B03", "B04", "B08"]

# R=B04, G=B03, B=B02 → indices in [B02,B03,B04,B08] ordering
RGB_INDICES = [2, 1, 0]

STRETCH_PLOW  = 2.0
STRETCH_PHIGH = 98.0

CHECKPOINT_PATH = _project_root / "outputs" / "checkpoints" / "checkpoint_best_4x.pth"
OUTPUT_BASE     = _project_root / "outputs" / "test_visuals_4x"
RGB_DIR         = OUTPUT_BASE / "rgb"
BANDS_DIR       = OUTPUT_BASE / "bands"
DIFF_DIR        = OUTPUT_BASE / "differences"
SAMPLES_CSV     = OUTPUT_BASE / "selected_samples.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Display utilities
# ─────────────────────────────────────────────────────────────────────────────

def _to_rgb(patch_chw: np.ndarray) -> np.ndarray:
    """
    Convert (4, H, W) float32 reflectance to (H, W, 3) using B04-B03-B02.
    No stretch applied here; caller handles display limits.
    """
    rgb = patch_chw[RGB_INDICES, :, :]   # (3, H, W)
    return rgb.transpose(1, 2, 0)        # (H, W, 3)


def _compute_stretch(arr_hwc: np.ndarray):
    """2nd–98th percentile stretch limits computed from a single (H,W,C) array."""
    lo = float(np.percentile(arr_hwc, STRETCH_PLOW))
    hi = float(np.percentile(arr_hwc, STRETCH_PHIGH))
    return lo, hi


def _apply_stretch(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Apply linear stretch and clip to [0,1]. Returns float32 (H,W,C)."""
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _psnr_ssim(pred_t: torch.Tensor, target_t: torch.Tensor,
               train_mean: torch.Tensor, train_std: torch.Tensor):
    """
    Compute PSNR and SSIM for a single (1,4,H,W) pair.
    Denormalize → clip [0,1] → metric. Consistent with metrics.py convention.
    """
    p_ref  = denormalize(pred_t[0].cpu(),   train_mean.cpu(), train_std.cpu())
    t_ref  = denormalize(target_t[0].cpu(), train_mean.cpu(), train_std.cpu())
    p_clip = torch.clamp(p_ref,  0.0, 1.0).unsqueeze(0)
    t_clip = torch.clamp(t_ref,  0.0, 1.0).unsqueeze(0)
    psnr   = calculate_psnr(p_clip, t_clip, data_range=1.0)
    ssim   = calculate_ssim(p_clip, t_clip, data_range=1.0)
    return float(psnr), float(ssim)


# ─────────────────────────────────────────────────────────────────────────────
# Figure builders
# ─────────────────────────────────────────────────────────────────────────────

def _save_rgb_comparison(
    bicubic_disp: np.ndarray,
    sr_disp:      np.ndarray,
    hr_disp:      np.ndarray,
    abs_diff_rgb: np.ndarray,
    stretch_lo:   float,
    stretch_hi:   float,
    meta:         dict,
    save_path:    Path,
):
    """
    4-panel RGB comparison:
    [Bicubic 4×] [MSRResNet4x SR] [HR Reference] [|SR − HR| (abs diff)]
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.suptitle(
        f"4× SR Comparison  |  Region: {meta['region'].capitalize()}"
        f"  |  {meta['filename']}\n"
        f"Stretch [{stretch_lo:.4f}, {stretch_hi:.4f}] from HR (2nd–98th %)  "
        f"|  R=B04  G=B03  B=B02",
        fontsize=9, y=1.01,
    )

    panels = [
        (bicubic_disp,
         f"4× Bicubic Baseline\nPSNR: {meta['bicubic_psnr']:.2f} dB"
         f"  SSIM: {meta['bicubic_ssim']:.4f}"),
        (sr_disp,
         f"MSRResNet4x SR\nPSNR: {meta['model_psnr']:.2f} dB"
         f"  SSIM: {meta['model_ssim']:.4f}"),
        (hr_disp, "HR Ground Truth (10m)\n(Reference)"),
        (None,     "|SR − HR|  (reflectance)"),
    ]

    # Absolute difference colormap
    diff_vmax = float(np.percentile(abs_diff_rgb, 99))

    for ax, (img, title) in zip(axes, panels):
        ax.set_title(title, fontsize=8, pad=5)
        ax.axis("off")
        if img is not None:
            ax.imshow(img, interpolation="nearest")
        else:
            # Absolute difference — mean across RGB channels, single colormap
            diff_mean = abs_diff_rgb.mean(axis=2)
            im = ax.imshow(diff_mean, cmap="inferno",
                           vmin=0.0, vmax=diff_vmax, interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label="Mean |ΔReflectance|")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_band_comparison(
    bicubic_ref: np.ndarray,
    sr_ref:      np.ndarray,
    hr_ref:      np.ndarray,
    meta:        dict,
    save_path:   Path,
    ):
    """
    4-row × 4-column band-wise comparison:
    Rows: B02, B03, B04, B08
    Cols: Bicubic 4×  |  MSRResNet4x SR  |  HR Reference  |  |SR − HR|

    All panels within a row share the same display scale (HR 2–98 percentile).
    """
    n_bands = 4
    n_cols  = 4
    fig, axes = plt.subplots(n_bands, n_cols, figsize=(20, 18))

    fig.suptitle(
        f"Band-wise Comparison  |  Region: {meta['region'].capitalize()}"
        f"  |  {meta['filename']}\n"
        f"All scales: per-band stretch from HR (2nd–98th %)  "
        f"|  Model PSNR: {meta['model_psnr']:.2f} dB",
        fontsize=9, y=1.005,
    )

    col_labels = [
        "4× Bicubic Baseline",
        "MSRResNet4x SR",
        "HR Ground Truth (10m)",
        "|SR − HR|  (reflectance)",
    ]

    for b_idx, band_name in enumerate(CHANNELS):
        bi_band  = bicubic_ref[b_idx]   # (H, W)
        sr_band  = sr_ref[b_idx]
        hr_band  = hr_ref[b_idx]

        # Per-band stretch from HR only
        b_lo = float(np.percentile(hr_band, STRETCH_PLOW))
        b_hi = float(np.percentile(hr_band, STRETCH_PHIGH))

        def _stretch_1ch(arr):
            return np.clip((arr - b_lo) / (b_hi - b_lo + 1e-8), 0.0, 1.0)

        abs_diff = np.abs(sr_band - hr_band)
        diff_vmax = float(np.percentile(abs_diff, 99))

        for c_idx, (arr, label) in enumerate([
            (_stretch_1ch(bi_band),  col_labels[0]),
            (_stretch_1ch(sr_band),  col_labels[1]),
            (_stretch_1ch(hr_band),  col_labels[2]),
            (None,                   col_labels[3]),
        ]):
            ax = axes[b_idx, c_idx]
            if c_idx == 0:
                ax.set_ylabel(band_name, fontsize=10, fontweight="bold")
            if b_idx == 0:
                ax.set_title(label, fontsize=8, pad=4)
            ax.axis("off")

            if arr is not None:
                ax.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0,
                          interpolation="nearest")
            else:
                im = ax.imshow(abs_diff, cmap="inferno",
                               vmin=0.0, vmax=max(diff_vmax, 1e-6),
                               interpolation="nearest")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             label="Δ Refl.")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_diff_summary(
    sr_ref:   np.ndarray,
    hr_ref:   np.ndarray,
    meta:     dict,
    save_path: Path,
):
    """
    4-panel absolute-difference map (one per spectral band).
    Common colormap scale across all bands for comparability.
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"Absolute Difference  |  |MSRResNet4x SR − HR|  |  "
        f"Region: {meta['region'].capitalize()}  |  {meta['filename']}",
        fontsize=9, y=1.02,
    )

    abs_diff = np.abs(sr_ref - hr_ref)   # (4, H, W)
    # Common scale: 99th percentile of the worst channel
    global_vmax = float(np.percentile(abs_diff, 99.5))

    for b_idx, band_name in enumerate(CHANNELS):
        ax   = axes[b_idx]
        diff = abs_diff[b_idx]
        im   = ax.imshow(diff, cmap="hot", vmin=0.0, vmax=global_vmax,
                         interpolation="nearest")
        ax.set_title(f"|SR − HR|  {band_name}\n"
                     f"max={diff.max():.4f}  mean={diff.mean():.4f}",
                     fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Δ Refl.")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main Visualization Function
# ─────────────────────────────────────────────────────────────────────────────

def visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Ensure training has completed before running visualization."
        )

    # ── Output directories ────────────────────────────────────────────────────
    for d in [RGB_DIR, BANDS_DIR, DIFF_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt         = torch.load(CHECKPOINT_PATH, map_location=device)
    model_config = ckpt["model_config"]
    best_epoch   = ckpt.get("epoch", "?")

    # Normalization parameters (for denormalization in metrics)
    import json
    norm_path = _project_root / "data" / "processed" / "normalization_stats_4x.json"
    with open(norm_path) as f:
        norm_json = json.load(f)
    canonical = norm_json["canonical_normalization"]
    channels  = norm_json["metadata"]["channels"]
    train_mean = torch.tensor(
        [canonical["mean"][ch] for ch in channels], dtype=torch.float32
    ).view(4, 1, 1)
    train_std = torch.tensor(
        [canonical["std"][ch] for ch in channels], dtype=torch.float32
    ).view(4, 1, 1)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MSRResNet4x(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Loaded MSRResNet4x (epoch {best_epoch}) on {device}")

    # ── Test dataset ──────────────────────────────────────────────────────────
    test_ds = SentinelSR4xDataset(split="test", normalize=True)
    df      = test_ds.df   # fixed-order metadata DataFrame

    # ── Deterministic sample selection — first + midpoint per region ──────────
    print("\n  Deterministic sample selection:")
    selected = {}   # {region: [dataset_idx, ...]}
    for region in REGIONS:
        # Use np.flatnonzero to guarantee positional integer indices (0-based)
        positions = np.flatnonzero(df["region"] == region).tolist()
        assert len(positions) >= SAMPLES_PER_REGION, (
            f"Region '{region}' has fewer than {SAMPLES_PER_REGION} test samples."
        )
        idx_first = positions[0]
        idx_mid   = positions[len(positions) // 2]
        selected[region] = [idx_first, idx_mid]
        print(f"    {region:12s}: indices [{idx_first}, {idx_mid}] "
              f"→ {df.iloc[idx_first]['filename']}  |  "
              f"{df.iloc[idx_mid]['filename']}")

    # ── CSV header ────────────────────────────────────────────────────────────
    csv_columns = [
        "filename", "region",
        "model_psnr_db", "bicubic_psnr_db",
        "model_ssim",    "bicubic_ssim",
    ]
    csv_rows = []

    # ── Visualization loop ────────────────────────────────────────────────────
    print("\n  Generating visualizations...")
    n_saved = 0

    for region in REGIONS:
        for sample_num, ds_idx in enumerate(selected[region]):
            row      = df.iloc[ds_idx]
            filename = row["filename"]
            stem     = Path(filename).stem

            print(f"\n    [{region}] sample {sample_num}  |  {filename}")

            # ── Load tensors ─────────────────────────────────────────────────
            lr_norm, hr_norm = test_ds[ds_idx]
            lr_t = lr_norm.unsqueeze(0).to(device)   # (1,4,64,64)
            hr_t = hr_norm.unsqueeze(0).to(device)   # (1,4,256,256)

            # Shape assertions
            assert lr_t.shape == (1, 4, 64, 64),  f"Unexpected LR shape: {lr_t.shape}"
            assert hr_t.shape == (1, 4, 256, 256), f"Unexpected HR shape: {hr_t.shape}"

            with torch.no_grad():
                sr_t       = model(lr_t)
                bicubic_t  = F.interpolate(
                    lr_t, scale_factor=4, mode="bicubic", align_corners=False
                )

            # Shape assertion on model output
            assert sr_t.shape == (1, 4, 256, 256), (
                f"Model output shape mismatch: {sr_t.shape}"
            )

            # NaN/Inf checks
            for tensor, name in [
                (lr_t,      "LR input"),
                (hr_t,      "HR target"),
                (sr_t,      "SR prediction"),
                (bicubic_t, "Bicubic"),
            ]:
                assert not (torch.isnan(tensor).any() or torch.isinf(tensor).any()), (
                    f"NaN/Inf in {name} for {filename}"
                )

            # ── Denormalize all to physical reflectance (float32, unclipped) ──
            sr_ref      = denormalize(sr_t[0].cpu(),       train_mean, train_std).numpy()
            bicubic_ref = denormalize(bicubic_t[0].cpu(),  train_mean, train_std).numpy()
            hr_ref      = denormalize(hr_t[0].cpu(),       train_mean, train_std).numpy()
            # Shapes: (4,256,256) float32 — no clip, no alteration

            # ── PSNR / SSIM ───────────────────────────────────────────────────
            model_psnr,   model_ssim   = _psnr_ssim(sr_t,      hr_t, train_mean, train_std)
            bicubic_psnr, bicubic_ssim = _psnr_ssim(bicubic_t, hr_t, train_mean, train_std)
            print(f"      Model PSNR:   {model_psnr:.4f} dB   SSIM: {model_ssim:.6f}")
            print(f"      Bicubic PSNR: {bicubic_psnr:.4f} dB   SSIM: {bicubic_ssim:.6f}")

            meta = {
                "region":       region,
                "filename":     filename,
                "model_psnr":   model_psnr,
                "model_ssim":   model_ssim,
                "bicubic_psnr": bicubic_psnr,
                "bicubic_ssim": bicubic_ssim,
            }

            # ── RGB extraction (B04-B03-B02, no channel alteration) ───────────
            sr_rgb      = _to_rgb(sr_ref)       # (256,256,3)
            bicubic_rgb = _to_rgb(bicubic_ref)  # (256,256,3)
            hr_rgb      = _to_rgb(hr_ref)       # (256,256,3)

            # Stretch from HR only; identical limits applied to all panels
            stretch_lo, stretch_hi = _compute_stretch(hr_rgb)
            sr_disp      = _apply_stretch(sr_rgb,      stretch_lo, stretch_hi)
            bicubic_disp = _apply_stretch(bicubic_rgb, stretch_lo, stretch_hi)
            hr_disp      = _apply_stretch(hr_rgb,      stretch_lo, stretch_hi)

            # Absolute difference in physical reflectance RGB space (no clip)
            abs_diff_rgb = np.abs(sr_rgb - hr_rgb)   # (256,256,3)

            # ── Save RGB comparison ───────────────────────────────────────────
            rgb_path = RGB_DIR / f"{region}_sample{sample_num}_{stem}_rgb.png"
            _save_rgb_comparison(
                bicubic_disp, sr_disp, hr_disp, abs_diff_rgb,
                stretch_lo, stretch_hi, meta, rgb_path,
            )
            print(f"      RGB comparison: {rgb_path.name}")

            # ── Save band-wise comparison ─────────────────────────────────────
            bands_path = BANDS_DIR / f"{region}_sample{sample_num}_{stem}_bands.png"
            _save_band_comparison(
                bicubic_ref, sr_ref, hr_ref, meta, bands_path,
            )
            print(f"      Band comparison: {bands_path.name}")

            # ── Save difference maps ──────────────────────────────────────────
            diff_path = DIFF_DIR / f"{region}_sample{sample_num}_{stem}_diff.png"
            _save_diff_summary(sr_ref, hr_ref, meta, diff_path)
            print(f"      Difference map: {diff_path.name}")

            # ── CSV row ───────────────────────────────────────────────────────
            csv_rows.append({
                "filename":        filename,
                "region":          region,
                "model_psnr_db":   round(model_psnr,   4),
                "bicubic_psnr_db": round(bicubic_psnr, 4),
                "model_ssim":      round(model_ssim,   6),
                "bicubic_ssim":    round(bicubic_ssim, 6),
            })

            n_saved += 3   # rgb + bands + diff

    # ── Save selected_samples.csv ─────────────────────────────────────────────
    with open(SAMPLES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(csv_rows)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_samples_vis = len(REGIONS) * SAMPLES_PER_REGION
    print("\n" + "=" * 62)
    print("4× VISUALIZATION COMPLETE")
    print("=" * 62)
    print(f"  Samples visualized : {total_samples_vis}  "
          f"({SAMPLES_PER_REGION} per region × {len(REGIONS)} regions)")
    print(f"  Images saved       : {n_saved}  "
          f"(rgb + bands + diff per sample)")
    print(f"  Output root        : {OUTPUT_BASE}")
    print(f"    rgb/             : {RGB_DIR}")
    print(f"    bands/           : {BANDS_DIR}")
    print(f"    differences/     : {DIFF_DIR}")
    print(f"    selected_samples.csv")
    print("  Per-sample PSNR summary:")
    print(f"  {'Region':<12} | {'Filename':<30} | "
          f"{'Mdl PSNR':>9} | {'Bic PSNR':>9} | {'ΔPSNR':>7}")
    print("  " + "-" * 72)
    for r in csv_rows:
        delta = r["model_psnr_db"] - r["bicubic_psnr_db"]
        print(f"  {r['region']:<12} | {r['filename']:<30} | "
              f"{r['model_psnr_db']:>9.4f} | {r['bicubic_psnr_db']:>9.4f} | "
              f"{delta:>+7.4f}")
    print("=" * 62)
    print("\n  SCIENTIFIC SCOPE:")
    print("  These visuals demonstrate the synthetic 40m-equivalent LR4x →")
    print("  10m HR benchmark. They do NOT prove genuine 2.5m spatial")
    print("  reconstruction. Region labels are dataset acquisition categories,")
    print("  not scientifically validated land-cover classes.")
    print("=" * 62)


if __name__ == "__main__":
    visualize()
