"""
4× Test-Set Evaluation — MSRResNet4x vs. 4× Bicubic Baseline

Scientific Scope
────────────────
This evaluation measures performance on the SYNTHETIC 40m-equivalent
LR4x → 10m HR benchmark using the held-out test split (676 patches).

It does NOT provide direct validation against genuine 2.5m ground-truth
imagery. The trained model, when deployed on real 10m Sentinel-2 inputs,
produces a 2.5m-scale INFERRED product whose fine spatial detail is
model-generated. No sub-10m reference dataset was used to supervise or
evaluate genuine spatial content at that scale.

Metadata Alignment Guarantee
──────────────────────────────
test_dataset.df is the fixed-order DataFrame used internally by
SentinelSR4xDataset.__getitem__. shuffle=False + num_workers=0 ensures
strictly sequential DataLoader iteration. A sample_idx cursor tracks
which DataFrame rows correspond to each batch. The final partial batch
(676 % 8 = 4 samples) is handled via lr_batch.shape[0].

Sample-Weighted Aggregation
────────────────────────────
Per-sample metrics are aggregated with equal weight per sample.
This is mathematically equivalent to weighting each batch by its actual
batch size, which is critical for the final partial batch.

Isolation Guarantee
────────────────────
Reads:
  outputs/checkpoints/checkpoint_best_4x.pth
  data/processed/normalization_stats_4x.json  (via dataset)
  data/processed/hr_split_index.csv            (via dataset)
  data/processed/lr4x/                         (via dataset, read only)
  data/processed/hr/                           (via dataset, read only)
  src/training/metrics.py                      (unchanged 2× metrics module)

Writes:
  outputs/evaluation_4x_results.json
  outputs/evaluation_4x_per_sample.csv

Modifies: nothing.
"""

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# ─── Path registration ────────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(_project_root / "src" / "data"))
sys.path.append(str(_project_root / "src" / "models"))
sys.path.append(str(_project_root / "src" / "training"))

from metrics import evaluate_batch_metrics
from msr_resnet_4x import MSRResNet4x
from sentinel_sr_dataset_4x import SentinelSR4xDataset
from torch.utils.data import DataLoader

# ─── Constants ────────────────────────────────────────────────────────────────
EXPECTED_TEST_COUNT = 676
REGIONS  = ["forest", "urban", "agriculture", "water"]
CHANNELS = ["B02", "B03", "B04", "B08"]
BATCH_SIZE = 8

CHECKPOINT_PATH = _project_root / "outputs" / "checkpoints" / "checkpoint_best_4x.pth"
OUTPUT_JSON_PATH = _project_root / "outputs" / "evaluation_4x_results.json"
OUTPUT_CSV_PATH  = _project_root / "outputs" / "evaluation_4x_per_sample.csv"

CSV_COLUMNS = [
    "filename", "region",
    "model_l1",              "bicubic_l1",
    "model_reflectance_mae", "bicubic_reflectance_mae",
    "model_psnr_db",         "bicubic_psnr_db",
    "model_ssim",            "bicubic_ssim",
    "delta_l1",              # model − bicubic  (negative = model better)
    "delta_reflectance_mae",
    "delta_psnr_db",         # model − bicubic  (positive = model better)
    "delta_ssim",
]


# ─────────────────────────────────────────────────────────────────────────────
# Metric accumulator helpers (identical pattern to evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

def _empty_accum() -> dict:
    return {
        "l1_sum":      0.0,
        "ref_mae_sum": 0.0,
        "psnr_sum":    0.0,
        "ssim_sum":    0.0,
        "oob_pct_sum": 0.0,
        "n_samples":   0,
    }


def _accum(accum: dict, metrics: dict, n: int):
    """
    Sample-weighted accumulation: multiply each batch-level metric by the
    actual number of samples in that batch before summing.
    This ensures the final partial batch (n < BATCH_SIZE) contributes
    proportionally, exactly as in the corrected 2× evaluate.py.
    """
    accum["ref_mae_sum"] += metrics["reflectance_mae"] * n
    accum["psnr_sum"]    += metrics["psnr_db"]         * n
    accum["ssim_sum"]    += metrics["ssim"]             * n
    accum["oob_pct_sum"] += metrics["out_of_bounds_pct"] * n
    accum["n_samples"]   += n


def _finalize(accum: dict, l1_weighted_sum: float) -> dict:
    """Divide all accumulated weighted sums by total sample count."""
    ns = accum["n_samples"]
    return {
        "l1_loss":          round(l1_weighted_sum / ns, 6),
        "reflectance_mae":  round(accum["ref_mae_sum"] / ns, 6),
        "psnr_db":          round(accum["psnr_sum"]    / ns, 4),
        "ssim":             round(accum["ssim_sum"]    / ns, 6),
        "out_of_bounds_pct": round(accum["oob_pct_sum"] / ns, 6),
        "n_samples":        ns,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation Function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ── Checkpoint ───────────────────────────────────────────────────────────
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Ensure training has completed and checkpoint_best_4x.pth exists."
        )

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    model_config   = ckpt["model_config"]
    train_mean_list = ckpt.get("training_config", {})
    best_epoch     = ckpt.get("epoch", "?")
    best_val_loss  = ckpt.get("best_val_loss", float("nan"))

    # ── Normalization parameters — from checkpoint or normalization_stats_4x.json
    # The dataset already normalizes tensors. We need train_mean/std for
    # denormalization inside evaluate_batch_metrics (same as 2× evaluate.py).
    norm_stats_path = _project_root / "data" / "processed" / "normalization_stats_4x.json"
    with open(norm_stats_path, "r") as f:
        norm_json = json.load(f)

    canonical = norm_json["canonical_normalization"]
    channels  = norm_json["metadata"]["channels"]

    train_mean = torch.tensor(
        [canonical["mean"][ch] for ch in channels], dtype=torch.float32
    ).view(4, 1, 1)
    train_std = torch.tensor(
        [canonical["std"][ch] for ch in channels], dtype=torch.float32
    ).view(4, 1, 1)

    print(f"  Checkpoint epoch:       {best_epoch}")
    print(f"  Best val L1 (training): {best_val_loss:.6f}")
    print(f"  Model config:           {model_config}")
    print(f"  Norm channels:          {channels}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = MSRResNet4x(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:             {total_params:,}")
    print(f"  Device:                 {device}")
    if device.type == "cuda":
        print(f"  GPU:                    {torch.cuda.get_device_name(0)}")

    # ── Test Dataset & DataLoader ─────────────────────────────────────────────
    test_dataset = SentinelSR4xDataset(split="test", normalize=True)
    assert len(test_dataset) == EXPECTED_TEST_COUNT, (
        f"Expected {EXPECTED_TEST_COUNT} test samples, found {len(test_dataset)}"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = (device.type == "cuda"),
    )
    print(f"  Test samples:           {len(test_dataset)}  (verified = {EXPECTED_TEST_COUNT})")

    # ── Accumulators ─────────────────────────────────────────────────────────
    model_overall   = _empty_accum()
    bicubic_overall = _empty_accum()
    model_by_region   = {r: _empty_accum() for r in REGIONS}
    bicubic_by_region = {r: _empty_accum() for r in REGIONS}
    model_l1_wsum   = 0.0
    bicubic_l1_wsum = 0.0

    # ── Internal validation tracking ─────────────────────────────────────────
    n_evaluated    = 0
    seen_filenames = set()
    nan_inf_errors = []
    per_sample_rows = []

    # ── Evaluation Loop — metadata aligned via sequential sample_idx cursor ──
    print("\nRunning 4× test evaluation...")
    t_eval_start = time.perf_counter()
    sample_idx   = 0

    with torch.no_grad():
        for lr_batch, hr_batch in tqdm(test_loader, desc="4× Test Evaluation"):
            actual_bs = lr_batch.shape[0]

            # Metadata alignment: exactly actual_bs rows from dataset.df
            batch_meta = test_dataset.df.iloc[sample_idx: sample_idx + actual_bs]
            filenames     = batch_meta["filename"].tolist()
            regions_batch = batch_meta["region"].tolist()
            sample_idx   += actual_bs

            # Dimension assertions
            assert lr_batch.shape[1:] == (4, 64, 64), (
                f"Unexpected LR shape {lr_batch.shape}"
            )
            assert hr_batch.shape[1:] == (4, 256, 256), (
                f"Unexpected HR shape {hr_batch.shape}"
            )

            # NaN/Inf in inputs
            if torch.isnan(lr_batch).any() or torch.isinf(lr_batch).any():
                nan_inf_errors.append(f"NaN/Inf in lr_batch at sample_idx {sample_idx}")
            if torch.isnan(hr_batch).any() or torch.isinf(hr_batch).any():
                nan_inf_errors.append(f"NaN/Inf in hr_batch at sample_idx {sample_idx}")

            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            # ── Model forward pass ────────────────────────────────────────────
            model_pred = model(lr_batch)              # (B,4,256,256)

            # ── Bicubic 4× baseline ───────────────────────────────────────────
            bicubic_pred = F.interpolate(
                lr_batch, scale_factor=4,
                mode="bicubic", align_corners=False,
            )                                         # (B,4,256,256)

            # NaN/Inf in predictions
            for tensor, name in [
                (model_pred,   "model_pred"),
                (bicubic_pred, "bicubic_pred"),
            ]:
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    nan_inf_errors.append(
                        f"NaN/Inf in {name} at sample_idx {sample_idx}"
                    )

            # ── Batch-level normalized L1 ─────────────────────────────────────
            model_l1_batch   = F.l1_loss(model_pred,   hr_batch).item()
            bicubic_l1_batch = F.l1_loss(bicubic_pred, hr_batch).item()
            model_l1_wsum   += model_l1_batch   * actual_bs
            bicubic_l1_wsum += bicubic_l1_batch * actual_bs

            # ── Batch-level reflectance metrics (reused from metrics.py exactly)
            model_met   = evaluate_batch_metrics(model_pred,   hr_batch, train_mean, train_std)
            bicubic_met = evaluate_batch_metrics(bicubic_pred, hr_batch, train_mean, train_std)

            _accum(model_overall,   model_met,   actual_bs)
            _accum(bicubic_overall, bicubic_met, actual_bs)

            # ── Per-sample metrics (for CSV and per-region accumulators) ──────
            for i in range(actual_bs):
                region   = regions_batch[i]
                filename = filenames[i]

                m_s = model_pred[i: i + 1]
                b_s = bicubic_pred[i: i + 1]
                h_s = hr_batch[i: i + 1]

                m_met_s = evaluate_batch_metrics(m_s, h_s, train_mean, train_std)
                b_met_s = evaluate_batch_metrics(b_s, h_s, train_mean, train_std)
                m_l1_s  = F.l1_loss(m_s, h_s).item()
                b_l1_s  = F.l1_loss(b_s, h_s).item()

                # Per-region accumulation (n=1 per sample — already sample-weighted)
                _accum(model_by_region[region],   m_met_s, 1)
                _accum(bicubic_by_region[region], b_met_s, 1)
                model_by_region[region]["l1_sum"]   += m_l1_s
                bicubic_by_region[region]["l1_sum"] += b_l1_s

                # Duplicate filename check
                assert filename not in seen_filenames, (
                    f"Duplicate filename in test set: {filename}"
                )
                seen_filenames.add(filename)
                n_evaluated += 1

                per_sample_rows.append({
                    "filename":              filename,
                    "region":                region,
                    "model_l1":              round(m_l1_s, 6),
                    "bicubic_l1":            round(b_l1_s, 6),
                    "model_reflectance_mae": round(m_met_s["reflectance_mae"], 6),
                    "bicubic_reflectance_mae": round(b_met_s["reflectance_mae"], 6),
                    "model_psnr_db":         round(m_met_s["psnr_db"], 4),
                    "bicubic_psnr_db":       round(b_met_s["psnr_db"], 4),
                    "model_ssim":            round(m_met_s["ssim"], 6),
                    "bicubic_ssim":          round(b_met_s["ssim"], 6),
                    "delta_l1":              round(m_l1_s - b_l1_s, 6),
                    "delta_reflectance_mae": round(
                        m_met_s["reflectance_mae"] - b_met_s["reflectance_mae"], 6
                    ),
                    "delta_psnr_db":  round(m_met_s["psnr_db"]  - b_met_s["psnr_db"],  4),
                    "delta_ssim":     round(m_met_s["ssim"]      - b_met_s["ssim"],      6),
                })

    eval_time_sec = time.perf_counter() - t_eval_start

    # ── Internal validation assertions ────────────────────────────────────────
    assert n_evaluated == EXPECTED_TEST_COUNT, (
        f"Evaluated {n_evaluated} samples, expected {EXPECTED_TEST_COUNT}"
    )
    assert len(seen_filenames) == EXPECTED_TEST_COUNT, (
        f"Duplicate filenames: {EXPECTED_TEST_COUNT - len(seen_filenames)} duplicates"
    )
    assert len(nan_inf_errors) == 0, (
        "NaN/Inf errors:\n" + "\n".join(nan_inf_errors)
    )
    print(f"\n  Internal validation: {n_evaluated} samples, no duplicates, no NaN/Inf  ✓")

    # ── Finalize overall metrics ──────────────────────────────────────────────
    n_samples       = EXPECTED_TEST_COUNT
    overall_model   = _finalize(model_overall,   model_l1_wsum)
    overall_bicubic = _finalize(bicubic_overall, bicubic_l1_wsum)

    # ── Finalize per-region metrics ───────────────────────────────────────────
    region_model_results   = {}
    region_bicubic_results = {}
    for r in REGIONS:
        rm = model_by_region[r]
        rb = bicubic_by_region[r]
        ns_r = rm["n_samples"]
        ns_rb = rb["n_samples"]

        def _reg_finalize(a, l1_wsum):
            ns = a["n_samples"]
            if ns == 0:
                return {k: None for k in
                        ["l1_loss","reflectance_mae","psnr_db",
                         "ssim","out_of_bounds_pct","n_samples"]}
            return {
                "l1_loss":          round(l1_wsum / ns, 6),
                "reflectance_mae":  round(a["ref_mae_sum"] / ns, 6),
                "psnr_db":          round(a["psnr_sum"]    / ns, 4),
                "ssim":             round(a["ssim_sum"]    / ns, 6),
                "out_of_bounds_pct": round(a["oob_pct_sum"] / ns, 6),
                "n_samples":        ns,
            }

        region_model_results[r]   = _reg_finalize(rm, rm["l1_sum"])
        region_bicubic_results[r] = _reg_finalize(rb, rb["l1_sum"])

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_json = {
        "scientific_scope": (
            "Evaluation on synthetic 40m-equivalent LR4x → 10m HR benchmark. "
            "Does NOT validate genuine 2.5m ground-truth SR performance. "
            "The deployed 10m → 2.5m-scale product contains model-inferred "
            "fine spatial detail without sub-10m reference validation."
        ),
        "evaluation_info": {
            "date":                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint_path":     str(CHECKPOINT_PATH),
            "best_epoch":          best_epoch,
            "best_val_l1_training": best_val_loss,
            "test_split":          "test",
            "test_sample_count":   n_evaluated,
            "batch_size":          BATCH_SIZE,
            "device":              str(device),
            "evaluation_time_sec": round(eval_time_sec, 2),
            "peak_vram_mb":        round(peak_vram_mb, 2),
        },
        "model_config": model_config,
        "normalization": {
            "source": "data/processed/normalization_stats_4x.json",
            "convention": (
                "uint16 / 10000.0 → channel-wise Z-score using canonical "
                "HR train statistics. Same mean/std applied to both LR4x "
                "and HR targets."
            ),
            "channels":    channels,
            "train_mean":  [canonical["mean"][ch] for ch in channels],
            "train_std":   [canonical["std"][ch]  for ch in channels],
        },
        "metric_definitions": {
            "l1_loss":          "L1 in Z-score normalized space",
            "reflectance_mae":  "MAE in physical reflectance space "
                                "(unclipped, uint16 / 10000)",
            "psnr_db":          "PSNR (dB) with data_range=1.0 on "
                                "[0,1] clipped reflectance",
            "ssim":             "4-channel averaged SSIM with data_range=1.0 "
                                "on [0,1] clipped reflectance",
            "out_of_bounds_pct": "% of target pixels outside [0.0, 1.0] "
                                "in physical reflectance space",
        },
        "baseline": {
            "method":       "4× bicubic interpolation",
            "torch_call":   "F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False)",
        },
        "overall_model_metrics":   overall_model,
        "overall_bicubic_metrics": overall_bicubic,
        "per_region_model_metrics":   region_model_results,
        "per_region_bicubic_metrics": region_bicubic_results,
    }

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Saved JSON: {OUTPUT_JSON_PATH}")

    # ── Save per-sample CSV ───────────────────────────────────────────────────
    with open(OUTPUT_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(per_sample_rows)
    print(f"  Saved CSV:  {OUTPUT_CSV_PATH}")

    # ── Console Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("4× SENTINEL-2 SUPER-RESOLUTION TEST EVALUATION")
    print("=" * 62)
    print(f"Checkpoint  : {CHECKPOINT_PATH.name}")
    print(f"Best epoch  : {best_epoch}")
    print(f"Test samples: {n_evaluated}")
    print(f"Device      : {device}")
    print(f"Eval time   : {eval_time_sec:.1f} s")
    if device.type == "cuda":
        print(f"Peak VRAM   : {peak_vram_mb:.2f} MB")

    print("\nOVERALL RESULTS")
    print(f"{'Metric':<24} | {'Model':>10} | {'Bicubic':>10} | {'Delta':>10} | {'Better?':>7}")
    print("-" * 68)
    display_metrics = [
        ("Normalized L1",   "l1_loss",         True),
        ("Reflectance MAE", "reflectance_mae",  True),
        ("PSNR (dB)",       "psnr_db",          False),
        ("SSIM",            "ssim",             False),
        ("Target OOB (%)",  "out_of_bounds_pct", True),
    ]
    for label, key, lower_is_better in display_metrics:
        mv = overall_model[key]
        bv = overall_bicubic[key]
        delta = mv - bv
        better = ("✓" if (delta < 0) == lower_is_better else "✗")
        print(f"{label:<24} | {mv:>10.6f} | {bv:>10.6f} | {delta:>+10.6f} | {better:>7}")

    print("\nPER-REGION RESULTS")
    print(f"{'Region':<14} | {'Src Type':<10} | "
          f"{'PSNR Mdl':>9} | {'PSNR Bic':>9} | {'ΔPSNR':>7} | "
          f"{'SSIM Mdl':>9} | {'SSIM Bic':>9} | {'ΔSSIM':>8} | N")
    print("-" * 100)
    for r in REGIONS:
        rm = region_model_results[r]
        rb = region_bicubic_results[r]
        # "Src Type" deliberately avoids "land-cover class" — these are
        # dataset acquisition categories, not scientifically validated classes.
        print(
            f"{r:<14} | {'acq.cat.':<10} | "
            f"{rm['psnr_db']:>9.4f} | {rb['psnr_db']:>9.4f} | "
            f"{rm['psnr_db'] - rb['psnr_db']:>+7.4f} | "
            f"{rm['ssim']:>9.6f} | {rb['ssim']:>9.6f} | "
            f"{rm['ssim'] - rb['ssim']:>+8.6f} | "
            f"{rm['n_samples']}"
        )

    print("\n" + "=" * 62)
    print("SCIENTIFIC INTERPRETATION")
    print("=" * 62)
    print(
        "  This evaluation measures performance on the synthetic\n"
        "  40m-equivalent LR4x → 10m HR benchmark.\n\n"
        "  It does NOT provide direct validation against genuine\n"
        "  2.5m ground-truth imagery.\n\n"
        "  The deployed 10m → 2.5m-scale product therefore contains\n"
        "  model-inferred fine spatial detail. This is an extrapolation\n"
        "  beyond the training degradation assumption and must be\n"
        "  clearly stated in any public-facing demonstration."
    )
    print("=" * 62)


if __name__ == "__main__":
    evaluate()