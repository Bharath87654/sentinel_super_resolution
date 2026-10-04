"""
Test-Set Evaluation Script — MSRResNet2x vs. Bicubic Baseline

Evaluates ONLY the test split (676 patches) using checkpoint_best.pth.

Metadata alignment guarantee:
  - test_dataset.df is the fixed-order DataFrame used internally by __getitem__.
  - shuffle=False + num_workers=0 ensures sequential iteration.
  - A sample_idx cursor tracks which DataFrame rows correspond to each batch.
  - The final partial batch is handled via lr_batch.shape[0].

Reuses src/training/metrics.py exactly — no duplicate metric definitions.

Outputs:
  outputs/test_results.json   — overall + per-region model & bicubic metrics
  outputs/test_results.csv    — one row per test patch
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Resolve project root and register src subdirs on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / "src" / "data"))
sys.path.append(str(project_root / "src" / "models"))
sys.path.append(str(project_root / "src" / "training"))

from metrics import evaluate_batch_metrics
from msr_resnet import MSRResNet2x
from sentinel_sr_dataset import SentinelSRDataset
from torch.utils.data import DataLoader

EXPECTED_TEST_COUNT = 676
REGIONS = ["forest", "urban", "agriculture", "water"]
BATCH_SIZE = 8


def _empty_metric_accum():
    return {
        "l1_sum": 0.0,
        "ref_mae_sum": 0.0,
        "psnr_sum": 0.0,
        "ssim_sum": 0.0,
        "oob_pct_sum": 0.0,
        "n_batches": 0,
        "n_samples": 0,
    }


def _accum_metrics(accum: dict, metrics: dict, n_samples: int):
    accum["ref_mae_sum"] += metrics["reflectance_mae"]
    accum["psnr_sum"] += metrics["psnr_db"]
    accum["ssim_sum"] += metrics["ssim"]
    accum["oob_pct_sum"] += metrics["out_of_bounds_pct"]
    accum["n_batches"] += 1
    accum["n_samples"] += n_samples


def _finalize_metrics(accum: dict, l1_total_sum: float, n_batches: int) -> dict:
    nb = accum["n_batches"]
    return {
        "l1_loss": round(l1_total_sum / n_batches, 6),
        "reflectance_mae": round(accum["ref_mae_sum"] / nb, 6),
        "psnr_db": round(accum["psnr_sum"] / nb, 4),
        "ssim": round(accum["ssim_sum"] / nb, 6),
        "out_of_bounds_pct": round(accum["oob_pct_sum"] / nb, 6),
        "n_samples": accum["n_samples"],
    }


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    checkpoint_path = project_root / "outputs" / "checkpoints" / "checkpoint_best.pth"
    output_dir = project_root / "outputs"
    output_json = output_dir / "test_results.json"
    output_csv = output_dir / "test_results.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # -------------------------------------------------------------------------
    # Load Checkpoint
    # -------------------------------------------------------------------------
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_config = ckpt["model_config"]
    train_mean = torch.tensor(ckpt["train_mean"], dtype=torch.float32).view(4, 1, 1)
    train_std = torch.tensor(ckpt["train_std"], dtype=torch.float32).view(4, 1, 1)
    scale_factor_norm = ckpt["scale_factor"]  # 10000
    best_epoch = ckpt["epoch"]
    best_val_loss = ckpt["best_val_loss"]

    print(f"  Checkpoint epoch:        {best_epoch}")
    print(f"  Best validation L1 loss: {best_val_loss:.6f}")
    print(f"  Model config:            {model_config}")

    # -------------------------------------------------------------------------
    # Instantiate Model
    # -------------------------------------------------------------------------
    model = MSRResNet2x(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Model loaded on: {device}")

    # -------------------------------------------------------------------------
    # Test Dataset & DataLoader — shuffle=False, num_workers=0
    # -------------------------------------------------------------------------
    test_dataset = SentinelSRDataset(split="test", normalize=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Confirm expected test count
    assert len(test_dataset) == EXPECTED_TEST_COUNT, (
        f"Expected {EXPECTED_TEST_COUNT} test samples, got {len(test_dataset)}"
    )
    print(f"  Test dataset size: {len(test_dataset)} patches (verified)")

    # -------------------------------------------------------------------------
    # Metric Accumulators — overall + per-region for model and bicubic
    # -------------------------------------------------------------------------
    model_overall = _empty_metric_accum()
    bicubic_overall = _empty_metric_accum()
    model_by_region = {r: _empty_metric_accum() for r in REGIONS}
    bicubic_by_region = {r: _empty_metric_accum() for r in REGIONS}

    model_l1_total = 0.0
    bicubic_l1_total = 0.0
    n_total_batches = 0

    # -------------------------------------------------------------------------
    # CSV Setup
    # -------------------------------------------------------------------------
    csv_columns = [
        "filename", "region",
        "model_l1", "bicubic_l1",
        "model_reflectance_mae", "bicubic_reflectance_mae",
        "model_psnr_db", "bicubic_psnr_db",
        "model_ssim", "bicubic_ssim",
        "delta_l1",           # model_l1 - bicubic_l1         (negative = model better)
        "delta_reflectance_mae",
        "delta_psnr_db",      # model_psnr - bicubic_psnr     (positive = model better)
        "delta_ssim",
    ]

    per_sample_rows = []

    # -------------------------------------------------------------------------
    # Internal Validation Tracking
    # -------------------------------------------------------------------------
    n_evaluated = 0
    seen_filenames = set()
    nan_inf_errors = []

    # -------------------------------------------------------------------------
    # Evaluation Loop — metadata aligned via sequential sample_idx cursor
    # -------------------------------------------------------------------------
    sample_idx = 0

    print("\nRunning test evaluation...")
    with torch.no_grad():
        for lr_batch, hr_batch in tqdm(test_loader, desc="Evaluating Test Set"):
            actual_bs = lr_batch.shape[0]

            # --- Metadata alignment: read exactly actual_bs rows from dataset.df ---
            batch_meta = test_dataset.df.iloc[sample_idx: sample_idx + actual_bs]
            filenames = batch_meta["filename"].tolist()
            regions_batch = batch_meta["region"].tolist()
            sample_idx += actual_bs

            # Validate dimension expectations
            assert lr_batch.shape[1:] == (4, 128, 128), (
                f"Unexpected LR shape {lr_batch.shape}"
            )
            assert hr_batch.shape[1:] == (4, 256, 256), (
                f"Unexpected HR shape {hr_batch.shape}"
            )

            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            # --- Model Forward Pass ---
            model_pred = model(lr_batch)

            # --- Bicubic Baseline (same LR input, 2x bicubic) ---
            bicubic_pred = F.interpolate(
                lr_batch, scale_factor=2, mode="bicubic", align_corners=False
            )

            # --- Normalized L1 losses ---
            model_l1 = F.l1_loss(model_pred, hr_batch).item()
            bicubic_l1 = F.l1_loss(bicubic_pred, hr_batch).item()
            model_l1_total += model_l1
            bicubic_l1_total += bicubic_l1
            n_total_batches += 1

            # --- Physical reflectance metrics (reusing metrics.py exactly) ---
            model_metrics = evaluate_batch_metrics(
                model_pred, hr_batch, train_mean, train_std
            )
            bicubic_metrics = evaluate_batch_metrics(
                bicubic_pred, hr_batch, train_mean, train_std
            )

            # --- NaN/Inf checks ---
            for tensor, name in [
                (model_pred, "model_pred"),
                (bicubic_pred, "bicubic_pred"),
            ]:
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    nan_inf_errors.append(f"NaN/Inf in {name} at sample_idx {sample_idx}")

            # --- Accumulate overall ---
            _accum_metrics(model_overall, model_metrics, actual_bs)
            _accum_metrics(bicubic_overall, bicubic_metrics, actual_bs)

            # --- Accumulate per-region (per sample in the batch) ---
            for i in range(actual_bs):
                region = regions_batch[i]
                filename = filenames[i]

                # Fetch per-sample metrics from single-sample evaluation
                m_pred_s = model_pred[i: i + 1]
                b_pred_s = bicubic_pred[i: i + 1]
                hr_s = hr_batch[i: i + 1]

                m_metrics_s = evaluate_batch_metrics(m_pred_s, hr_s, train_mean, train_std)
                b_metrics_s = evaluate_batch_metrics(b_pred_s, hr_s, train_mean, train_std)

                m_l1_s = F.l1_loss(m_pred_s, hr_s).item()
                b_l1_s = F.l1_loss(b_pred_s, hr_s).item()

                _accum_metrics(model_by_region[region], m_metrics_s, 1)
                _accum_metrics(bicubic_by_region[region], b_metrics_s, 1)
                model_by_region[region]["l1_sum"] += m_l1_s
                bicubic_by_region[region]["l1_sum"] += b_l1_s

                # Track filename uniqueness
                assert filename not in seen_filenames, f"Duplicate filename: {filename}"
                seen_filenames.add(filename)
                n_evaluated += 1

                # CSV row
                per_sample_rows.append({
                    "filename": filename,
                    "region": region,
                    "model_l1": round(m_l1_s, 6),
                    "bicubic_l1": round(b_l1_s, 6),
                    "model_reflectance_mae": round(m_metrics_s["reflectance_mae"], 6),
                    "bicubic_reflectance_mae": round(b_metrics_s["reflectance_mae"], 6),
                    "model_psnr_db": round(m_metrics_s["psnr_db"], 4),
                    "bicubic_psnr_db": round(b_metrics_s["psnr_db"], 4),
                    "model_ssim": round(m_metrics_s["ssim"], 6),
                    "bicubic_ssim": round(b_metrics_s["ssim"], 6),
                    "delta_l1": round(m_l1_s - b_l1_s, 6),
                    "delta_reflectance_mae": round(
                        m_metrics_s["reflectance_mae"] - b_metrics_s["reflectance_mae"], 6
                    ),
                    "delta_psnr_db": round(m_metrics_s["psnr_db"] - b_metrics_s["psnr_db"], 4),
                    "delta_ssim": round(m_metrics_s["ssim"] - b_metrics_s["ssim"], 6),
                })

    # -------------------------------------------------------------------------
    # Internal Validation Assertions
    # -------------------------------------------------------------------------
    assert n_evaluated == EXPECTED_TEST_COUNT, (
        f"Evaluation count mismatch: evaluated {n_evaluated}, expected {EXPECTED_TEST_COUNT}"
    )
    assert len(seen_filenames) == EXPECTED_TEST_COUNT, (
        f"Duplicate filenames detected: {EXPECTED_TEST_COUNT - len(seen_filenames)} duplicates"
    )
    assert len(nan_inf_errors) == 0, f"NaN/Inf errors detected:\n" + "\n".join(nan_inf_errors)
    print(f"\n  Internal validation passed: {n_evaluated} samples, no duplicates, no NaN/Inf.")

    # -------------------------------------------------------------------------
    # Finalize Overall Metrics
    # -------------------------------------------------------------------------
    overall_model = _finalize_metrics(model_overall, model_l1_total, n_total_batches)
    overall_bicubic = _finalize_metrics(bicubic_overall, bicubic_l1_total, n_total_batches)

    # -------------------------------------------------------------------------
    # Finalize Per-Region Metrics
    # -------------------------------------------------------------------------
    region_model_results = {}
    region_bicubic_results = {}
    for r in REGIONS:
        rm = model_by_region[r]
        rb = bicubic_by_region[r]
        nb_r = rm["n_batches"]
        nb_rb = rb["n_batches"]

        region_model_results[r] = {
            "l1_loss": round(rm["l1_sum"] / nb_r, 6) if nb_r > 0 else None,
            "reflectance_mae": round(rm["ref_mae_sum"] / nb_r, 6) if nb_r > 0 else None,
            "psnr_db": round(rm["psnr_sum"] / nb_r, 4) if nb_r > 0 else None,
            "ssim": round(rm["ssim_sum"] / nb_r, 6) if nb_r > 0 else None,
            "out_of_bounds_pct": round(rm["oob_pct_sum"] / nb_r, 6) if nb_r > 0 else None,
            "n_samples": rm["n_samples"],
        }
        region_bicubic_results[r] = {
            "l1_loss": round(rb["l1_sum"] / nb_rb, 6) if nb_rb > 0 else None,
            "reflectance_mae": round(rb["ref_mae_sum"] / nb_rb, 6) if nb_rb > 0 else None,
            "psnr_db": round(rb["psnr_sum"] / nb_rb, 4) if nb_rb > 0 else None,
            "ssim": round(rb["ssim_sum"] / nb_rb, 6) if nb_rb > 0 else None,
            "out_of_bounds_pct": round(rb["oob_pct_sum"] / nb_rb, 6) if nb_rb > 0 else None,
            "n_samples": rb["n_samples"],
        }

    # -------------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------------
    results_json = {
        "evaluation_info": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint_path": str(checkpoint_path),
            "best_epoch": best_epoch,
            "best_val_l1_loss_during_training": best_val_loss,
            "test_split": "test",
            "test_sample_count": n_evaluated,
            "batch_size": BATCH_SIZE,
            "device": str(device),
        },
        "model_config": model_config,
        "normalization": {
            "convention": "uint16 / 10000.0 then channel-wise Z-score using HR train statistics",
            "scale_factor": scale_factor_norm,
            "train_mean": ckpt["train_mean"],
            "train_std": ckpt["train_std"],
            "channels": ["B02", "B03", "B04", "B08"],
        },
        "metric_definitions": {
            "l1_loss": "L1 in Z-score normalized space",
            "reflectance_mae": "MAE in physical reflectance space (unclipped, uint16 / 10000)",
            "psnr_db": "PSNR (dB) with data_range=1.0 on [0,1] clipped reflectance",
            "ssim": "4-channel averaged SSIM with data_range=1.0 on [0,1] clipped reflectance",
            "out_of_bounds_pct": "% of target pixels outside [0.0, 1.0] in physical reflectance space",
        },
        "overall_model_metrics": overall_model,
        "overall_bicubic_metrics": overall_bicubic,
        "per_region_model_metrics": region_model_results,
        "per_region_bicubic_metrics": region_bicubic_results,
    }

    with open(output_json, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved test results JSON: {output_json}")

    # -------------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------------
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(per_sample_rows)
    print(f"Saved test results CSV: {output_csv}")

    # -------------------------------------------------------------------------
    # Console Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("TEST-SET EVALUATION SUMMARY")
    print("=" * 75)
    print(f"{'Metric':<28} | {'Model':>12} | {'Bicubic':>12} | {'Delta':>12}")
    print("-" * 75)
    metrics_display = [
        ("L1 Loss (norm.)", "l1_loss", True),
        ("Reflectance MAE", "reflectance_mae", True),
        ("PSNR (dB)", "psnr_db", False),
        ("SSIM", "ssim", False),
        ("OOB Pixel %", "out_of_bounds_pct", True),
    ]
    for label, key, lower_is_better in metrics_display:
        mv = overall_model[key]
        bv = overall_bicubic[key]
        delta = mv - bv
        better = (delta < 0) if lower_is_better else (delta > 0)
        flag = "✓" if better else "✗"
        print(f"{label:<28} | {mv:>12.6f} | {bv:>12.6f} | {delta:>+12.6f} {flag}")
    print("-" * 75)
    print("\nPer-Region PSNR (dB):")
    print(f"{'Region':<14} | {'Model PSNR':>12} | {'Bicubic PSNR':>14} | {'Delta':>10}")
    print("-" * 55)
    for r in REGIONS:
        mp = region_model_results[r]["psnr_db"]
        bp = region_bicubic_results[r]["psnr_db"]
        print(f"{r:<14} | {mp:>12.4f} | {bp:>14.4f} | {mp - bp:>+10.4f}")
    print("=" * 75)


if __name__ == "__main__":
    evaluate()
