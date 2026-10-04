"""
MSRResNet2x (RCAN-Lite) Training & Validation Script

Model: MSRResNet2x (~0.636 M parameters)
Loss: L1 Loss in Z-score normalized space
Optimizer: AdamW (lr=2e-4, weight_decay=1e-4)
Scheduler: CosineAnnealingLR (T_max=50, eta_min=1e-6)
AMP: torch.autocast('cuda', dtype=torch.float16) + torch.amp.GradScaler('cuda')
Checkpoints: outputs/checkpoints/checkpoint_best.pth & checkpoint_latest.pth
"""

import csv
import json
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add src directories to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / "src" / "data"))
sys.path.append(str(project_root / "src" / "models"))
sys.path.append(str(project_root / "src" / "training"))

from metrics import evaluate_batch_metrics
from msr_resnet import MSRResNet2x
from sentinel_sr_dataset import SentinelSRDataset, create_dataloaders


def set_seed(seed: int = 42):
    """Enforces strict reproducibility across random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    batch_size: int = 8,
    max_epochs: int = 50,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    patience: int = 12,
    seed: int = 42,
    num_workers: int = 0,
):
    set_seed(seed)

    # Output paths
    output_dir = project_root / "outputs"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    csv_log_path = output_dir / "training_log.csv"
    json_summary_path = output_dir / "train_summary.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"STARTING MSRRESNET2X TRAINING ON DEVICE: {device}")
    print("=" * 75)
    print(f"Batch Size:              {batch_size}")
    print(f"Max Epochs:              {max_epochs}")
    print(f"Initial Learning Rate:   {lr}")
    print(f"Gradient Clipping Norm:  {max_grad_norm}")
    print(f"Early Stopping Patience: {patience}")
    print(f"Random Seed:             {seed}")
    print("=" * 75)

    # 1. Create DataLoaders (Train & Val ONLY)
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=batch_size, num_workers=num_workers, normalize=True
    )

    # Extract train mean and std for metric de-normalization
    val_ds = val_loader.dataset
    train_mean = val_ds.train_mean
    train_std = val_ds.train_std

    # 2. Instantiate Model
    model = MSRResNet2x(in_channels=4, out_channels=4, num_features=64, num_blocks=6)
    model = model.to(device)

    # 3. Loss, Optimizer, Scheduler, AMP Scaler
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 4. Prepare CSV Logger
    csv_header = [
        "epoch",
        "train_l1_loss",
        "val_l1_loss",
        "val_reflectance_mae",
        "val_psnr_db",
        "val_ssim",
        "lr",
        "epoch_time_sec",
        "peak_vram_mb",
    ]
    with open(csv_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    best_val_loss = float("inf")
    best_val_psnr = 0.0
    best_epoch = 0
    patience_counter = 0

    total_start_time = time.time()

    # 5. Main Training Loop
    for epoch in range(1, max_epochs + 1):
        epoch_start_time = time.time()

        # --- TRAINING PHASE ---
        model.train()
        train_l1_accum = 0.0

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch:02d}/{max_epochs:02d} [Train]"
        )
        for lr_batch, hr_batch in pbar:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            optimizer.zero_grad()

            # AMP Forward Pass
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                pred_hr = model(lr_batch)
                loss = criterion(pred_hr, hr_batch)

            # AMP Backward Pass & Gradient Clipping
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            train_l1_accum += loss.item()
            pbar.set_postfix({"l1_loss": f"{loss.item():.4f}"})

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        avg_train_l1 = train_l1_accum / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_l1_accum = 0.0
        val_mae_accum = 0.0
        val_psnr_accum = 0.0
        val_ssim_accum = 0.0
        out_of_bounds_pct_accum = 0.0

        with torch.no_grad():
            for lr_batch, hr_batch in tqdm(
                val_loader, desc=f"Epoch {epoch:02d}/{max_epochs:02d} [Val]  ", leave=False
            ):
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)

                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    pred_hr = model(lr_batch)
                    val_loss = criterion(pred_hr, hr_batch)

                val_l1_accum += val_loss.item()

                # Evaluate physical reflectance metrics
                metrics = evaluate_batch_metrics(
                    pred_hr, hr_batch, train_mean, train_std
                )
                val_mae_accum += metrics["reflectance_mae"]
                val_psnr_accum += metrics["psnr_db"]
                val_ssim_accum += metrics["ssim"]
                out_of_bounds_pct_accum += metrics["out_of_bounds_pct"]

        num_val_batches = len(val_loader)
        avg_val_l1 = val_l1_accum / num_val_batches
        avg_val_mae = val_mae_accum / num_val_batches
        avg_val_psnr = val_psnr_accum / num_val_batches
        avg_val_ssim = val_ssim_accum / num_val_batches
        avg_out_bounds_pct = out_of_bounds_pct_accum / num_val_batches

        epoch_time = time.time() - epoch_start_time
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024**2)
            if device.type == "cuda"
            else 0.0
        )

        # Print Epoch Summary
        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} | "
            f"Train L1: {avg_train_l1:.6f} | "
            f"Val L1: {avg_val_l1:.6f} | "
            f"Val MAE: {avg_val_mae:.6f} | "
            f"Val PSNR: {avg_val_psnr:.2f} dB | "
            f"Val SSIM: {avg_val_ssim:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Log to CSV
        with open(csv_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{avg_train_l1:.6f}",
                    f"{avg_val_l1:.6f}",
                    f"{avg_val_mae:.6f}",
                    f"{avg_val_psnr:.4f}",
                    f"{avg_val_ssim:.4f}",
                    f"{current_lr:.6e}",
                    f"{epoch_time:.2f}",
                    f"{peak_vram_mb:.2f}",
                ]
            )

        # Save Checkpoints
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": avg_val_l1,
            "best_val_psnr": avg_val_psnr,
            "train_mean": train_mean.squeeze().tolist(),
            "train_std": train_std.squeeze().tolist(),
            "scale_factor": 10000,
            "model_config": {
                "in_channels": 4,
                "out_channels": 4,
                "num_features": 64,
                "num_blocks": 6,
                "scale_factor": 2,
            },
            "random_seed": seed,
        }

        # Save Latest Checkpoint
        torch.save(checkpoint_data, checkpoint_dir / "checkpoint_latest.pth")

        # Save Best Checkpoint (Lowest Normalized Validation L1)
        if avg_val_l1 < best_val_loss:
            best_val_loss = avg_val_l1
            best_val_psnr = avg_val_psnr
            best_epoch = epoch
            patience_counter = 0

            checkpoint_data["best_val_loss"] = best_val_loss
            checkpoint_data["best_val_psnr"] = best_val_psnr
            torch.save(checkpoint_data, checkpoint_dir / "checkpoint_best.pth")
            print(f"  --> Saved BEST checkpoint at Epoch {epoch:02d} (Val L1: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  --> Validation loss did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\nEARLY STOPPING TRIGGERED at Epoch {epoch:02d}!")
            break

    total_training_time = time.time() - total_start_time

    # Save Summary JSON
    summary_data = {
        "best_epoch": best_epoch,
        "best_val_l1_loss": round(best_val_loss, 6),
        "best_val_psnr_db": round(best_val_psnr, 4),
        "target_out_of_bounds_pixel_pct": round(avg_out_bounds_pct, 4),
        "total_epochs_run": epoch,
        "total_training_time_min": round(total_training_time / 60, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "config": {
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "initial_lr": lr,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "max_grad_norm": max_grad_norm,
            "seed": seed,
        },
    }

    with open(json_summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 75)
    print("TRAINING FINISHED SUCCESSFULLY!")
    print("=" * 75)
    print(f"Best Epoch:                    {best_epoch}")
    print(f"Best Validation L1 Loss:       {best_val_loss:.6f}")
    print(f"Best Validation PSNR:          {best_val_psnr:.2f} dB")
    print(f"Target Out-of-Bounds Pixels:   {avg_out_bounds_pct:.4f}%")
    print(f"Total Training Time:           {total_training_time / 60:.2f} minutes")
    print(f"Peak VRAM Memory:              {peak_vram_mb:.2f} MB")
    print(f"Best Checkpoint:               {checkpoint_dir / 'checkpoint_best.pth'}")
    print(f"Training Log CSV:              {csv_log_path}")
    print("=" * 75)


if __name__ == "__main__":
    train_model(batch_size=8, max_epochs=50, lr=2e-4, patience=12)
