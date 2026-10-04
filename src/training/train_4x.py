"""
4× Sentinel-2 Super-Resolution — Training Script

Scientific Scope
────────────────
Training benchmark:
  Synthetic 40m-equivalent LR4x (4,64,64) → 10m HR (4,256,256).
  PSNR and validation loss are measured against the 10m HR ground truth
  within this synthetic degradation framework.

Deployment interpretation:
  The trained model is applied to REAL 10m Sentinel-2 inputs to produce
  a 2.5m-scale INFERRED output.

No claim:
  This training does NOT validate genuine 2.5m spatial content.
  Synthetic benchmark PSNR/SSIM does not constitute ground-truth validation
  of sub-10m features in the deployed output.

Isolation Guarantee
────────────────────
Reads:
  src/models/msr_resnet_4x.py
  src/data/sentinel_sr_dataset_4x.py
  data/processed/normalization_stats_4x.json  (via dataset)

Writes:
  outputs/checkpoints/checkpoint_best_4x.pth
  outputs/checkpoints/checkpoint_final_4x.pth
  outputs/training_history_4x.csv

Does NOT touch:
  outputs/checkpoints/checkpoint_best.pth    (2× validated model — untouched)
  outputs/checkpoints/checkpoint_latest.pth  (2× model — untouched)
  data/processed/                            (all datasets — untouched)
  src/training/train.py                      (2× training — untouched)
"""

import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ─── Project path registration ────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(_project_root / "src" / "models"))
sys.path.append(str(_project_root / "src" / "data"))

from msr_resnet_4x import MSRResNet4x
from sentinel_sr_dataset_4x import create_4x_dataloaders

# ─── Constants ────────────────────────────────────────────────────────────────
SEED          = 42
MAX_EPOCHS    = 50
BATCH_SIZE    = 16
LR            = 2e-4
WEIGHT_DECAY  = 1e-4
BETAS         = (0.9, 0.999)
ETA_MIN       = 1e-6
PATIENCE      = 12
MAX_GRAD_NORM = 1.0
NUM_WORKERS   = 0

MODEL_CONFIG = dict(
    in_channels  = 4,
    out_channels = 4,
    num_features = 64,
    num_blocks   = 6,
    scale_factor = 4,
)

CHECKPOINT_DIR        = _project_root / "outputs" / "checkpoints"
BEST_CKPT_PATH        = CHECKPOINT_DIR / "checkpoint_best_4x.pth"
FINAL_CKPT_PATH       = CHECKPOINT_DIR / "checkpoint_final_4x.pth"
HISTORY_CSV_PATH      = _project_root / "outputs" / "training_history_4x.csv"

# Safety guard — must never overwrite the validated 2× model
PROTECTED_2X_CKPT    = CHECKPOINT_DIR / "checkpoint_best.pth"


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Sample-weighted validation (handles unequal final batch)
# ─────────────────────────────────────────────────────────────────────────────

def _validate(model, val_loader, criterion, device, amp_enabled):
    """
    Compute sample-weighted mean validation L1 loss.

    Each sample contributes equally regardless of which batch it falls in.
    This is critical because the final batch contains only
    661 % 16 = 5 samples, not 16.
    """
    model.eval()
    weighted_loss_sum = 0.0
    total_samples     = 0

    with torch.no_grad():
        for lr_batch, hr_batch in val_loader:
            actual_bs = lr_batch.shape[0]
            lr_batch  = lr_batch.to(device)
            hr_batch  = hr_batch.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                pred = model(lr_batch)
                loss = criterion(pred, hr_batch)

            # Accumulate weighted by actual batch size
            weighted_loss_sum += loss.item() * actual_bs
            total_samples     += actual_bs

    return weighted_loss_sum / total_samples


# ─────────────────────────────────────────────────────────────────────────────
# Pre-training sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def _sanity_checks(model, train_loader, device, amp_enabled):
    print("\n  Running pre-training sanity checks...")

    lr_batch, hr_batch = next(iter(train_loader))
    actual_bs = lr_batch.shape[0]

    # Expected shapes
    exp_lr = (actual_bs, 4,  64,  64)
    exp_hr = (actual_bs, 4, 256, 256)

    assert tuple(lr_batch.shape) == exp_lr, (
        f"LR batch shape mismatch: expected {exp_lr}, got {tuple(lr_batch.shape)}"
    )
    assert tuple(hr_batch.shape) == exp_hr, (
        f"HR batch shape mismatch: expected {exp_hr}, got {tuple(hr_batch.shape)}"
    )
    print(f"    LR batch shape : {tuple(lr_batch.shape)}  ✓")
    print(f"    HR batch shape : {tuple(hr_batch.shape)}  ✓")

    # NaN/Inf in inputs
    assert not (torch.isnan(lr_batch).any() or torch.isinf(lr_batch).any()), \
        "NaN/Inf in LR input batch"
    assert not (torch.isnan(hr_batch).any() or torch.isinf(hr_batch).any()), \
        "NaN/Inf in HR target batch"
    print("    Input NaN/Inf  : none  ✓")

    # Model forward pass
    lr_batch = lr_batch.to(device)
    hr_batch = hr_batch.to(device)
    model.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            out = model(lr_batch)

    assert tuple(out.shape) == exp_hr, (
        f"Model output shape mismatch: expected {exp_hr}, got {tuple(out.shape)}"
    )
    print(f"    Model output   : {tuple(out.shape)}  ✓")

    # NaN/Inf in model output
    out_f32 = out.float()
    assert not (torch.isnan(out_f32).any() or torch.isinf(out_f32).any()), \
        "NaN/Inf in model output"
    print("    Output NaN/Inf : none  ✓")

    # Loss is finite
    criterion = nn.L1Loss()
    loss_val = criterion(out_f32, hr_batch.float()).item()
    assert np.isfinite(loss_val), f"Initial loss is not finite: {loss_val}"
    print(f"    Initial L1     : {loss_val:.6f}  (random init)  ✓")

    model.train()
    print("  All sanity checks passed.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helper
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(path: Path, epoch: int, model, optimizer, scheduler,
                     scaler, best_val_loss: float, config: dict):
    # Safety guard
    if path.resolve() == PROTECTED_2X_CKPT.resolve():
        raise RuntimeError(
            f"Attempted to overwrite the protected 2× checkpoint at {path}. "
            "Aborting."
        )
    torch.save({
        "epoch":              epoch,
        "model_state_dict":   model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":  scaler.state_dict(),
        "best_val_loss":      best_val_loss,
        "model_config":       config["model_config"],
        "training_config":    config["training_config"],
        "random_seed":        SEED,
        "scientific_scope": (
            "4× synthetic SR: 40m-equivalent LR4x → 10m HR training benchmark. "
            "Deployment: real 10m Sentinel-2 → 2.5m-scale inferred output. "
            "No sub-10m ground-truth validation claimed."
        ),
    }, path)


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train():
    set_seed(SEED)

    # ── Directories ──────────────────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────────
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda")

    if amp_enabled:
        torch.cuda.reset_peak_memory_stats()

    # ── DataLoaders ──────────────────────────────────────────────────────────
    try:
        train_loader, val_loader, _ = create_4x_dataloaders(
            batch_size  = BATCH_SIZE,
            num_workers = NUM_WORKERS,
            normalize   = True,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(
                "\nCUDA OUT OF MEMORY during DataLoader initialisation.\n"
                f"batch_size={BATCH_SIZE} did not fit on this GPU.\n"
                "Do NOT retry with a different batch size automatically.\n"
                "Please reduce BATCH_SIZE manually and restart."
            )
            raise
        raise

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MSRResNet4x(**MODEL_CONFIG).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Loss / Optimizer / Scheduler / Scaler ─────────────────────────────────
    criterion = nn.L1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = LR,
        betas        = BETAS,
        weight_decay = WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=ETA_MIN
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ── Config snapshot (for checkpoints) ────────────────────────────────────
    ckpt_config = {
        "model_config": MODEL_CONFIG,
        "training_config": {
            "max_epochs":    MAX_EPOCHS,
            "batch_size":    BATCH_SIZE,
            "lr":            LR,
            "weight_decay":  WEIGHT_DECAY,
            "betas":         list(BETAS),
            "eta_min":       ETA_MIN,
            "patience":      PATIENCE,
            "max_grad_norm": MAX_GRAD_NORM,
            "seed":          SEED,
            "amp":           amp_enabled,
        },
    }

    # ── Header ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("4× SENTINEL-2 SUPER-RESOLUTION TRAINING")
    print("=" * 60)
    print(f"Device          : {device}")
    if device.type == "cuda":
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"Parameters      : {total_params:,}  ({total_params/1e6:.3f} M)")
    print(f"Train samples   : {n_train}")
    print(f"Val samples     : {n_val}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Max epochs      : {MAX_EPOCHS}")
    print(f"Initial LR      : {LR}")
    print(f"Patience        : {PATIENCE}")
    print(f"AMP             : {amp_enabled}")
    print(f"Seed            : {SEED}")
    print(f"Best checkpoint : {BEST_CKPT_PATH}")
    print(f"Final checkpoint: {FINAL_CKPT_PATH}")
    print(f"History CSV     : {HISTORY_CSV_PATH}")
    print("=" * 60)
    print("Scientific scope:")
    print("  Training benchmark = synthetic 40m-equiv LR4x → 10m HR.")
    print("  Deployment target  = real 10m S2 → 2.5m-scale inferred output.")
    print("  No sub-10m ground-truth validation is claimed.")
    print("=" * 60)

    # ── Pre-training sanity checks ────────────────────────────────────────────
    _sanity_checks(model, train_loader, device, amp_enabled)

    # ── CSV logger setup ──────────────────────────────────────────────────────
    csv_columns = [
        "epoch", "train_l1", "val_l1", "learning_rate", "epoch_time_seconds",
    ]
    with open(HISTORY_CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_columns).writeheader()

    # ── Training state ────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    best_epoch       = 0
    patience_counter = 0
    total_start      = time.time()
    final_epoch      = 0

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(1, MAX_EPOCHS + 1):
        final_epoch = epoch
        epoch_start = time.time()

        # ── Training phase ───────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_samples  = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{MAX_EPOCHS} [Train]",
            leave=False,
        )
        for lr_batch, hr_batch in pbar:
            actual_bs = lr_batch.shape[0]
            lr_batch  = lr_batch.to(device)
            hr_batch  = hr_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            try:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    pred = model(lr_batch)
                    loss = criterion(pred, hr_batch)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(
                        f"\nCUDA OUT OF MEMORY at epoch {epoch}.\n"
                        f"batch_size={BATCH_SIZE} did not fit on this GPU.\n"
                        "Do NOT retry with a different batch size automatically.\n"
                        "Please reduce BATCH_SIZE manually and restart."
                    )
                    raise
                raise

            # Sample-weighted train loss accumulation
            train_loss_sum += loss.item() * actual_bs
            train_samples  += actual_bs
            pbar.set_postfix({"l1": f"{loss.item():.4f}"})

        avg_train_l1 = train_loss_sum / train_samples

        # ── Validation phase (sample-weighted) ───────────────────────────────
        avg_val_l1 = _validate(model, val_loader, criterion, device, amp_enabled)

        # ── Scheduler step ────────────────────────────────────────────────────
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # ── Best checkpoint ───────────────────────────────────────────────────
        is_best = avg_val_l1 < best_val_loss
        if is_best:
            best_val_loss    = avg_val_l1
            best_epoch       = epoch
            patience_counter = 0
            _save_checkpoint(
                BEST_CKPT_PATH, epoch, model, optimizer, scheduler,
                scaler, best_val_loss, ckpt_config,
            )
        else:
            patience_counter += 1

        # ── Console epoch summary ─────────────────────────────────────────────
        best_flag = "YES ← new best" if is_best else "no"
        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS}  |  "
            f"Train L1: {avg_train_l1:.6f}  |  "
            f"Val L1: {avg_val_l1:.6f}  |  "
            f"LR: {current_lr:.2e}  |  "
            f"Time: {epoch_time:.1f}s  |  "
            f"Best: {best_flag}"
        )
        if not is_best:
            print(f"           Early-stop counter: {patience_counter}/{PATIENCE}")

        # ── CSV row ───────────────────────────────────────────────────────────
        with open(HISTORY_CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_columns).writerow({
                "epoch":               epoch,
                "train_l1":            round(avg_train_l1, 6),
                "val_l1":              round(avg_val_l1,   6),
                "learning_rate":       f"{current_lr:.6e}",
                "epoch_time_seconds":  round(epoch_time, 2),
            })

        # ── Early stopping ────────────────────────────────────────────────────
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {PATIENCE} epochs without improvement.")
            break

    # ── Save final checkpoint (last completed epoch) ───────────────────────────
    _save_checkpoint(
        FINAL_CKPT_PATH, final_epoch, model, optimizer, scheduler,
        scaler, best_val_loss, ckpt_config,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time_min = (time.time() - total_start) / 60.0
    peak_vram_mb   = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )
    final_lr = scheduler.get_last_lr()[0]

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best epoch              : {best_epoch}")
    print(f"Best validation L1      : {best_val_loss:.6f}")
    print(f"Total training time     : {total_time_min:.2f} minutes")
    print(f"Peak VRAM               : {peak_vram_mb:.2f} MB")
    print(f"Final learning rate     : {final_lr:.2e}")
    print(f"Best checkpoint         : {BEST_CKPT_PATH}")
    print(f"Final checkpoint        : {FINAL_CKPT_PATH}")
    print(f"History CSV             : {HISTORY_CSV_PATH}")
    print("=" * 60)
    print("Scientific scope:")
    print("  Synthetic 40m-equivalent LR4x → 10m HR training benchmark.")
    print("  This does NOT validate genuine 2.5m ground-truth performance.")
    print("=" * 60)


if __name__ == "__main__":
    train()
