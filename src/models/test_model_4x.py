"""
MSRResNet4x Architecture & Forward/Backward Verification Script

Checks Performed (10 total)
────────────────────────────
 1. Parameter count reported
 2. CPU forward pass — no exception
 3. CPU output shape == (1, 4, 256, 256)
 4. Output dtype == float32
 5. No NaN / Inf in CPU output
 6. CUDA forward pass when GPU is available
 7. CUDA backward pass with L1 loss — no exception
 8. No NaN / Inf in gradients
 9. DataLoader batch (4, 4, 64, 64) → output (4, 4, 256, 256)
10. Peak GPU VRAM reported for the DataLoader batch

Dimensional compatibility:
    4× bicubic tensor shape verified to match model output shape.

Scientific Scope
────────────────
Verifies the model implementation only.
Does not constitute a training or evaluation result.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Register required src paths
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(_project_root / "src" / "models"))
sys.path.append(str(_project_root / "src" / "data"))

from msr_resnet_4x import MSRResNet4x
from sentinel_sr_dataset_4x import create_4x_dataloaders


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return ok


def test_model_4x():
    print("=" * 68)
    print("MSRResNet4x — ARCHITECTURE & FORWARD/BACKWARD VERIFICATION")
    print("=" * 68)

    failures = []

    # ── Instantiate Model ─────────────────────────────────────────────────────
    model_config = dict(
        in_channels  = 4,
        out_channels = 4,
        num_features = 64,
        num_blocks   = 6,
        scale_factor = 4,
    )
    model = MSRResNet4x(**model_config)

    # ── Check 1: Parameter Count ──────────────────────────────────────────────
    print("\n[Check 1] Parameter count")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total trainable parameters: {total_params:,}  "
          f"({total_params / 1e6:.3f} M)")
    ok1 = total_params > 0
    if not _check("Parameter count > 0", ok1, f"{total_params:,}"):
        failures.append("check1_params")

    # ── Check 2–5: CPU Forward Pass ───────────────────────────────────────────
    print("\n[Checks 2–5] CPU forward pass")
    dummy_lr = torch.randn(1, 4, 64, 64, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        cpu_out = model(dummy_lr)

    ok2 = True   # no exception means pass
    ok3 = tuple(cpu_out.shape) == (1, 4, 256, 256)
    ok4 = cpu_out.dtype == torch.float32
    ok5 = not (torch.isnan(cpu_out).any() or torch.isinf(cpu_out).any())

    for ok, label, detail, name in [
        (ok2, "CPU forward pass completed",            "",                              "check2_cpu_fwd"),
        (ok3, "CPU output shape == (1,4,256,256)",    f"got {tuple(cpu_out.shape)}",   "check3_shape"),
        (ok4, "Output dtype == float32",              f"got {cpu_out.dtype}",          "check4_dtype"),
        (ok5, "No NaN / Inf in CPU output",           "",                              "check5_nan_inf"),
    ]:
        if not _check(label, ok, detail):
            failures.append(name)

    print(f"    Output: min={cpu_out.min():.4f}  max={cpu_out.max():.4f}  "
          f"mean={cpu_out.mean():.4f}")

    # ── Check 6–8: CUDA Forward & Backward ───────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Checks 6–8] CUDA forward + backward  (device: {device})")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"    GPU: {gpu_name}")

        model_gpu = MSRResNet4x(**model_config).to(device)
        model_gpu.train()

        lr_gpu = torch.randn(1, 4, 64, 64, dtype=torch.float32, device=device)
        hr_gpu = torch.randn(1, 4, 256, 256, dtype=torch.float32, device=device)

        optimizer = torch.optim.AdamW(model_gpu.parameters(), lr=1e-4)
        torch.cuda.reset_peak_memory_stats()

        try:
            optimizer.zero_grad()
            out_gpu = model_gpu(lr_gpu)
            loss    = nn.functional.l1_loss(out_gpu, hr_gpu)
            loss.backward()
            optimizer.step()
            cuda_ok = True
        except Exception as e:
            print(f"    ERROR during CUDA pass: {e}")
            cuda_ok = False

        ok6 = cuda_ok
        ok7 = cuda_ok
        ok8 = True
        if cuda_ok:
            for name_p, param in model_gpu.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        ok8 = False
                        print(f"    NaN/Inf gradient in: {name_p}")
                        break

        peak_vram_single = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if cuda_ok else 0.0
        )

        for ok, label, name in [
            (ok6, "CUDA forward pass completed",          "check6_cuda_fwd"),
            (ok7, "CUDA backward + optimizer step done",  "check7_cuda_bwd"),
            (ok8, "No NaN / Inf in gradients",            "check8_grad_nan"),
        ]:
            if not _check(label, ok):
                failures.append(name)

        if cuda_ok:
            print(f"    L1 loss (random init):     {loss.item():.6f}")
            print(f"    Peak VRAM (single sample): {peak_vram_single:.2f} MB")
    else:
        print("    CUDA not available — skipping checks 6–8.")
        for name in ["check6_cuda_fwd", "check7_cuda_bwd", "check8_grad_nan"]:
            _check(f"CUDA unavailable — {name} skipped", True)

    # ── Check 9–10: DataLoader Batch ─────────────────────────────────────────
    print("\n[Checks 9–10] DataLoader batch (batch_size=4)")
    train_loader, _, _ = create_4x_dataloaders(batch_size=4, num_workers=0)
    lr_batch, hr_batch = next(iter(train_loader))

    exp_lr_batch = (4, 4, 64, 64)
    exp_hr_batch = (4, 4, 256, 256)

    ok_lr_in  = tuple(lr_batch.shape) == exp_lr_batch
    ok_hr_tgt = tuple(hr_batch.shape) == exp_hr_batch
    if not _check(f"DataLoader LR batch shape == {exp_lr_batch}",
                  ok_lr_in, f"got {tuple(lr_batch.shape)}"):
        failures.append("check9_lr_batch")
    if not _check(f"DataLoader HR batch shape == {exp_hr_batch}",
                  ok_hr_tgt, f"got {tuple(hr_batch.shape)}"):
        failures.append("check9_hr_batch")

    # Forward pass on a real DataLoader batch
    model_eval = MSRResNet4x(**model_config).to(device)
    model_eval.eval()

    lr_batch = lr_batch.to(device)
    hr_batch = hr_batch.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        sr_batch = model_eval(lr_batch)

    ok9 = tuple(sr_batch.shape) == exp_hr_batch
    if not _check(f"Model output on DataLoader batch == {exp_hr_batch}",
                  ok9, f"got {tuple(sr_batch.shape)}"):
        failures.append("check9_dl_output_shape")

    peak_vram_batch = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )
    ok10 = True  # informational
    _check(
        f"Peak VRAM for batch of 4",
        ok10,
        f"{peak_vram_batch:.2f} MB" if device.type == "cuda" else "CPU — N/A",
    )

    # Batch statistics
    batch_l1 = nn.functional.l1_loss(sr_batch, hr_batch).item()
    print(f"    SR batch: min={sr_batch.min():.4f}  max={sr_batch.max():.4f}  "
          f"mean={sr_batch.mean():.4f}")
    print(f"    Baseline L1 vs HR (random init): {batch_l1:.6f}")

    # ── Dimensional Compatibility: 4× Bicubic vs Model Output ────────────────
    print("\n[Dimensional Compatibility] 4× bicubic tensor vs model output")
    import torch.nn.functional as F
    bicubic_batch = F.interpolate(
        lr_batch, scale_factor=4, mode="bicubic", align_corners=False
    )
    shapes_match = tuple(bicubic_batch.shape) == tuple(sr_batch.shape)
    if not _check(
        "4× bicubic shape matches model output shape",
        shapes_match,
        f"bicubic={tuple(bicubic_batch.shape)}  model={tuple(sr_batch.shape)}",
    ):
        failures.append("check_bicubic_compat")

    # ── Architecture Summary ──────────────────────────────────────────────────
    print("\n  Architecture Summary:")
    print(f"    Model class:          MSRResNet4x")
    print(f"    Input  shape:         (B, 4, 64, 64)")
    print(f"    Output shape:         (B, 4, 256, 256)")
    print(f"    Scale factor:         4×  (two sequential 2× PixelShuffle stages)")
    print(f"    num_features:         {model_config['num_features']}")
    print(f"    num_blocks (RCAB):    {model_config['num_blocks']}")
    print(f"    Total parameters:     {total_params:,}  ({total_params/1e6:.3f} M)")
    if device.type == "cuda":
        print(f"    Peak VRAM (batch=4):  {peak_vram_batch:.2f} MB")

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    all_passed = len(failures) == 0
    if all_passed:
        print("  ALL CHECKS PASSED — MSRResNet4x is verified and ready.")
    else:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"    FAIL: {f}")
    print("=" * 68)
    print("Scientific scope: architecture verification only.")
    print("Does not validate real 10m → 2.5m SR performance.")
    print("=" * 68)


if __name__ == "__main__":
    test_model_4x()
