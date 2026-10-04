"""
Model Architecture & Forward Pass Test Script

Validates:
1. Total parameter count (~0.7 M).
2. Input/Output tensor shapes: (B, 4, 128, 128) -> (B, 4, 256, 256).
3. CPU and GPU (CUDA) execution compatibility.
4. Peak VRAM allocation on RTX GPU.
5. Integration with real DataLoader batches from SentinelSRDataset.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to sys.path for dataset import
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / "src" / "data"))

from msr_resnet import MSRResNet2x
from sentinel_sr_dataset import SentinelSRDataset, create_dataloaders


def test_model():
    print("=" * 75)
    print("TESTING MSR-RESNET 2X (RCAN-LITE) MODEL ARCHITECTURE")
    print("=" * 75)

    # 1. Instantiate Model & Count Parameters
    model = MSRResNet2x(in_channels=4, out_channels=4, num_features=64, num_blocks=6)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model Class:             {model.__class__.__name__}")
    print(f"Total Trainable Params:  {total_params:,} ({total_params / 1e6:.3f} M)")
    print("-" * 75)

    assert 500_000 <= total_params <= 1_000_000, f"Unexpected parameter count: {total_params}"

    # 2. CPU Dummy Forward Pass Test
    dummy_input = torch.randn(4, 4, 128, 128, dtype=torch.float32)
    print(f"Dummy Input Shape (CPU): {dummy_input.shape}")

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output Shape (CPU):      {output.shape}")
    print(f"Output Dtype:            {output.dtype}")
    print(f"Output Min: {output.min():.4f}, Max: {output.max():.4f}, Mean: {output.mean():.4f}")

    assert output.shape == (4, 4, 256, 256), f"Invalid output shape {output.shape}"
    assert not torch.isnan(output).any(), "NaN found in output!"
    assert not torch.isinf(output).any(), "Inf found in output!"

    # 3. GPU (CUDA) Memory Allocation Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTesting Device: {device}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Detected: {gpu_name}")

        model = model.to(device)
        gpu_input = dummy_input.to(device)

        torch.cuda.reset_peak_memory_stats()
        model.train()

        # Simulated Forward + Backward Pass for VRAM measurement
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        target = torch.randn(4, 4, 256, 256, device=device)

        optimizer.zero_grad()
        gpu_output = model(gpu_input)
        loss = nn.functional.l1_loss(gpu_output, target)
        loss.backward()
        optimizer.step()

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  Forward + Backward Loss: {loss.item():.4f}")
        print(f"  Peak VRAM Memory Used:   {peak_vram_mb:.2f} MB")
        print("  PASS: CUDA execution & backward pass completed cleanly!")

    # 4. DataLoader Integration Test
    print("\nTesting Model Forward Pass on Real DataLoader Batch...")
    train_loader, _, _ = create_dataloaders(batch_size=4, num_workers=0)
    lr_batch, hr_batch = next(iter(train_loader))

    lr_batch = lr_batch.to(device)
    hr_batch = hr_batch.to(device)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        pred_hr = model(lr_batch)
        sample_l1_loss = nn.functional.l1_loss(pred_hr, hr_batch).item()

    print(f"  LR Input Batch Shape:  {lr_batch.shape}")
    print(f"  HR Target Batch Shape: {hr_batch.shape}")
    print(f"  Predicted HR Shape:    {pred_hr.shape}")
    print(f"  Baseline L1 Loss:      {sample_l1_loss:.6f}")

    assert pred_hr.shape == hr_batch.shape, f"Mismatch: pred {pred_hr.shape} vs target {hr_batch.shape}"

    print("\n" + "=" * 75)
    print("SUCCESS: MODEL ARCHITECTURE & FORWARD/BACKWARD QA PASSED!")
    print("=" * 75)


if __name__ == "__main__":
    test_model()
