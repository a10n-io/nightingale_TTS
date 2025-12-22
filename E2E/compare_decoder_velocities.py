#!/usr/bin/env python3
"""Compare decoder velocities between Python reference and Swift."""

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

print("=" * 80)
print("PYTHON DECODER VELOCITIES (Step 1)")
print("=" * 80)

# Load Python's conditional and unconditional velocities
dxdt_cond = np.load(ref_dir / "step7_step1_dxdt_cond.npy")
dxdt_uncond = np.load(ref_dir / "step7_step1_dxdt_uncond.npy")
dxdt_cfg = np.load(ref_dir / "step7_step1_dxdt_cfg.npy")

print(f"\nConditional velocity (vCond):")
print(f"  Shape: {dxdt_cond.shape}")
print(f"  Range: [{dxdt_cond.min():.4f}, {dxdt_cond.max():.4f}]")
print(f"  Mean: {dxdt_cond.mean():.4f}")

print(f"\nUnconditional velocity (vUncond):")
print(f"  Shape: {dxdt_uncond.shape}")
print(f"  Range: [{dxdt_uncond.min():.4f}, {dxdt_uncond.max():.4f}]")
print(f"  Mean: {dxdt_uncond.mean():.4f}")

print(f"\nCFG velocity (after combining):")
print(f"  Shape: {dxdt_cfg.shape}")
print(f"  Range: [{dxdt_cfg.min():.4f}, {dxdt_cfg.max():.4f}]")
print(f"  Mean: {dxdt_cfg.mean():.4f}")

# Verify CFG formula
cfg_rate = 0.7
computed_cfg = (1.0 + cfg_rate) * dxdt_cond - cfg_rate * dxdt_uncond
print(f"\nVerify CFG formula: (1 + 0.7) * vCond - 0.7 * vUncond")
print(f"  Computed: [{computed_cfg.min():.4f}, {computed_cfg.max():.4f}], mean={computed_cfg.mean():.4f}")
print(f"  Reference: [{dxdt_cfg.min():.4f}, {dxdt_cfg.max():.4f}], mean={dxdt_cfg.mean():.4f}")
print(f"  Match: {np.allclose(computed_cfg, dxdt_cfg, rtol=1e-4)}")

print("\n" + "=" * 80)
print("EXPECTED SWIFT VALUES:")
print("  vCond:   range ≈ [{:.2f}, {:.2f}], mean ≈ {:.2f}".format(dxdt_cond.min(), dxdt_cond.max(), dxdt_cond.mean()))
print("  vUncond: range ≈ [{:.2f}, {:.2f}], mean ≈ {:.2f}".format(dxdt_uncond.min(), dxdt_uncond.max(), dxdt_uncond.mean()))
print("  v (CFG): range ≈ [{:.2f}, {:.2f}], mean ≈ {:.2f}".format(dxdt_cfg.min(), dxdt_cfg.max(), dxdt_cfg.mean()))
print("=" * 80)
