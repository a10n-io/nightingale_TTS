#!/usr/bin/env python3
"""Debug why Python vUncond is all zeros."""

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

dxdt_uncond = np.load(ref_dir / "step7_step1_dxdt_uncond.npy")

print("Python vUncond (step 1):")
print(f"  Shape: {dxdt_uncond.shape}")
print(f"  Min: {dxdt_uncond.min()}")
print(f"  Max: {dxdt_uncond.max()}")
print(f"  Mean: {dxdt_uncond.mean()}")
print(f"  Std: {dxdt_uncond.std()}")
print(f"  Non-zero count: {np.count_nonzero(dxdt_uncond)}")
print(f"  Total elements: {dxdt_uncond.size}")

# Check if it's actually all zeros or just very small
print(f"\nAbsolute max: {np.abs(dxdt_uncond).max()}")
print(f"First 10 values: {dxdt_uncond.flatten()[:10]}")

# Check a few random positions
print(f"\nRandom samples:")
for i in range(5):
    idx = np.random.randint(0, dxdt_uncond.size)
    print(f"  [{idx}] = {dxdt_uncond.flatten()[idx]}")
