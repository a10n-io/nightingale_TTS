#!/usr/bin/env python3
"""Check Python encoder output reference."""

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

print("=" * 80)
print("PYTHON ENCODER OUTPUT (MU)")
print("=" * 80)

# Check encoder output (mu)
mu = np.load(ref_dir / "step6_mu.npy")
print(f"Shape: {mu.shape}")
print(f"Range: [{mu.min():.4f}, {mu.max():.4f}]")
print(f"Mean: {mu.mean():.4f}")
print(f"Std: {mu.std():.4f}")

# Check x_cond (the conditioning passed to decoder)
x_cond = np.load(ref_dir / "step6_x_cond.npy")
print(f"\nPython x_cond (decoder conditioning):")
print(f"Shape: {x_cond.shape}")
print(f"Range: [{x_cond.min():.4f}, {x_cond.max():.4f}]")
print(f"Mean: {x_cond.mean():.4f}")

print("\n" + "=" * 80)
