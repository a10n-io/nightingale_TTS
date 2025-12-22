#!/usr/bin/env python3
"""Check Python ODE reference values."""

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

print("=" * 80)
print("PYTHON ODE REFERENCE VALUES")
print("=" * 80)

# Check initial noise
initial_noise = np.load(ref_dir / "step7_initial_noise.npy")
print(f"\nInitial noise:")
print(f"  Shape: {initial_noise.shape}")
print(f"  Range: [{initial_noise.min():.4f}, {initial_noise.max():.4f}]")
print(f"  Mean: {initial_noise.mean():.4f}")

# Check ODE trajectory
for step in [1, 5, 10]:
    print(f"\n--- Step {step} ---")

    x_before = np.load(ref_dir / f"step7_step{step}_x_before.npy")
    print(f"  x_before: [{x_before.min():.4f}, {x_before.max():.4f}], mean={x_before.mean():.4f}")

    dxdt_cfg = np.load(ref_dir / f"step7_step{step}_dxdt_cfg.npy")
    print(f"  velocity: [{dxdt_cfg.min():.4f}, {dxdt_cfg.max():.4f}], mean={dxdt_cfg.mean():.4f}")

    x_after = np.load(ref_dir / f"step7_step{step}_x_after.npy")
    print(f"  x_after:  [{x_after.min():.4f}, {x_after.max():.4f}], mean={x_after.mean():.4f}")

# Check final mel
final_mel = np.load(ref_dir / "step7_final_mel.npy")
print(f"\n--- Final Mel ---")
print(f"  Shape: {final_mel.shape}")
print(f"  Range: [{final_mel.min():.4f}, {final_mel.max():.4f}]")
print(f"  Mean: {final_mel.mean():.4f}")

print("\n" + "=" * 80)
print("KEY OBSERVATIONS:")
print("  1. Python ODE should stay in NEGATIVE range throughout")
print("  2. Velocities should be relatively small (< 1.0)")
print("  3. Final mel should be mostly negative (log-scale)")
print("=" * 80)
