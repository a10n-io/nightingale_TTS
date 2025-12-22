#!/usr/bin/env python3
"""Check reference final mel to understand expected range."""

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

print("=" * 80)
print("REFERENCE FINAL MEL")
print("=" * 80)

# The reference final mel is what gets passed to the vocoder
final_mel = np.load(ref_dir / "step7_final_mel.npy")
print(f"Shape: {final_mel.shape}")
print(f"Range: [{final_mel.min():.4f}, {final_mel.max():.4f}]")
print(f"Mean: {final_mel.mean():.4f}")
print(f"Std: {final_mel.std():.4f}")

print("\nPer-channel stats (first 10 channels):")
for i in range(min(10, final_mel.shape[1])):
    channel = final_mel[0, i, :]
    print(f"  Channel {i:2d}: min={channel.min():7.4f}, max={channel.max():7.4f}, mean={channel.mean():7.4f}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("  Python's final mel (after ODE) has positive values up to 4.7")
print("  This is NORMAL - it's the raw output from the flow decoder")
print("  The vocoder is trained to handle this range")
print("=" * 80)
