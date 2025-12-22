#!/usr/bin/env python3
"""Simple test of Swift-generated mel with Python vocoder."""

import numpy as np
import torch
from safetensors import safe_open
import scipy.io.wavfile as wavfile

# Load Swift-generated mel
print("Loading Swift-generated mel...")
with safe_open('/Users/a10n/Projects/nightingale_TTS/E2E/swift_generated_mel_raw.safetensors', framework="numpy") as f:
    swift_mel = f.get_tensor("mel")

print(f"Swift mel shape: {swift_mel.shape}")
print(f"Swift mel range: [{swift_mel.min():.4f}, {swift_mel.max():.4f}]")
print(f"Swift mel mean: {swift_mel.mean():.4f}")
print(f"Swift mel std: {swift_mel.std():.4f}")

# Analyze frequency structure
print("\nSwift mel channel energies:")
for i in [0, 10, 20, 30, 40, 50, 60, 70, 79]:
    energy = swift_mel[0, i, :].mean()
    print(f"  Channel {i:2d}: {energy:.4f}")

# Load Python reference mel for comparison
print("\nLoading Python reference mel...")
import os
ref_dir = '/Users/a10n/Projects/nightingale_TTS/E2E/reference_outputs/samantha/expressive_surprise_en'
mel_file = os.path.join(ref_dir, 'test_mel_BTC.npy')
if os.path.exists(mel_file):
    python_mel = np.load(mel_file)
    print(f"Python mel shape: {python_mel.shape}")
    print(f"Python mel range: [{python_mel.min():.4f}, {python_mel.max():.4f}]")
    print(f"Python mel mean: {python_mel.mean():.4f}")

    # Python mel is [B, T, C], need to transpose for comparison
    python_mel_transposed = python_mel.transpose(0, 2, 1)  # [B, C, T]

    print("\nPython mel channel energies:")
    for i in [0, 10, 20, 30, 40, 50, 60, 70, 79]:
        energy = python_mel_transposed[0, i, :].mean()
        print(f"  Channel {i:2d}: {energy:.4f}")

    # Compare shapes
    print(f"\n📊 Comparison:")
    print(f"  Swift  mel has {swift_mel.shape[2]} time steps")
    print(f"  Python mel has {python_mel_transposed.shape[2]} time steps")
    print(f"  Time length ratio: {swift_mel.shape[2] / python_mel_transposed.shape[2]:.2f}x")

    # Compare value ranges
    print(f"\n  Swift  mel range: [{swift_mel.min():.2f}, {swift_mel.max():.2f}], mean={swift_mel.mean():.2f}")
    print(f"  Python mel range: [{python_mel_transposed.min():.2f}, {python_mel_transposed.max():.2f}], mean={python_mel_transposed.mean():.2f}")

    # Check frequency gradient
    swift_low = swift_mel[0, 0, :].mean()
    swift_high = swift_mel[0, 79, :].mean()
    python_low = python_mel_transposed[0, 0, :].mean()
    python_high = python_mel_transposed[0, 79, :].mean()

    print(f"\n  Swift  gradient: low={swift_low:.2f}, high={swift_high:.2f}, diff={swift_high-swift_low:.2f}")
    print(f"  Python gradient: low={python_low:.2f}, high={python_high:.2f}, diff={python_high-python_low:.2f}")

    if abs(swift_high - swift_low) < 0.5:
        print("  ⚠️  Swift mel has NO frequency gradient (essentially flat)")
    if abs(python_high - python_low) > 3.0:
        print("  ✅ Python mel has proper frequency gradient")

print("\n✅ Analysis complete!")
print("\nConclusion:")
print("  If Swift mel is flat with no frequency gradient, the issue is in:")
print("  1. Flow decoder weights not loading correctly")
print("  2. Flow decoder forward pass implementation")
print("  3. ODE solver producing incorrect results")
