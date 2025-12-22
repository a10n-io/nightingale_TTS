#!/usr/bin/env python3
"""Compare Swift and Python mel spectrograms."""

import numpy as np
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

print("="*80)
print("MEL SPECTROGRAM COMPARISON")
print("="*80)

# Load Python mel
python_mel_path = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en" / "step7_final_mel.npy"
python_mel = np.load(python_mel_path)
print(f"\nPython mel:")
print(f"  Shape: {python_mel.shape}")
print(f"  Range: [{python_mel.min():.4f}, {python_mel.max():.4f}]")
print(f"  Mean: {python_mel.mean():.4f}")
print(f"  Std: {python_mel.std():.4f}")

# Load Swift mel
swift_mel_path = PROJECT_ROOT / "E2E" / "swift_generated_mel_raw.safetensors"
swift_mel_dict = load_file(str(swift_mel_path))
swift_mel = swift_mel_dict['mel'].numpy()
print(f"\nSwift mel:")
print(f"  Shape: {swift_mel.shape}")
print(f"  Range: [{swift_mel.min():.4f}, {swift_mel.max():.4f}]")
print(f"  Mean: {swift_mel.mean():.4f}")
print(f"  Std: {swift_mel.std():.4f}")

# Check if same shape
if python_mel.shape != swift_mel.shape:
    print(f"\n⚠️  WARNING: Different shapes!")
    print(f"   Python: {python_mel.shape}")
    print(f"   Swift:  {swift_mel.shape}")
    min_shape = tuple(min(p, s) for p, s in zip(python_mel.shape, swift_mel.shape))
    python_mel = python_mel[:min_shape[0], :min_shape[1], :min_shape[2]]
    swift_mel = swift_mel[:min_shape[0], :min_shape[1], :min_shape[2]]
    print(f"   Truncating to {min_shape}")

# Compute difference
diff = swift_mel - python_mel
print(f"\nDifference (Swift - Python):")
print(f"  Range: [{diff.min():.4f}, {diff.max():.4f}]")
print(f"  Mean: {diff.mean():.4f}")
print(f"  Max abs diff: {np.abs(diff).max():.4f}")
print(f"  RMSE: {np.sqrt(np.mean(diff**2)):.4f}")

# Compute correlation
correlation = np.corrcoef(python_mel.flatten(), swift_mel.flatten())[0, 1]
print(f"\nCorrelation: {correlation:.6f}")

if correlation > 0.99:
    print("✅ EXCELLENT: Mels are nearly identical")
elif correlation > 0.95:
    print("✅ GOOD: Mels are very similar")
elif correlation > 0.90:
    print("⚠️  OK: Mels are similar but may have differences")
else:
    print("❌ POOR: Mels differ significantly")

print("="*80)
