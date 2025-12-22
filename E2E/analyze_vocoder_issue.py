#!/usr/bin/env python3
"""Analyze the Swift vocoder output to diagnose clicking issues."""

import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

print("=" * 80)
print("VOCODER OUTPUT ANALYSIS")
print("=" * 80)

# Load Swift audio
swift_path = PROJECT_ROOT / "test_audio" / "swift_output.wav"
sr, swift_audio = wavfile.read(str(swift_path))

# Convert to float32 in range [-1, 1]
if swift_audio.dtype == np.int16:
    swift_audio = swift_audio.astype(np.float32) / 32768.0
elif swift_audio.dtype == np.int32:
    swift_audio = swift_audio.astype(np.float32) / 2147483648.0

print(f"\nSwift Audio (normalized):")
print(f"  Sample rate: {sr} Hz")
print(f"  Duration: {len(swift_audio)/sr:.2f}s")
print(f"  Range: [{swift_audio.min():.6f}, {swift_audio.max():.6f}]")
print(f"  Mean: {swift_audio.mean():.6f}")
print(f"  Std: {swift_audio.std():.6f}")
print(f"  RMS: {np.sqrt(np.mean(swift_audio**2)):.6f}")

# Check for clipping
clipped_pos = np.sum(swift_audio >= 0.99)
clipped_neg = np.sum(swift_audio <= -0.99)
total_samples = len(swift_audio)
print(f"\nClipping analysis:")
print(f"  Samples >= 0.99: {clipped_pos} ({100*clipped_pos/total_samples:.2f}%)")
print(f"  Samples <= -0.99: {clipped_neg} ({100*clipped_neg/total_samples:.2f}%)")

if clipped_pos + clipped_neg > total_samples * 0.01:
    print(f"  ⚠️  {100*(clipped_pos+clipped_neg)/total_samples:.1f}% of audio is clipped!")
    print("     This indicates the vocoder is producing values outside [-1, 1]")

# Analyze discontinuities
diffs = np.abs(np.diff(swift_audio))
max_diff = diffs.max()
mean_diff = diffs.mean()
large_jumps = np.sum(diffs > 0.5)

print(f"\nDiscontinuity analysis:")
print(f"  Max sample jump: {max_diff:.6f}")
print(f"  Mean sample jump: {mean_diff:.6f}")
print(f"  Jumps > 0.5: {large_jumps} ({100*large_jumps/len(diffs):.2f}%)")

# Find the worst discontinuities
worst_indices = np.argsort(diffs)[-10:][::-1]
print(f"\nTop 10 worst discontinuities:")
for i, idx in enumerate(worst_indices, 1):
    print(f"  {i}. Sample {idx}: {swift_audio[idx]:.6f} → {swift_audio[idx+1]:.6f} (Δ={diffs[idx]:.6f})")

# Spectrogram analysis
from scipy import signal
f, t, Sxx = signal.spectrogram(swift_audio, sr, nperseg=512)
power_by_freq = np.mean(Sxx, axis=1)

# Find dominant frequencies
top_freq_indices = np.argsort(power_by_freq)[-5:][::-1]
print(f"\nTop 5 frequency components:")
for i, idx in enumerate(top_freq_indices, 1):
    print(f"  {i}. {f[idx]:.1f} Hz: power={power_by_freq[idx]:.2e}")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)

if clipped_pos + clipped_neg > total_samples * 0.01:
    print("\n⚠️  PRIMARY ISSUE: Heavy clipping detected!")
    print("   The vocoder is generating values much larger than [-1, 1]")
    print("   Possible causes:")
    print("   1. Missing normalization in vocoder output")
    print("   2. Incorrect weight scaling")
    print("   3. Missing activation functions")
    print("   4. Incorrect tensor shapes causing broadcasting errors")
elif large_jumps > len(diffs) * 0.5:
    print("\n⚠️  PRIMARY ISSUE: Excessive discontinuities!")
    print("   The audio has many large sample-to-sample jumps")
    print("   Possible causes:")
    print("   1. Incorrect upsampling in vocoder")
    print("   2. Wrong phase/timing in synthesis")
    print("   3. Numerical instability")
else:
    print("\n✅ No obvious vocoder pathologies detected")
    print("   The issue may be more subtle (quality rather than artifacts)")

print("\n" + "=" * 80)
