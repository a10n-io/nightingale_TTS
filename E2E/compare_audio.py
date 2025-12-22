#!/usr/bin/env python3
"""Compare Swift and Python generated audio."""

import numpy as np
from scipy.io import wavfile
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

# Load Python reference audio
python_audio_path = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en" / "step8_python_audio.wav"
swift_audio_path = PROJECT_ROOT / "test_audio" / "python_tokens_swift_audio.wav"

print("=" * 80)
print("AUDIO COMPARISON: Swift vs Python")
print("=" * 80)

# Load Python audio
sr_py, audio_py = wavfile.read(str(python_audio_path))
print(f"\nPython audio:")
print(f"  Sample rate: {sr_py} Hz")
print(f"  Shape: {audio_py.shape}")
print(f"  Duration: {len(audio_py) / sr_py:.2f}s")
print(f"  Range: [{audio_py.min()}, {audio_py.max()}]")
print(f"  Mean: {audio_py.mean():.4f}")
print(f"  Std: {audio_py.std():.4f}")

# Load Swift audio
sr_sw, audio_sw = wavfile.read(str(swift_audio_path))
print(f"\nSwift audio:")
print(f"  Sample rate: {sr_sw} Hz")
print(f"  Shape: {audio_sw.shape}")
print(f"  Duration: {len(audio_sw) / sr_sw:.2f}s")
print(f"  Range: [{audio_sw.min()}, {audio_sw.max()}]")
print(f"  Mean: {audio_sw.mean():.4f}")
print(f"  Std: {audio_sw.std():.4f}")

# Check if same length
if len(audio_py) != len(audio_sw):
    print(f"\n⚠️  WARNING: Different lengths! Python={len(audio_py)}, Swift={len(audio_sw)}")
    min_len = min(len(audio_py), len(audio_sw))
    audio_py = audio_py[:min_len]
    audio_sw = audio_sw[:min_len]
    print(f"   Truncating to {min_len} samples for comparison")

# Compute difference
diff = audio_sw - audio_py
print(f"\nDifference (Swift - Python):")
print(f"  Range: [{diff.min()}, {diff.max()}]")
print(f"  Mean: {diff.mean():.4f}")
print(f"  Std: {diff.std():.4f}")
print(f"  Max abs diff: {np.abs(diff).max()}")
print(f"  RMSE: {np.sqrt(np.mean(diff**2)):.4f}")

# Compute correlation
correlation = np.corrcoef(audio_py.flatten(), audio_sw.flatten())[0, 1]
print(f"\nCorrelation: {correlation:.6f}")

# Compute relative error
relative_rmse = np.sqrt(np.mean((diff / (np.abs(audio_py) + 1e-8))**2))
print(f"Relative RMSE: {relative_rmse:.6f}")

if correlation > 0.99:
    print("\n✅ EXCELLENT: Correlation > 0.99 - Audio is nearly identical")
elif correlation > 0.95:
    print("\n✅ GOOD: Correlation > 0.95 - Audio is very similar")
elif correlation > 0.90:
    print("\n⚠️  OK: Correlation > 0.90 - Audio is similar but may have noticeable differences")
else:
    print("\n❌ POOR: Correlation < 0.90 - Audio differs significantly")

print("=" * 80)
