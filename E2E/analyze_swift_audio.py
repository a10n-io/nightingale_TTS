#!/usr/bin/env python3
"""Analyze Swift audio output to check if it's proper speech."""

import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

print("="*80)
print("SWIFT AUDIO ANALYSIS")
print("="*80)

# Load Swift audio
swift_audio_path = PROJECT_ROOT / "test_audio" / "swift_output.wav"
sr, swift_audio = wavfile.read(str(swift_audio_path))
swift_audio = swift_audio.astype(np.float32)

print(f"\nSwift Audio:")
print(f"  Sample rate: {sr} Hz")
print(f"  Duration: {len(swift_audio)/sr:.2f}s")
print(f"  Range: [{swift_audio.min():.4f}, {swift_audio.max():.4f}]")
print(f"  Mean: {swift_audio.mean():.4f}")
print(f"  Std: {swift_audio.std():.4f}")

# Check for clicking/popping (large sample-to-sample jumps)
diffs = np.abs(np.diff(swift_audio))
max_diff = diffs.max()
mean_diff = diffs.mean()
print(f"\nSample-to-sample analysis:")
print(f"  Max jump: {max_diff:.4f}")
print(f"  Mean jump: {mean_diff:.4f}")

if max_diff > 0.5:
    print(f"  ⚠️  Large jumps detected - may cause clicking/popping")
    # Find locations of large jumps
    large_jumps = np.where(diffs > 0.3)[0]
    print(f"  Found {len(large_jumps)} jumps > 0.3")
    if len(large_jumps) > 0:
        print(f"  First 5 locations: {large_jumps[:5]}")
else:
    print(f"  ✓ No large jumps - audio should be smooth")

# Check if audio is mostly silent
rms = np.sqrt(np.mean(swift_audio**2))
print(f"\nRMS energy: {rms:.4f}")
if rms < 0.01:
    print("  ⚠️  Audio is very quiet or mostly silent")
elif rms > 0.5:
    print("  ⚠️  Audio is very loud")
else:
    print("  ✓ Energy level looks reasonable")

# Check for DC offset
if abs(swift_audio.mean()) > 0.01:
    print(f"  ⚠️  DC offset detected: {swift_audio.mean():.4f}")
else:
    print("  ✓ No significant DC offset")

# Simple frequency analysis
fft = np.fft.rfft(swift_audio)
freqs = np.fft.rfftfreq(len(swift_audio), 1/sr)
power = np.abs(fft)**2

# Speech is mostly 100-8000 Hz
speech_mask = (freqs >= 100) & (freqs <= 8000)
speech_power = power[speech_mask].sum()
total_power = power.sum()
speech_pct = 100 * speech_power / total_power

print(f"\nFrequency content:")
print(f"  Speech range (100-8kHz): {speech_pct:.1f}% of total power")

# Find dominant frequency
dominant_idx = np.argmax(power[:len(freqs)//10])  # Look in first 10% of spectrum
dominant_freq = freqs[dominant_idx]
print(f"  Dominant frequency: {dominant_freq:.1f} Hz")

if dominant_freq < 100:
    print("    ⚠️  Very low frequency - might be rumble or artifact")
elif dominant_freq > 2000:
    print("    ⚠️  High frequency dominant - unusual for speech")
else:
    print("    ✓ Reasonable for speech")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)

issues = []
if max_diff > 0.5:
    issues.append("Large sample jumps (clicking/popping)")
if rms < 0.01:
    issues.append("Very low energy (possibly silent)")
if speech_pct < 50:
    issues.append(f"Low speech content ({speech_pct:.1f}%)")
if dominant_freq < 100 or dominant_freq > 2000:
    issues.append(f"Unusual dominant frequency ({dominant_freq:.1f} Hz)")

if issues:
    print("\n⚠️  ISSUES DETECTED:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\n✅ Audio looks good! No obvious issues detected.")
    print("   Please listen to the audio to verify quality.")

print("\n" + "="*80)
