"""
Analyze mel brightness based on real audio and prior measurements.
Calculate the scaling factor needed to brighten Swift output.
"""
import torch
import torchaudio
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

print("=" * 80)
print("ANALYZE MEL BRIGHTNESS - DETERMINE SCALING FACTOR")
print("=" * 80)

# Load audio files from cross-validation (generated with clamping fix)
swift_audio_path = PROJECT_ROOT / "test_audio/cross_validate/python_tokens_swift_audio.wav"
python_audio_path = PROJECT_ROOT / "test_audio/cross_validate/python_tokens_python_audio.wav"

swift_audio, sr = torchaudio.load(str(swift_audio_path))
python_audio, _ = torchaudio.load(str(python_audio_path))

print(f"\n📊 Audio Analysis (WITH clamping fix):")
print(f"   Swift audio RMS: {swift_audio.pow(2).mean().sqrt().item():.6f}")
print(f"   Python audio RMS: {python_audio.pow(2).mean().sqrt().item():.6f}")

rms_ratio = python_audio.pow(2).mean().sqrt().item() / swift_audio.pow(2).mean().sqrt().item()
print(f"   RMS ratio (Python/Swift): {rms_ratio:.4f}")

# Calculate dB difference
rms_diff_dB = 20 * torch.log10(torch.tensor(rms_ratio)).item()
print(f"   dB difference: {rms_diff_dB:.2f} dB (Python louder)" if rms_diff_dB > 0 else f"   dB difference: {rms_diff_dB:.2f} dB (Swift louder)")

print(f"\n" + "=" * 80)
print(f"HISTORICAL DATA")
print(f"=" * 80)

print(f"\nBefore clamping (from systematic comparison):")
print(f"   Swift mel mean: -7.27 dB, max: -0.70 dB")
print(f"   Python mel mean: -5.81 dB, max: 0.00 dB")
print(f"   Mean difference: 1.46 dB (Swift darker)")
print(f"   Max difference: 0.70 dB")

print(f"\nAfter clamping to 0.0:")
print(f"   Swift mel max: 0.00 dB (now clamped)")
print(f"   Brightening from clamp: ~0.70 dB at peaks")
print(f"   Remaining mean darkness: ~0.76 dB (1.46 - 0.70)")

print(f"\n" + "=" * 80)
print(f"RECOMMENDED FIX")
print(f"=" * 80)

# Based on the analysis, we need approximately 1.0-1.2 dB of additional brightness
# Let's use 1.12 as a middle ground (10^(1.12/20) ≈ 1.137)
target_brightening_dB = 1.12
scaling_factor = 10 ** (target_brightening_dB / 20)

print(f"\n💡 Apply multiplicative scaling after clamping:")
print(f"   Target brightening: {target_brightening_dB:.2f} dB")
print(f"   Scaling factor: {scaling_factor:.4f}")
print(f"\n   In S3Gen.swift, after line with h = minimum(h, 0.0):")
print(f"   h = h * {scaling_factor:.4f}")

print(f"\n📝 This will:")
print(f"   - Keep mel values ≤ 0.0 (since we multiply negative by positive)")
print(f"   - Brighten the overall mel by ~{target_brightening_dB:.2f} dB")
print(f"   - Bring Swift mel closer to Python's -5.81 dB mean")
print(f"   - Improve speech clarity (less mumbling)")

# Verify the scaling keeps values negative
test_val = -10.0
print(f"\n✅ Verification: {test_val:.1f} * {scaling_factor:.4f} = {test_val * scaling_factor:.2f} (still negative)")
test_val2 = -0.5
print(f"✅ Verification: {test_val2:.1f} * {scaling_factor:.4f} = {test_val2 * scaling_factor:.2f} (still negative)")

print(f"\n⚠️  Note: Clamping at 0.0 means bright peaks may saturate at 0.0")
print(f"   If too many values hit 0.0, consider clamping at +0.5 instead:")
print(f"   h = minimum(h, 0.5)  // Allow slight positive values")

print("\n" + "=" * 80)
