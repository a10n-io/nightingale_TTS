"""
Calculate the PRECISE mathematical correction needed to match Python mel.
Uses audio RMS as a proxy for mel energy since we don't have direct mel access.
"""
import torch
import torchaudio
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

print("=" * 80)
print("CALCULATE PRECISE CORRECTION")
print("=" * 80)

# Load audio files - these were generated with RAW mel (no adjustments)
cross_val_dir = PROJECT_ROOT / "test_audio" / "cross_validate"

python_audio_path = cross_val_dir / "python_tokens_python_audio.wav"
swift_audio_path = cross_val_dir / "python_tokens_swift_audio.wav"

python_audio, sr = torchaudio.load(str(python_audio_path))
swift_audio, _  = torchaudio.load(str(swift_audio_path))

print(f"\n📊 Audio Analysis (RAW Swift - no corrections):")
print(f"   Sample rate: {sr} Hz")
print(f"   Python audio: {python_audio.shape}")
print(f"   Swift audio: {swift_audio.shape}")

# Calculate RMS energy
python_rms = python_audio.pow(2).mean().sqrt().item()
swift_rms = swift_audio.pow(2).mean().sqrt().item()

print(f"\n   Python RMS: {python_rms:.8f}")
print(f"   Swift RMS: {swift_rms:.8f}")
print(f"   Ratio (Python/Swift): {python_rms/swift_rms:.6f}")

# Calculate dB difference
rms_diff_dB = 20 * np.log10(python_rms / swift_rms)
print(f"   dB difference: {rms_diff_dB:.6f} dB (Python louder)")

# Calculate spectrogram energy distribution
def analyze_spectrogram(audio):
    # Create mel spectrogram
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        hop_length=240,
        n_mels=80
    )(audio)

    # Convert to dB scale
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    return mel_db

python_mel_db = analyze_spectrogram(python_audio)
swift_mel_db = analyze_spectrogram(swift_audio)

python_mel_mean = python_mel_db.mean().item()
swift_mel_mean = swift_mel_db.mean().item()
python_mel_max = python_mel_db.max().item()
swift_mel_max = swift_mel_db.max().item()

print(f"\n📊 Reconstructed Mel Statistics (from audio):")
print(f"   Python mel: mean={python_mel_mean:.4f} dB, max={python_mel_max:.4f} dB")
print(f"   Swift mel: mean={swift_mel_mean:.4f} dB, max={swift_mel_max:.4f} dB")
print(f"   Mean difference: {python_mel_mean - swift_mel_mean:.4f} dB")
print(f"   Max difference: {python_mel_max - swift_mel_max:.4f} dB")

print(f"\n" + "=" * 80)
print("RECOMMENDED CORRECTION")
print("=" * 80)

# The correction needed
mean_correction = python_mel_mean - swift_mel_mean
max_correction = python_mel_max - swift_mel_max

print(f"\n💡 Mathematical Correction Needed:")
print(f"   Add {mean_correction:.4f} dB to overall brightness")
print(f"   This will shift mean from {swift_mel_mean:.2f} to {python_mel_mean:.2f}")

# Since mel is already in log scale, we ADD the dB directly
print(f"\n📝 Implementation in S3Gen.swift:")
print(f"   After: h = finalProj(h)")
print(f"   Add: h = h + {mean_correction:.4f}")
print(f"   Note: Since mel is in dB/log scale, we add (not multiply)")

# Verification
print(f"\n✅ This will:")
print(f"   - Brighten Swift mel by {mean_correction:.2f} dB")
print(f"   - Match Python's mean energy")
print(f"   - Improve speech clarity")

# Check if clamping is still needed
if swift_mel_max + mean_correction > 0:
    print(f"\n⚠️  After brightening, max will be: {swift_mel_max + mean_correction:.2f} dB")
    print(f"   This is POSITIVE - clamping at 0.0 will be needed:")
    print(f"   h = minimum(h, 0.0)")
else:
    print(f"\n✅ After brightening, max will be: {swift_mel_max + mean_correction:.2f} dB")
    print(f"   Still negative - no clamping needed")

print(f"\n" + "=" * 80)
