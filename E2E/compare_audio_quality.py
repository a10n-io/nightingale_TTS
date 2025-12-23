"""
Compare audio quality between Python and Swift after corrections.
"""
import torch
import torchaudio
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

print("=" * 80)
print("AUDIO QUALITY COMPARISON")
print("=" * 80)

cross_val_dir = PROJECT_ROOT / "test_audio" / "cross_validate"

python_path = cross_val_dir / "python_tokens_python_audio.wav"
swift_path = cross_val_dir / "python_tokens_swift_audio.wav"

python_audio, sr = torchaudio.load(str(python_path))
swift_audio, _ = torchaudio.load(str(swift_path))

# Calculate metrics
python_rms = python_audio.pow(2).mean().sqrt().item()
swift_rms = swift_audio.pow(2).mean().sqrt().item()
rms_ratio = swift_rms / python_rms

print(f"\n📊 Audio Metrics:")
print(f"   Python RMS: {python_rms:.6f}")
print(f"   Swift RMS: {swift_rms:.6f}")
print(f"   Ratio (Swift/Python): {rms_ratio:.4f}")

if rms_ratio > 0.8 and rms_ratio < 1.2:
    print(f"   ✅ Audio levels are well-matched (ratio {rms_ratio:.2f})")
elif rms_ratio < 0.5:
    print(f"   ⚠️  Swift is too quiet (ratio {rms_ratio:.2f})")
elif rms_ratio > 2.0:
    print(f"   ⚠️  Swift is too loud (ratio {rms_ratio:.2f})")
else:
    print(f"   ℹ️  Moderate difference (ratio {rms_ratio:.2f})")

# Peak analysis
python_peak = python_audio.abs().max().item()
swift_peak = swift_audio.abs().max().item()

print(f"\n   Python peak: {python_peak:.6f}")
print(f"   Swift peak: {swift_peak:.6f}")

print(f"\n" + "=" * 80)
print(f"RESULT")
print(f"=" * 80)
print(f"\n🎵 Please listen to both files:")
print(f"   1. {python_path.name} (Python reference - should be perfect)")
print(f"   2. {swift_path.name} (Swift output - check clarity)")
print(f"\n   Listen for:")
print(f"   - Clear vs mumbled speech")
print(f"   - Similar volume/energy")
print(f"   - No static or distortion")
print(f"\n" + "=" * 80)
