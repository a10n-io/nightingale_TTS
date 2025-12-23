"""
Compare Python vocoder vs Swift vocoder when both are given the SAME Python decoder mel.
This proves whether the vocoder implementations match.
"""
import torch
import numpy as np
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"

# Load audio from both vocoders (both fed Python's decoder mel)
python_mel_python_vocoder = load_file(str(FORENSIC_DIR / "python_mel_python_vocoder.safetensors"))["audio"]
python_mel_swift_vocoder = load_file(str(FORENSIC_DIR / "python_mel_swift_vocoder.safetensors"))["audio"]

print("=" * 80)
print("VOCODER CROSS-VALIDATION COMPARISON")
print("=" * 80)
print("\nTest: Both vocoders given IDENTICAL Python decoder mel")
print("Python mel → Python vocoder → Audio A")
print("Python mel → Swift vocoder  → Audio B")
print("\nIf vocoders match: Audio A ≈ Audio B (correlation ≈ 1.0)")
print("=" * 80)

print(f"\nShapes: Python vocoder={python_mel_python_vocoder.shape}, Swift vocoder={python_mel_swift_vocoder.shape}")

print(f"\nPython vocoder output (from Python mel):")
print(f"  Mean: {python_mel_python_vocoder.mean().item():.8f}")
print(f"  Std: {python_mel_python_vocoder.std().item():.8f}")
print(f"  Range: [{python_mel_python_vocoder.min().item():.8f}, {python_mel_python_vocoder.max().item():.8f}]")

print(f"\nSwift vocoder output (from Python mel):")
print(f"  Mean: {python_mel_swift_vocoder.mean().item():.8f}")
print(f"  Std: {python_mel_swift_vocoder.std().item():.8f}")
print(f"  Range: [{python_mel_swift_vocoder.min().item():.8f}, {python_mel_swift_vocoder.max().item():.8f}]")

# Ensure same length
min_len = min(python_mel_python_vocoder.shape[0], python_mel_swift_vocoder.shape[0])
audio_py = python_mel_python_vocoder[:min_len]
audio_sw = python_mel_swift_vocoder[:min_len]

# Compute correlation
correlation = np.corrcoef(audio_py.numpy(), audio_sw.numpy())[0, 1]

print(f"\n" + "=" * 80)
print(f"VOCODER PARITY CORRELATION: {correlation:.10f}")
print("=" * 80)

# Compute differences
diff = (audio_py - audio_sw).abs()
print(f"\nElement-wise differences:")
print(f"  Mean absolute diff: {diff.mean().item():.8f}")
print(f"  Max absolute diff: {diff.max().item():.8f}")
print(f"  Median absolute diff: {diff.median().item():.8f}")
print(f"  Std of differences: {diff.std().item():.8f}")

# Compute SNR
signal_power = (audio_py ** 2).mean()
noise_power = (diff ** 2).mean()
if noise_power > 0:
    snr_db = 10 * np.log10((signal_power / noise_power).item())
    print(f"\nSignal-to-Noise Ratio: {snr_db:.2f} dB")
else:
    print(f"\nSignal-to-Noise Ratio: ∞ dB (perfect match)")

# Check first 10 samples
print(f"\nFirst 10 audio samples:")
py_vals = audio_py[:10].tolist()
sw_vals = audio_sw[:10].tolist()
print("Python vocoder:", [f"{v:.6f}" for v in py_vals])
print("Swift vocoder: ", [f"{v:.6f}" for v in sw_vals])

# Result
print("\n" + "=" * 80)
print("VOCODER IMPLEMENTATION VERDICT")
print("=" * 80)

if correlation > 0.9999:
    print(f"✅ VOCODER IMPLEMENTATIONS MATCH PERFECTLY!")
    print(f"   Correlation = {correlation:.10f}")
    print(f"   The 0.19 end-to-end correlation is NOT from the vocoder.")
    print(f"   It's from the decoder's 0.98 correlation being amplified.")
elif correlation > 0.99:
    print(f"✅ VOCODER IMPLEMENTATIONS MATCH VERY WELL")
    print(f"   Correlation = {correlation:.10f}")
    print(f"   Minor differences, but essentially correct.")
elif correlation > 0.95:
    print(f"⚠️  VOCODER IMPLEMENTATIONS MOSTLY MATCH")
    print(f"   Correlation = {correlation:.10f}")
    print(f"   Some implementation differences exist.")
else:
    print(f"❌ VOCODER IMPLEMENTATIONS DIFFER SIGNIFICANTLY")
    print(f"   Correlation = {correlation:.10f}")
    print(f"   The Swift vocoder has implementation issues.")

print("\n" + "=" * 80)
