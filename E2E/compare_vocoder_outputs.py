"""
Compare vocoder audio outputs between Python and Swift
"""
import torch
import numpy as np
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"

# Load audio from both implementations
python_audio = load_file(str(FORENSIC_DIR / "python_vocoder_audio.safetensors"))["audio"]
swift_audio = load_file(str(FORENSIC_DIR / "swift_vocoder_audio.safetensors"))["audio"]

print("=" * 80)
print("VOCODER AUDIO OUTPUT COMPARISON")
print("=" * 80)

print(f"\nShapes: Python={python_audio.shape}, Swift={swift_audio.shape}")

print(f"\nPython vocoder audio:")
print(f"  Mean: {python_audio.mean().item():.8f}")
print(f"  Std: {python_audio.std().item():.8f}")
print(f"  Range: [{python_audio.min().item():.8f}, {python_audio.max().item():.8f}]")
print(f"  Duration: {python_audio.shape[0] / 24000:.2f} seconds")

print(f"\nSwift vocoder audio:")
print(f"  Mean: {swift_audio.mean().item():.8f}")
print(f"  Std: {swift_audio.std().item():.8f}")
print(f"  Range: [{swift_audio.min().item():.8f}, {swift_audio.max().item():.8f}]")
print(f"  Duration: {swift_audio.shape[0] / 24000:.2f} seconds")

# Ensure same length (trim to shorter if needed)
min_len = min(python_audio.shape[0], swift_audio.shape[0])
python_audio = python_audio[:min_len]
swift_audio = swift_audio[:min_len]

# Compute correlation
correlation = np.corrcoef(python_audio.numpy(), swift_audio.numpy())[0, 1]

print(f"\n" + "=" * 80)
print(f"VOCODER CORRELATION: {correlation:.10f}")
print("=" * 80)

# Compute element-wise differences
diff = (python_audio - swift_audio).abs()
print(f"\nElement-wise differences:")
print(f"  Mean absolute diff: {diff.mean().item():.8f}")
print(f"  Max absolute diff: {diff.max().item():.8f}")
print(f"  Median absolute diff: {diff.median().item():.8f}")
print(f"  Std of differences: {diff.std().item():.8f}")

# Compute SNR (Signal-to-Noise Ratio)
signal_power = (python_audio ** 2).mean()
noise_power = (diff ** 2).mean()
snr_db = 10 * np.log10((signal_power / noise_power).item())
print(f"\nSignal-to-Noise Ratio: {snr_db:.2f} dB")

# Check first 10 samples
print(f"\nFirst 10 audio samples:")
py_vals = python_audio[:10].tolist()
sw_vals = swift_audio[:10].tolist()
print("Python:", [f"{v:.6f}" for v in py_vals])
print("Swift: ", [f"{v:.6f}" for v in sw_vals])

# Result
print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

if correlation > 0.9999:
    print(f"✅ PERFECT MATCH! Correlation = {correlation:.10f}")
    print(f"   SNR = {snr_db:.2f} dB")
    print("   Vocoder has achieved mathematical precision!")
elif correlation > 0.999:
    print(f"✅ EXCELLENT! Correlation = {correlation:.10f}")
    print(f"   SNR = {snr_db:.2f} dB")
    print("   Vocoder is very close to mathematical precision.")
elif correlation > 0.99:
    print(f"⚠️  GOOD: Correlation = {correlation:.10f}")
    print(f"   SNR = {snr_db:.2f} dB")
    print("   Small differences remain.")
elif correlation > 0.95:
    print(f"⚠️  ACCEPTABLE: Correlation = {correlation:.10f}")
    print(f"   SNR = {snr_db:.2f} dB")
    print("   Audio quality should be good but not perfect.")
else:
    print(f"❌ POOR: Correlation = {correlation:.10f}")
    print(f"   SNR = {snr_db:.2f} dB")
    print("   Significant differences exist.")

print("\n" + "=" * 80)
