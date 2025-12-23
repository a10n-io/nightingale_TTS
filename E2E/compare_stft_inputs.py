"""
Compare the EXACT STFT inputs to source_downs[0] between Python and Swift
This will definitively show if the bug is in STFT or Conv1d
"""
import torch
from safetensors.torch import load_file
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
STFT_DIR = PROJECT_ROOT / "test_audio" / "stft_dump"

print("=" * 80)
print("STFT INPUT COMPARISON: Python vs Swift")
print("=" * 80)
print("This is the EXACT tensor that enters source_downs[0]")
print("=" * 80)

# Load Python and Swift STFT outputs
python_stft = load_file(str(STFT_DIR / "python_stft_input.safetensors"))
swift_stft = load_file(str(STFT_DIR / "swift_stft_input.safetensors"))

# Get the concatenated tensors (what source_downs[0] actually sees)
py_concat = python_stft["s_stft_concat"]  # [B, 2F, T'] PyTorch format
sw_concat = swift_stft["s_stft_concat"]   # [B, 2F, T'] PyTorch format (saved in this format)

print(f"\n📊 SHAPE COMPARISON:")
print(f"  Python: {py_concat.shape}")
print(f"  Swift:  {sw_concat.shape}")

if py_concat.shape != sw_concat.shape:
    print("\n❌ SHAPE MISMATCH - Cannot compare!")
    exit(1)

# Split into Real and Imag
F = py_concat.shape[1] // 2
py_real = py_concat[:, :F, :]
py_imag = py_concat[:, F:, :]
sw_real = sw_concat[:, :F, :]
sw_imag = sw_concat[:, F:, :]

print(f"\n📊 STATISTICS:")
print(f"\n  Python Real:")
print(f"    Mean: {py_real.mean().item():.8f}")
print(f"    Std:  {py_real.std().item():.8f}")
print(f"    Range: [{py_real.min().item():.6f}, {py_real.max().item():.6f}]")

print(f"\n  Swift Real:")
print(f"    Mean: {sw_real.mean().item():.8f}")
print(f"    Std:  {sw_real.std().item():.8f}")
print(f"    Range: [{sw_real.min().item():.6f}, {sw_real.max().item():.6f}]")

print(f"\n  Python Imag:")
print(f"    Mean: {py_imag.mean().item():.8f}")
print(f"    Std:  {py_imag.std().item():.8f}")
print(f"    Range: [{py_imag.min().item():.6f}, {py_imag.max().item():.6f}]")

print(f"\n  Swift Imag:")
print(f"    Mean: {sw_imag.mean().item():.8f}")
print(f"    Std:  {sw_imag.std().item():.8f}")
print(f"    Range: [{sw_imag.min().item():.6f}, {sw_imag.max().item():.6f}]")

# Correlation analysis
print(f"\n📊 CORRELATION ANALYSIS:")

# Real part correlation
real_py_flat = py_real.flatten().numpy()
real_sw_flat = sw_real.flatten().numpy()
real_corr = np.corrcoef(real_py_flat, real_sw_flat)[0, 1]

# Imag part correlation
imag_py_flat = py_imag.flatten().numpy()
imag_sw_flat = sw_imag.flatten().numpy()
imag_corr = np.corrcoef(imag_py_flat, imag_sw_flat)[0, 1]

# Overall correlation
full_py_flat = py_concat.flatten().numpy()
full_sw_flat = sw_concat.flatten().numpy()
full_corr = np.corrcoef(full_py_flat, full_sw_flat)[0, 1]

print(f"  Real part correlation: {real_corr:.8f} {'✅' if real_corr > 0.99 else '❌'}")
print(f"  Imag part correlation: {imag_corr:.8f} {'✅' if imag_corr > 0.99 else '❌'}")
print(f"  Full STFT correlation: {full_corr:.8f} {'✅' if full_corr > 0.99 else '❌'}")

# Sample comparison - first frequency bin, first 10 time steps
print(f"\n📊 SAMPLE VALUES (Bin 0, first 10 time steps):")
print(f"\n  Python Real: {py_real[0, 0, :10].tolist()}")
print(f"  Swift Real:  {sw_real[0, 0, :10].tolist()}")
print(f"\n  Python Imag: {py_imag[0, 0, :10].tolist()}")
print(f"  Swift Imag:  {sw_imag[0, 0, :10].tolist()}")

# Check for time shift (classic center=True bug)
print(f"\n📊 TIME SHIFT DETECTION:")
best_shift = 0
best_corr = full_corr

for shift in range(-5, 6):
    if shift == 0:
        continue

    if shift > 0:
        # Swift is ahead, Python is behind
        shifted_corr = np.corrcoef(
            full_py_flat[:len(full_py_flat)-shift*F],
            full_sw_flat[shift*F:]
        )[0, 1]
    else:
        # Python is ahead, Swift is behind
        shifted_corr = np.corrcoef(
            full_py_flat[-shift*F:],
            full_sw_flat[:len(full_sw_flat)+shift*F]
        )[0, 1]

    if shifted_corr > best_corr:
        best_corr = shifted_corr
        best_shift = shift

if best_shift != 0:
    print(f"  ⚠️  Best correlation at shift={best_shift}: {best_corr:.8f}")
    print(f"      This suggests a time alignment issue (center=True bug)")
else:
    print(f"  ✅ No time shift detected (best correlation at shift=0)")

# Diagnosis
print("\n" + "=" * 80)
print("🔍 DIAGNOSIS:")
print("=" * 80)

if full_corr > 0.99:
    print("✅ STFT outputs MATCH!")
    print("   → Bug is NOT in the STFT implementation")
    print("   → Bug MUST be in source_downs[0] Conv1d (weights or implementation)")
elif best_shift != 0 and best_corr > 0.99:
    print(f"❌ STFT outputs are TIME-SHIFTED by {best_shift} frames!")
    print("   → This is the classic center=True alignment bug")
    print("   → Swift STFT is not applying center=True padding correctly")
    print(f"   → Fix: Ensure Swift pads n_fft/2 on BOTH sides before windowing")
elif real_corr > 0.99 and imag_corr < 0.9:
    print("❌ Real part matches but Imag part fails!")
    print("   → This suggests a sign error or FFT implementation bug")
elif real_corr < 0.9 and imag_corr > 0.99:
    print("❌ Imag part matches but Real part fails!")
    print("   → This suggests a sign error or FFT implementation bug")
else:
    print(f"❌ STFT outputs DIFFER (correlation: {full_corr:.8f})")
    print("   → Bug is in the STFT implementation")
    print("   → Possible causes:")
    print("     1. Window function mismatch")
    print("     2. FFT implementation difference")
    print("     3. Padding implementation (reflection vs zero)")
    print("     4. Frame extraction logic")

print("=" * 80)
