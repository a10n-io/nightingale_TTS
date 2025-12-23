"""
Compare decoder mel outputs between Python and Swift using correlation.
Check for transpose issues by examining actual values.
"""
import torch
import numpy as np
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"

# Load mels
python_mel = load_file(str(FORENSIC_DIR / "python_decoder_mel.safetensors"))["mel"]
swift_mel = load_file(str(FORENSIC_DIR / "swift_decoder_mel.safetensors"))["mel"]

def compute_correlation(a, b):
    a_flat = a.flatten().numpy()
    b_flat = b.flatten().numpy()
    return np.corrcoef(a_flat, b_flat)[0, 1]

print("=" * 80)
print("DECODER MEL OUTPUT COMPARISON")
print("=" * 80)

print(f"\nPython mel shape: {python_mel.shape}")
print(f"Swift mel shape:  {swift_mel.shape}")

print(f"\nPython: mean={python_mel.mean().item():.6f}, std={python_mel.std().item():.6f}")
print(f"Swift:  mean={swift_mel.mean().item():.6f}, std={swift_mel.std().item():.6f}")
print(f"Python range: [{python_mel.min().item():.6f}, {python_mel.max().item():.6f}]")
print(f"Swift range:  [{swift_mel.min().item():.6f}, {swift_mel.max().item():.6f}]")

corr = compute_correlation(python_mel, swift_mel)
print(f"\n📊 CORRELATION: {corr:.6f}")

if corr > 0.99:
    print("✅ PERFECT! Decoder outputs match.")
elif corr > 0.9:
    print("✅ Good correlation")
else:
    print("❌ Poor correlation - checking for transpose or other issues...")

    # Check first few VALUES to catch transpose
    print("\n" + "=" * 80)
    print("VALUE CHECK (to catch transpose):")
    print("=" * 80)
    print(f"\nPython [0, :5, 0] (first 5 mel bins, first time frame):")
    print(f"  {python_mel[0, :5, 0].tolist()}")
    print(f"Swift  [0, :5, 0]:")
    print(f"  {swift_mel[0, :5, 0].tolist()}")

    print(f"\nPython [0, 0, :5] (first mel bin, first 5 time frames):")
    print(f"  {python_mel[0, 0, :5].tolist()}")
    print(f"Swift  [0, 0, :5]:")
    print(f"  {swift_mel[0, 0, :5].tolist()}")

    # Check if dimensions are swapped [B, C, T] vs [B, T, C]
    if python_mel.shape != swift_mel.shape:
        print(f"\n⚠️  Shape mismatch!")
    else:
        # Try transpose
        python_transposed = python_mel.transpose(1, 2)  # Swap C and T
        corr_transposed = compute_correlation(python_transposed, swift_mel)
        print(f"\n🔍 Correlation with Python mel and freq TRANSPOSED: {corr_transposed:.6f}")
        if corr_transposed > corr:
            print("⚠️  WARNING: Mel dimensions may be swapped!")

print("\n" + "=" * 80)
