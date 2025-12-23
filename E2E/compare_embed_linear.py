"""
Compare embedLinear outputs between Python and Swift to check for transpose issues
"""
import torch
import numpy as np
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"

# Load outputs
python_out = load_file(str(FORENSIC_DIR / "python_after_embedlinear.safetensors"))["after_embedlinear"]
swift_out = load_file(str(FORENSIC_DIR / "swift_after_embedlinear.safetensors"))["after_embedlinear"]

print("=" * 80)
print("EMBEDLINEAR OUTPUT COMPARISON")
print("=" * 80)

print(f"\nPython shape: {python_out.shape}")
print(f"Swift shape:  {swift_out.shape}")

print(f"\nPython: mean={python_out.mean().item():.6f}, std={python_out.std().item():.6f}")
print(f"Swift:  mean={swift_out.mean().item():.6f}, std={swift_out.std().item():.6f}")
print(f"Python range: [{python_out.min().item():.6f}, {python_out.max().item():.6f}]")
print(f"Swift range:  [{swift_out.min().item():.6f}, {swift_out.max().item():.6f}]")

# Check correlation
def compute_correlation(a, b):
    a_flat = a.flatten().numpy()
    b_flat = b.flatten().numpy()
    return np.corrcoef(a_flat, b_flat)[0, 1]

corr = compute_correlation(python_out, swift_out)
print(f"\n📊 CORRELATION: {corr:.6f}")

if corr > 0.99:
    print("✅ PERFECT! embedLinear outputs match.")
elif corr > 0.9:
    print("✅ Good correlation - minor differences")
else:
    print("❌ Poor correlation - likely transpose or weight loading issue!")

# Check first few values to diagnose transpose
print("\n" + "=" * 80)
print("FIRST 5 VALUES CHECK (to catch transpose)")
print("=" * 80)
print("\nPosition [0, :5] (first token, first 5 features):")
print(f"  Python: {python_out[0, :5].tolist()}")
print(f"  Swift:  {swift_out[0, :5].tolist()}")

print("\nPosition [:5, 0] (first 5 tokens, first feature):")
print(f"  Python: {python_out[:5, 0].tolist()}")
print(f"  Swift:  {swift_out[:5, 0].tolist()}")

# Check if Swift matches Python transposed (common mistake)
python_transposed = python_out.T
corr_transposed = compute_correlation(python_transposed, swift_out)
print(f"\n🔍 Correlation with Python TRANSPOSED: {corr_transposed:.6f}")
if corr_transposed > corr:
    print("⚠️  WARNING: Swift correlates BETTER with transposed Python!")
    print("   This suggests a transpose bug in Swift!")
