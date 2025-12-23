"""
Compare decoder mel outputs between Python and Swift after fixing noise issue.
Check if we now achieve 1.0 correlation.
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

print("=" * 80)
print("DECODER MEL CORRELATION AFTER FIXING NOISE")
print("=" * 80)

print(f"\nShapes: Python={python_mel.shape}, Swift={swift_mel.shape}")

print(f"\nPython decoder mel:")
print(f"  Mean: {python_mel.mean().item():.8f}")
print(f"  Std: {python_mel.std().item():.8f}")
print(f"  Range: [{python_mel.min().item():.8f}, {python_mel.max().item():.8f}]")

print(f"\nSwift decoder mel:")
print(f"  Mean: {swift_mel.mean().item():.8f}")
print(f"  Std: {swift_mel.std().item():.8f}")
print(f"  Range: [{swift_mel.min().item():.8f}, {swift_mel.max().item():.8f}]")

# Compute correlation
flat_py = python_mel.flatten()
flat_sw = swift_mel.flatten()
correlation = np.corrcoef(flat_py.numpy(), flat_sw.numpy())[0, 1]

print(f"\n" + "=" * 80)
print(f"CORRELATION: {correlation:.10f}")
print("=" * 80)

# Compute element-wise differences
diff = (python_mel - swift_mel).abs()
print(f"\nElement-wise differences:")
print(f"  Mean absolute diff: {diff.mean().item():.8f}")
print(f"  Max absolute diff: {diff.max().item():.8f}")
print(f"  Median absolute diff: {diff.median().item():.8f}")
print(f"  Std of differences: {diff.std().item():.8f}")

# Check first 10 values to validate matching
print(f"\nFirst 10 values [0, :10]:")
py_vals = python_mel[0, :10].tolist()
sw_vals = swift_mel[0, :10].tolist()
print("Python:", [f"{v:.6f}" for v in py_vals])
print("Swift: ", [f"{v:.6f}" for v in sw_vals])

# Result
print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

if correlation > 0.9999:
    print(f"✅ PERFECT MATCH! Correlation = {correlation:.10f}")
    print("   Decoder has achieved mathematical precision (1.0 correlation)!")
elif correlation > 0.999:
    print(f"✅ EXCELLENT! Correlation = {correlation:.10f}")
    print("   Decoder is very close to mathematical precision.")
elif correlation > 0.99:
    print(f"⚠️  GOOD: Correlation = {correlation:.10f}")
    print("   Still some small differences remain.")
else:
    print(f"❌ POOR: Correlation = {correlation:.10f}")
    print("   Significant differences still exist.")

print("\n" + "=" * 80)
