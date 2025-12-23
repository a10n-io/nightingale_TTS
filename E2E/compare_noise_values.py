"""
Compare the actual noise values from Python and Swift to validate the assumption
that they are different.
"""
import torch
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"

# Load noise from both implementations
python_noise = load_file(str(FORENSIC_DIR / "python_decoder_noise.safetensors"))["noise"]
swift_noise = load_file(str(FORENSIC_DIR / "swift_decoder_noise.safetensors"))["noise"]

print("=" * 80)
print("NOISE VALUE COMPARISON")
print("=" * 80)

print(f"\nPython noise shape: {python_noise.shape}")
print(f"Swift noise shape: {swift_noise.shape}")

# Check first 10 values
print("\n" + "=" * 80)
print("FIRST 10 VALUES [0, 0, :10]")
print("=" * 80)

py_vals = python_noise[0, 0, :10].tolist()
sw_vals = swift_noise[0, 0, :10].tolist()

print("\nPython [0, 0, :10]:")
for i, v in enumerate(py_vals):
    print(f"  [{i}]: {v:.8f}")

print("\nSwift [0, 0, :10]:")
for i, v in enumerate(sw_vals):
    print(f"  [{i}]: {v:.8f}")

print("\nAbsolute differences:")
for i in range(10):
    diff = abs(py_vals[i] - sw_vals[i])
    print(f"  [{i}]: {diff:.8f}")

# Compute overall statistics
print("\n" + "=" * 80)
print("OVERALL STATISTICS")
print("=" * 80)

print(f"\nPython noise:")
print(f"  Mean: {python_noise.mean().item():.8f}")
print(f"  Std: {python_noise.std().item():.8f}")
print(f"  Range: [{python_noise.min().item():.8f}, {python_noise.max().item():.8f}]")

print(f"\nSwift noise:")
print(f"  Mean: {swift_noise.mean().item():.8f}")
print(f"  Std: {swift_noise.std().item():.8f}")
print(f"  Range: [{swift_noise.min().item():.8f}, {swift_noise.max().item():.8f}]")

# Check correlation
flat_py = python_noise.flatten()
flat_sw = swift_noise.flatten()
correlation = torch.corrcoef(torch.stack([flat_py, flat_sw]))[0, 1].item()

print(f"\nNoise correlation: {correlation:.8f}")

# Check element-wise difference
diff = (python_noise - swift_noise).abs()
print(f"\nElement-wise difference:")
print(f"  Mean absolute diff: {diff.mean().item():.8f}")
print(f"  Max absolute diff: {diff.max().item():.8f}")
print(f"  Median absolute diff: {diff.median().item():.8f}")

# Conclusion
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if correlation > 0.99:
    print("✅ Noise values are IDENTICAL (correlation > 0.99)")
    print("   The noise is NOT the source of decoder differences.")
elif correlation > 0.5:
    print("⚠️  Noise values are SIMILAR but not identical")
    print(f"   Correlation: {correlation:.8f}")
elif abs(correlation) < 0.1:
    print("❌ Noise values are COMPLETELY DIFFERENT (correlation ≈ 0)")
    print("   This confirms that PyTorch and MLX random generators produce different noise.")
    print("   This IS the source of the 2% decoder difference!")
else:
    print(f"? Noise correlation: {correlation:.8f}")
    print("  Unexpected correlation value.")

print("\n" + "=" * 80)
