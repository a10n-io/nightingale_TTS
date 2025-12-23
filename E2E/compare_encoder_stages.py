"""
Compare encoder outputs between Python and Swift.
"""
import torch
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"

print("=" * 80)
print("COMPARE ENCODER OUTPUTS")
print("=" * 80)

# Load Python encoder output
python_path = FORENSIC_DIR / "python_encoder_output.safetensors"
python_data = load_file(str(python_path))
python_encoder = python_data["encoder_output"]

# Load Swift encoder output
swift_path = FORENSIC_DIR / "swift_encoder_output.safetensors"
swift_data = load_file(str(swift_path))
swift_encoder = swift_data["encoder_output"]

print(f"\n✅ Loaded both encoder outputs")
print(f"   Python: {python_encoder.shape}")
print(f"   Swift: {swift_encoder.shape}")

# Extract generated region (batch 0 for simplicity, frames 500:696)
# Actually both are [1, 696, 80] so let's just use all of it
# But to match ODE comparison, let's use the generated region

prompt_frames = 500

python_encoder_gen = python_encoder[:, prompt_frames:, :]
swift_encoder_gen = swift_encoder[:, prompt_frames:, :]

print(f"\n📊 Extracted generated region (frames 500:696):")
print(f"   Python shape: {python_encoder_gen.shape}")
print(f"   Swift shape: {swift_encoder_gen.shape}")

# Statistics
print(f"\n📊 Python encoder (generated region):")
print(f"   Range: [{python_encoder_gen.min().item():.8f}, {python_encoder_gen.max().item():.8f}]")
print(f"   Mean: {python_encoder_gen.mean().item():.8f}")
print(f"   Std: {python_encoder_gen.std().item():.8f}")

print(f"\n📊 Swift encoder (generated region):")
print(f"   Range: [{swift_encoder_gen.min().item():.8f}, {swift_encoder_gen.max().item():.8f}]")
print(f"   Mean: {swift_encoder_gen.mean().item():.8f}")
print(f"   Std: {swift_encoder_gen.std().item():.8f}")

# Difference
diff = python_encoder_gen - swift_encoder_gen
abs_diff = diff.abs()

print(f"\n📊 Difference (Python - Swift):")
print(f"   Mean difference: {diff.mean().item():.8f}")
print(f"   Abs mean diff: {abs_diff.mean().item():.8f}")
print(f"   Max abs diff: {abs_diff.max().item():.8f}")
print(f"   Std of diff: {diff.std().item():.8f}")

# Check if encoder outputs match
if abs_diff.mean().item() < 1e-4:
    print(f"\n✅ ENCODER OUTPUTS MATCH (diff < 1e-4)")
    print(f"\n   → Encoder is NOT the source of the ODE divergence")
    print(f"   → Problem must be in ODE solver itself or conditioning")
elif abs_diff.mean().item() < 1e-2:
    print(f"\n⚠️  Small encoder difference (1e-4 < diff < 1e-2)")
elif abs_diff.mean().item() < 0.1:
    print(f"\n⚠️  Moderate encoder difference (diff ~ {abs_diff.mean().item():.6f})")
else:
    print(f"\n❌ LARGE ENCODER DIFFERENCE (diff = {abs_diff.mean().item():.6f})")
    print(f"\n   → Encoder producing different outputs!")
    print(f"   → This could explain the ODE divergence")

# Correlation
python_flat = python_encoder_gen.flatten()
swift_flat = swift_encoder_gen.flatten()
correlation = torch.corrcoef(torch.stack([python_flat, swift_flat]))[0, 1].item()

print(f"\n📊 Correlation between Python and Swift encoder outputs:")
print(f"   Correlation: {correlation:.8f}")
if correlation > 0.99:
    print(f"   ✅ Extremely high correlation")
elif correlation > 0.95:
    print(f"   ✅ High correlation")
else:
    print(f"   ⚠️  Lower correlation")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nEncoder comparison:")
print(f"  - Abs mean diff: {abs_diff.mean().item():.8f}")
print(f"  - Correlation: {correlation:.8f}")
print(f"\nFrom previous analysis:")
print(f"  - ODE output abs mean diff: 0.311227")
print(f"  - ODE correlation: 0.863728")
print(f"\nConclusion:")
if abs_diff.mean().item() < 0.01:
    print(f"  ✅ Encoder outputs match well")
    print(f"  → ODE divergence is NOT caused by encoder")
    print(f"  → Must trace ODE velocity field step-by-step")
else:
    print(f"  ⚠️  Encoder outputs differ")
    print(f"  → This could be contributing to ODE divergence")

print("\n" + "=" * 80)
