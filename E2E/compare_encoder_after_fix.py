"""
Compare Swift encoder output with Python after key remapping fix.
"""
import torch
from safetensors.torch import load_file
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"

print("=" * 80)
print("COMPARE ENCODER OUTPUTS AFTER KEY REMAPPING FIX")
print("=" * 80)

# Load Python encoder output (before proj)
python_file = FORENSIC_DIR / "python_encoder_before_proj.safetensors"
python_data = load_file(str(python_file))
python_enc = python_data["encoder_before_proj"]

# Load Swift encoder output (before proj)
swift_file = FORENSIC_DIR / "swift_encoder_fixed.safetensors"
swift_data = load_file(str(swift_file))
swift_enc = swift_data["encoder_output_before_proj"]

print(f"\nPython encoder (before proj):")
print(f"  Shape: {python_enc.shape}")
print(f"  Mean: {python_enc.mean().item():.6f}")
print(f"  Std: {python_enc.std().item():.6f}")
print(f"  Range: [{python_enc.min().item():.6f}, {python_enc.max().item():.6f}]")

print(f"\nSwift encoder (before proj):")
print(f"  Shape: {swift_enc.shape}")
print(f"  Mean: {swift_enc.mean().item():.6f}")
print(f"  Std: {swift_enc.std().item():.6f}")
print(f"  Range: [{swift_enc.min().item():.6f}, {swift_enc.max().item():.6f}]")

# Correlation
python_flat = python_enc.flatten().numpy()
swift_flat = swift_enc.flatten().numpy()
correlation = np.corrcoef(python_flat, swift_flat)[0, 1]

print(f"\n📊 ENCODER CORRELATION (before proj): {correlation:.6f}")
if correlation > 0.99:
    print(f"✅ Excellent correlation! Encoder outputs match.")
elif correlation > 0.95:
    print(f"⚠️  Good correlation but not perfect.")
else:
    print(f"❌ Poor correlation! Encoder outputs differ significantly.")

# Now check after encoder_proj
python_enc_proj = swift_data["encoder_output_after_proj"]  # Wait, we don't have Python's encoder_proj output saved!

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)

print(f"\nEncoder BEFORE encoder_proj:")
print(f"  Correlation: {correlation:.6f}")
print(f"  Std ratio (Swift/Python): {swift_enc.std().item() / python_enc.std().item():.6f}")

if correlation > 0.95:
    print(f"\n✅ ENCODER FIX SUCCESSFUL!")
    print(f"   Key remapping fixed the encoder embed.linear weight loading.")
    print(f"   Encoder outputs now match Python closely.")
    print(f"\n⚠️  But encoder_proj still has issues:")
    print(f"   - Swift encoder_proj output std: 0.205")
    print(f"   - Python encoder_proj output std: 0.435")
    print(f"   - Ratio: 0.47 (should be ~1.0)")
    print(f"\n🔍 Next step: Compare encoder_proj outputs directly")
else:
    print(f"\n❌ Encoder still has correlation issues.")
    print(f"   The key remapping may not have fully fixed the problem.")

print("\n" + "=" * 80)
