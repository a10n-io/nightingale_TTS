"""
Test mel clamping fix by checking Swift mel statistics.
"""
import torch
from safetensors.torch import load_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

print("=" * 80)
print("TEST MEL CLAMPING FIX")
print("=" * 80)

# Check if Swift mel files exist from cross-validation
swift_mel_path = PROJECT_ROOT / "test_audio/cross_validate/swift_mel.safetensors"
if swift_mel_path.exists():
    print("\n📊 Swift Mel (Samantha) from cross-validation:")
    swift_data = load_file(str(swift_mel_path))
    swift_mel = swift_data["mel"]

    print(f"   Shape: {swift_mel.shape}")
    print(f"   Range: [{swift_mel.min().item():.6f}, {swift_mel.max().item():.6f}]")
    print(f"   Mean: {swift_mel.mean().item():.6f}")

    # Check for positive values (should be clamped to 0.0 or below)
    positive_count = (swift_mel > 0).sum().item()
    if positive_count > 0:
        print(f"\n   ❌ WARNING: Found {positive_count} positive values!")
        print(f"   Max positive value: {swift_mel.max().item():.6f}")
    else:
        print(f"\n   ✅ No positive values (all ≤ 0.0)")

    # Check if max is exactly 0.0 (indicating clamping is active)
    if swift_mel.max().item() == 0.0:
        print(f"   ✅ Clamping is working (max exactly 0.0)")
    elif swift_mel.max().item() < -0.01:
        print(f"   ℹ️  Mel naturally stays below 0.0 (max={swift_mel.max().item():.6f})")
else:
    print(f"\n⚠️  Swift mel not found at {swift_mel_path}")
    print("   Run cross-validation first to generate it")

# Check sujano mel if available
sujano_mel_path = PROJECT_ROOT / "test_audio/swift_sujano_mel.safetensors"
if sujano_mel_path.exists():
    print(f"\n📊 Swift Mel (Sujano):")
    sujano_data = load_file(str(sujano_mel_path))
    sujano_mel = sujano_data["mel"]

    print(f"   Shape: {sujano_mel.shape}")
    print(f"   Range: [{sujano_mel.min().item():.6f}, {sujano_mel.max().item():.6f}]")
    print(f"   Mean: {sujano_mel.mean().item():.6f}")

    # Check for positive values
    positive_count = (sujano_mel > 0).sum().item()
    if positive_count > 0:
        print(f"\n   ❌ WARNING: Found {positive_count} positive values!")
        print(f"   Max positive value: {sujano_mel.max().item():.6f}")
        print(f"   This causes vocoder saturation → static audio")
    else:
        print(f"\n   ✅ No positive values (all ≤ 0.0)")
        print(f"   Clamping successfully fixed the sujano static issue!")

    # Check if max is exactly 0.0
    if sujano_mel.max().item() == 0.0:
        print(f"   ✅ Clamping is active (max exactly 0.0)")
else:
    print(f"\n⚠️  Sujano mel not found at {sujano_mel_path}")

print("\n" + "=" * 80)
