"""
Direct comparison of Python and Swift raw mel outputs.
Uses safetensors saved from cross-validation runs.
"""
import torch
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

print("=" * 80)
print("FORENSIC MEL COMPARISON")
print("="  * 80)

# Check for existing mel outputs from previous runs
forensic_dir = PROJECT_ROOT / "test_audio" / "forensic"

# For now, let's use outputs from the cross-validation if available
cross_val_dir = PROJECT_ROOT / "test_audio" / "cross_validate"

print("\n🔍 Searching for existing mel outputs...")

# Try to find Swift mel from any previous run
swift_mel_files = [
    forensic_dir / "swift_mel_raw.safetensors",
    PROJECT_ROOT / "test_audio" / "swift_sujano_mel.safetensors",
]

swift_mel_path = None
for path in swift_mel_files:
    if path.exists():
        swift_mel_path = path
        print(f"   Found Swift mel: {path.name}")
        break

if swift_mel_path:
    swift_data = load_file(str(swift_mel_path))
    # Get the mel (different files have different key names)
    if "mel" in swift_data:
        swift_mel = swift_data["mel"]
    elif "mel_gen" in swift_data:
        swift_mel = swift_data["mel_gen"]
    elif "mel_full" in swift_data:
        swift_mel = swift_data["mel_full"]
    else:
        swift_mel = list(swift_data.values())[0]  # Get first tensor

    print(f"\n📊 Swift Mel (from {swift_mel_path.name}):")
    print(f"   Shape: {swift_mel.shape}")
    print(f"   Range: [{swift_mel.min().item():.8f}, {swift_mel.max().item():.8f}]")
    print(f"   Mean: {swift_mel.mean().item():.8f}")
    print(f"   Std: {swift_mel.std().item():.8f}")
else:
    print("\n⚠️  No Swift mel found. Please run:")
    print("   swift run --package-path swift SaveSwiftMel")
    print("   Or: swift run GenerateAudio")

print("\n" + "=" * 80)
print("INSTRUCTION: Generate fresh mels for precise comparison")
print("=" * 80)
print("\n1. Run Python mel generation:")
print("   python E2E/forensic_trace_python.py")
print("\n2. Run Swift mel generation:")
print("   cd swift && swift run CrossValidate")
print("\n3. The analysis will show:")
print("   - Exact numerical differences")
print("   - Where divergence occurs")
print("   - Mathematical correction needed")
print("\n" + "=" * 80)
