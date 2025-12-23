"""
Compare mel spectrograms from Python and Swift decoders.
"""
import torch
from safetensors.torch import load_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

print("=" * 80)
print("COMPARE DECODER MEL OUTPUTS")
print("=" * 80)

# Load Python mel
python_mel_path = PROJECT_ROOT / "test_audio/cross_validate/python_mel.safetensors"
python_mel_data = load_file(str(python_mel_path))
python_mel = python_mel_data["mel"]

print(f"\n📊 Python Decoder Mel:")
print(f"   Shape: {python_mel.shape}")
print(f"   Range: [{python_mel.min().item():.6f}, {python_mel.max().item():.6f}]")
print(f"   Mean: {python_mel.mean().item():.6f}, Std: {python_mel.std().item():.6f}")

# Check if Swift mel exists
swift_mel_path = PROJECT_ROOT / "test_audio/cross_validate/swift_mel.safetensors"
if swift_mel_path.exists():
    swift_mel_data = load_file(str(swift_mel_path))
    swift_mel = swift_mel_data["mel"]

    print(f"\n📊 Swift Decoder Mel:")
    print(f"   Shape: {swift_mel.shape}")
    print(f"   Range: [{swift_mel.min().item():.6f}, {swift_mel.max().item():.6f}]")
    print(f"   Mean: {swift_mel.mean().item():.6f}, Std: {swift_mel.std().item():.6f}")

    # Compare first few channels
    print(f"\n📊 Channel-by-channel comparison (first 5 channels):")
    for i in range(5):
        py_chan = python_mel[0, i, :]
        sw_chan = swift_mel[0, i, :]
        py_mean = py_chan.mean().item()
        sw_mean = sw_chan.mean().item()
        diff = abs(py_mean - sw_mean)
        print(f"   Channel {i}: Python mean={py_mean:.6f}, Swift mean={sw_mean:.6f}, diff={diff:.6f}")

    # Overall difference
    diff_tensor = python_mel - swift_mel
    print(f"\n📊 Overall difference:")
    print(f"   Mean abs diff: {diff_tensor.abs().mean().item():.6f}")
    print(f"   Max abs diff: {diff_tensor.abs().max().item():.6f}")
    print(f"   RMS diff: {(diff_tensor ** 2).mean().sqrt().item():.6f}")
else:
    print(f"\n⚠️  Swift mel not found at {swift_mel_path}")
    print(f"   Please run SaveSwiftMel to generate it")

print("\n" + "=" * 80)
