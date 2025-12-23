"""
Compare Python and Swift decoder mel outputs to identify where they diverge.
"""
import torch
from safetensors.torch import load_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("COMPARE SWIFT VS PYTHON DECODER MEL OUTPUTS")
print("=" * 80)

# Load Python mel (generated region only)
python_mel_path = PROJECT_ROOT / "test_audio/cross_validate/python_mel.safetensors"
python_mel_data = load_file(str(python_mel_path))
python_mel = python_mel_data["mel"]

print(f"\n📊 Python Decoder Mel (Generated Region):")
print(f"   Shape: {python_mel.shape}")
print(f"   Range: [{python_mel.min().item():.6f}, {python_mel.max().item():.6f}]")
print(f"   Mean: {python_mel.mean().item():.6f}, Std: {python_mel.std().item():.6f}")

# Swift mel from cross-validation output
swift_mean = -8.188565
swift_min = -10.839035
swift_max = -1.435415

print(f"\n📊 Swift Decoder Mel (Generated Region from logs):")
print(f"   Shape: [1, 80, 196]")
print(f"   Range: [{swift_min:.6f}, {swift_max:.6f}]")
print(f"   Mean: {swift_mean:.6f}")

# Analyze difference
mean_diff = python_mel.mean().item() - swift_mean
max_diff = python_mel.max().item() - swift_max

print(f"\n🔍 Analysis:")
print(f"   Mean difference: {mean_diff:.6f} ({abs(mean_diff/python_mel.mean().item())*100:.1f}%)")
print(f"   Max difference: {max_diff:.6f}")
print(f"\n❌ ISSUE: Swift mel is too dark (more negative)")
print(f"   - Python has bright values near 0.0")
print(f"   - Swift's max is only {swift_max:.2f} (missing {max_diff:.2f} dB of brightness)")
print(f"   - This suppresses speech content, causing 'humming but no words'")

# Check if we can load a Swift mel file for detailed comparison
swift_mel_path = PROJECT_ROOT / "test_audio/cross_validate/swift_mel.safetensors"
if swift_mel_path.exists():
    print(f"\n📊 Loading Swift mel from file for detailed comparison...")
    swift_mel_data = load_file(str(swift_mel_path))
    # The Swift decoder outputs the full mel (prompt + generated), need to extract generated region
    # Prompt is 500 frames, so generated is [0, :, 500:]
    swift_mel_full = swift_mel_data["mel"]
    print(f"   Swift full mel shape: {swift_mel_full.shape}")

    if swift_mel_full.shape[2] > 500:
        swift_mel_gen = swift_mel_full[:, :, 500:]
        print(f"   Swift generated mel shape: {swift_mel_gen.shape}")
        print(f"   Range: [{swift_mel_gen.min().item():.6f}, {swift_mel_gen.max().item():.6f}]")
        print(f"   Mean: {swift_mel_gen.mean().item():.6f}")

        # Detailed channel-by-channel comparison
        print(f"\n📊 Channel-by-channel comparison (generated region):")
        diff_tensor = python_mel - swift_mel_gen
        for i in [0, 20, 40, 60, 79]:
            py_chan = python_mel[0, i, :]
            sw_chan = swift_mel_gen[0, i, :]
            diff_chan = diff_tensor[0, i, :]
            print(f"   Ch{i:2d}: Python mean={py_chan.mean().item():7.4f}, Swift mean={sw_chan.mean().item():7.4f}, diff={diff_chan.mean().item():7.4f}")

        print(f"\n📊 Overall difference:")
        print(f"   Mean abs diff: {diff_tensor.abs().mean().item():.6f}")
        print(f"   Max abs diff: {diff_tensor.abs().max().item():.6f}")
        print(f"   RMS diff: {(diff_tensor ** 2).mean().sqrt().item():.6f}")
else:
    print(f"\n⚠️  Swift mel file not found")
    print(f"   Run SaveSwiftMel to generate it for detailed comparison")

print("\n" + "=" * 80)
