"""
Compare Python and Swift vocoder intermediate outputs layer by layer
to identify where they diverge
"""
import torch
from safetensors.torch import load_file
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
TRACE_DIR = PROJECT_ROOT / "test_audio" / "vocoder_trace"

def compare_tensors(name, python_tensor, swift_tensor, transpose_swift=None):
    """Compare two tensors and report statistics"""
    print(f"\n{'=' * 80}")
    print(f"Comparing: {name}")
    print(f"{'=' * 80}")

    # Handle different shapes
    if transpose_swift:
        print(f"Python shape: {python_tensor.shape}")
        print(f"Swift shape (original): {swift_tensor.shape}")
        swift_tensor = swift_tensor.permute(*transpose_swift)
        print(f"Swift shape (permuted): {swift_tensor.shape}")
    else:
        print(f"Python shape: {python_tensor.shape}")
        print(f"Swift shape: {swift_tensor.shape}")

    # Flatten for comparison
    py_flat = python_tensor.flatten().numpy()
    sw_flat = swift_tensor.flatten().numpy()

    # Basic stats
    print(f"\nPython stats:")
    print(f"  Mean: {py_flat.mean():.8f}")
    print(f"  Std: {py_flat.std():.8f}")
    print(f"  Range: [{py_flat.min():.8f}, {py_flat.max():.8f}]")
    print(f"  First 5: {py_flat[:5]}")

    print(f"\nSwift stats:")
    print(f"  Mean: {sw_flat.mean():.8f}")
    print(f"  Std: {sw_flat.std():.8f}")
    print(f"  Range: [{sw_flat.min():.8f}, {sw_flat.max():.8f}]")
    print(f"  First 5: {sw_flat[:5]}")

    # Correlation
    if len(py_flat) == len(sw_flat):
        corr = np.corrcoef(py_flat, sw_flat)[0, 1]
        print(f"\n{'✅' if corr > 0.99 else '⚠️' if corr > 0.95 else '❌'} Correlation: {corr:.6f}")

        # Differences
        diff = np.abs(py_flat - sw_flat)
        print(f"Absolute differences:")
        print(f"  Mean: {diff.mean():.8f}")
        print(f"  Max: {diff.max():.8f}")
        print(f"  Median: {np.median(diff):.8f}")

        # Check if values match
        if corr > 0.999:
            print("✅ MATCH - Values are nearly identical!")
        elif corr > 0.99:
            print("⚠️ CLOSE - Minor differences, likely numerical precision")
        elif corr > 0.95:
            print("⚠️ SIMILAR - Noticeable differences, but still correlated")
        else:
            print("❌ DIVERGED - Significant implementation differences!")

        return corr
    else:
        print("❌ SHAPE MISMATCH - Cannot compute correlation")
        return None

print("=" * 80)
print("VOCODER LAYER-BY-LAYER COMPARISON")
print("=" * 80)

# Step 0: Input (Python transposes immediately to [B, C, T])
print("\n" + "=" * 80)
print("STEP 0: Input to vocoder")
print("=" * 80)
python_input = load_file(str(TRACE_DIR / "python_0_input.safetensors"))["speech_feat"]
swift_input = load_file(str(TRACE_DIR / "swift_0_input.safetensors"))["mel"]
print(f"Python input: {python_input.shape} = [B, C, T] (PyTorch format)")
print(f"Swift input: {swift_input.shape} = [B, C, T] (before transpose)")
compare_tensors("Input (before transpose)", python_input, swift_input)

# Step 1: F0 Prediction
print("\n" + "=" * 80)
print("STEP 1: F0 Prediction")
print("=" * 80)
python_f0 = load_file(str(TRACE_DIR / "python_1_f0.safetensors"))["f0"]
swift_f0_norm = load_file(str(TRACE_DIR / "swift_1_f0_normalized.safetensors"))["f0"]
swift_f0_hz = load_file(str(TRACE_DIR / "swift_1b_f0_hz.safetensors"))["f0_hz"]

print("Comparing F0 predictions (normalized):")
corr_f0 = compare_tensors("F0 (normalized)", python_f0, swift_f0_norm)

print("\n" + "-" * 80)
print("Swift then scales F0 by 24000.0:")
print(f"  Swift F0 (Hz) range: [{swift_f0_hz.min().item():.6f}, {swift_f0_hz.max().item():.6f}]")
print(f"  Swift F0 (Hz) first 10: {swift_f0_hz[0, :10].tolist()}")

print("\n⚠️ CRITICAL FINDING:")
if corr_f0 and corr_f0 < 0.99:
    print("  ❌ F0 predictions don't match!")
    print("  This means the f0_predictor has different weights or implementation!")
else:
    print("  ✅ F0 predictions match")
    print("  ❓ But Swift multiplies by 24000.0 - does Python?")
    print("  Need to check if this scaling is correct or causing issues")

# Step 2: F0 Upsampling
print("\n" + "=" * 80)
print("STEP 2: F0 Upsampling")
print("=" * 80)
python_f0_up = load_file(str(TRACE_DIR / "python_2_f0_upsampled.safetensors"))["f0_upsampled"]
swift_f0_up = load_file(str(TRACE_DIR / "swift_2_f0_upsampled.safetensors"))["f0_upsampled"]

print("Python uses torch.nn.Upsample (interpolation)")
print("Swift uses tiled() (repeat)")
print("")
corr_f0_up = compare_tensors("F0 Upsampled", python_f0_up, swift_f0_up)

if corr_f0_up and corr_f0_up < 0.99:
    print("\n⚠️ CRITICAL FINDING:")
    print("  ❌ F0 upsampling methods differ!")
    print("  Python: torch.nn.Upsample (linear interpolation by default)")
    print("  Swift: tiled() (simple repeat)")
    print("  This could cause vocoder differences!")

# Step 3: Source Generation
print("\n" + "=" * 80)
print("STEP 3: Source Signal Generation")
print("=" * 80)
python_source = load_file(str(TRACE_DIR / "python_3_source.safetensors"))["source"]
swift_source = load_file(str(TRACE_DIR / "swift_3_source.safetensors"))["source"]

corr_source = compare_tensors("Source Signal", python_source, swift_source)

if corr_source and corr_source < 0.99:
    print("\n⚠️ CRITICAL FINDING:")
    print("  ❌ Source signals differ!")
    print("  This is caused by either:")
    print("    1. Different F0 inputs (from scaling or upsampling)")
    print("    2. Different m_source weights or implementation")

# Step 4: Conv Pre
print("\n" + "=" * 80)
print("STEP 4: Conv Pre")
print("=" * 80)
python_conv_pre = load_file(str(TRACE_DIR / "python_5_conv_pre.safetensors"))["conv_pre"]
swift_conv_pre = load_file(str(TRACE_DIR / "swift_4_conv_pre.safetensors"))["conv_pre"]

print("Python conv_pre output: [B, C, T] = [1, 512, T]")
print("Swift conv_pre output: [B, T, C] = [1, T, 512]")
print("Need to transpose Swift output for comparison")

corr_conv_pre = compare_tensors("Conv Pre", python_conv_pre, swift_conv_pre, transpose_swift=(0, 2, 1))

if corr_conv_pre and corr_conv_pre < 0.99:
    print("\n⚠️ CRITICAL FINDING:")
    print("  ❌ Conv pre outputs differ!")
    print("  This could be from:")
    print("    1. Different weights")
    print("    2. Different convolution implementation")
    print("    3. Transpose issues")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("Layer-by-layer correlation:")
if corr_f0:
    print(f"  F0 prediction: {corr_f0:.6f}")
if corr_f0_up:
    print(f"  F0 upsampling: {corr_f0_up:.6f}")
if corr_source:
    print(f"  Source signal: {corr_source:.6f}")
if corr_conv_pre:
    print(f"  Conv pre: {corr_conv_pre:.6f}")

print("\nFirst divergence point:")
first_diverged = None
for name, corr in [("F0", corr_f0), ("F0_up", corr_f0_up), ("Source", corr_source), ("Conv_pre", corr_conv_pre)]:
    if corr and corr < 0.99:
        first_diverged = name
        print(f"  ⚠️ {name} (correlation: {corr:.6f})")
        break

if not first_diverged:
    print("  ✅ All layers match so far!")
    print("  Divergence must be in later layers (resblocks, ups, etc.)")

print("=" * 80)
