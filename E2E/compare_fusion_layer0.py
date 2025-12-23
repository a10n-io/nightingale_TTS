"""
Compare Python vs Swift Layer 0 Fusion to identify divergence point

This is THE CRITICAL DIAGNOSTIC to determine:
1. Main Path (ups[0]): Does it match? → Conv Transpose is OK/broken
2. Source Path (source_downs[0]): Does it match? → STFT/source processing is OK/broken
3. Alignment: If both match but fusion doesn't → Misalignment bug
"""
import torch
from safetensors.torch import load_file
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
TRACE_DIR = PROJECT_ROOT / "test_audio" / "fusion_trace"

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
    print(f"  First 10: {py_flat[:10]}")

    print(f"\nSwift stats:")
    print(f"  Mean: {sw_flat.mean():.8f}")
    print(f"  Std: {sw_flat.std():.8f}")
    print(f"  Range: [{sw_flat.min():.8f}, {sw_flat.max():.8f}]")
    print(f"  First 10: {sw_flat[:10]}")

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
print("LAYER 0 FUSION DIAGNOSTIC")
print("=" * 80)
print("This will tell us EXACTLY where the vocoder diverges:")
print("  - Main Path (ups[0])")
print("  - Source Path (source_downs[0])")
print("  - Alignment (fusion)")
print("=" * 80)

# Load Python fusion outputs
python_pre = load_file(str(TRACE_DIR / "python_fusion_layer0_pre.safetensors"))
python_post = load_file(str(TRACE_DIR / "python_fusion_layer0_post.safetensors"))

# Load Swift fusion outputs
swift_pre = load_file(str(TRACE_DIR / "swift_fusion_layer0_pre.safetensors"))
swift_post = load_file(str(TRACE_DIR / "swift_fusion_layer0_post.safetensors"))

# Compare Main Path (x_up)
print("\n" + "=" * 80)
print("TEST 1: MAIN PATH (ups[0])")
print("=" * 80)
print("This is the mel upsampling path (ConvTranspose1d)")
print("Python: [B, C, T], Swift: [B, T, C]")

python_x_up = python_pre["x_up"]
swift_x_up = swift_pre["x_up"]

corr_x_up = compare_tensors("Main Path (x_up)", python_x_up, swift_x_up, transpose_swift=(0, 2, 1))

# Compare Source Path (s_down)
print("\n" + "=" * 80)
print("TEST 2: SOURCE PATH (source_downs[0])")
print("=" * 80)
print("This is the source STFT downsampling path")
print("Python: [B, C, T], Swift: [B, T, C]")

python_s_down = python_pre["s_down"]
swift_s_down = swift_pre["s_down"]

corr_s_down = compare_tensors("Source Path (s_down)", python_s_down, swift_s_down, transpose_swift=(0, 2, 1))

# Compare Fusion Result (x_fused)
print("\n" + "=" * 80)
print("TEST 3: FUSION RESULT (x + s)")
print("=" * 80)

python_x_fused = python_post["x_fused"]
swift_x_fused = swift_post["x_fused"]

corr_x_fused = compare_tensors("Fusion Result (x_fused)", python_x_fused, swift_x_fused, transpose_swift=(0, 2, 1))

# Compare After Resblock
print("\n" + "=" * 80)
print("TEST 4: AFTER SOURCE RESBLOCK")
print("=" * 80)

python_after_res = python_post["x_after_resblock"]
swift_after_res = swift_post["x_after_resblock"]

corr_after_res = compare_tensors("After Resblock", python_after_res, swift_after_res, transpose_swift=(0, 2, 1))

# ========== DIAGNOSIS ==========
print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print("\nCorrelation Summary:")
print(f"  Main Path (x_up):     {corr_x_up:.6f}" if corr_x_up else "  Main Path: MISMATCH")
print(f"  Source Path (s_down): {corr_s_down:.6f}" if corr_s_down else "  Source Path: MISMATCH")
print(f"  Fusion (x+s):         {corr_x_fused:.6f}" if corr_x_fused else "  Fusion: MISMATCH")
print(f"  After Resblock:       {corr_after_res:.6f}" if corr_after_res else "  After Resblock: MISMATCH")

print("\n" + "=" * 80)
print("ROOT CAUSE ANALYSIS")
print("=" * 80)

if corr_x_up and corr_x_up > 0.99:
    print("✅ Main Path is CORRECT (ups[0] matches)")
else:
    print("❌ Main Path is BROKEN (ups[0] doesn't match)")
    print("   Bug is in: ConvTranspose1d implementation or weights")

if corr_s_down and corr_s_down > 0.99:
    print("✅ Source Path is CORRECT (source_downs[0] matches)")
else:
    print("❌ Source Path is BROKEN (source_downs[0] doesn't match)")
    print("   Bug is in: STFT or source_downs Conv1d implementation")

if corr_x_up and corr_s_down and corr_x_up > 0.99 and corr_s_down > 0.99:
    if corr_x_fused and corr_x_fused < 0.99:
        print("❌ ALIGNMENT BUG - Both paths match individually but fusion doesn't!")
        print("   Bug is in: Tensor shapes/alignment in addition (x + s)")
    else:
        print("✅ Fusion is correct!")

print("=" * 80)
