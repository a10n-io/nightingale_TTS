"""
Compare Python vs Swift STFT outputs to identify the exact source path bug
"""
import torch
from safetensors.torch import load_file
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
TRACE_DIR = PROJECT_ROOT / "test_audio" / "fusion_trace"

# Load Python and Swift source_downs inputs (which are STFT outputs)
print("=" * 80)
print("STFT OUTPUT COMPARISON")
print("=" * 80)
print("Checking if STFT implementations match...")
print("=" * 80)

# Load pre-fusion tensors
python_pre = load_file(str(TRACE_DIR / "python_fusion_layer0_pre.safetensors"))
swift_pre = load_file(str(TRACE_DIR / "swift_fusion_layer0_pre.safetensors"))

# Get s_down which is the output of source_downs[0](STFT_output)
python_s_down = python_pre["s_down"]  # [1, 256, 1568] Python format [B, C, T]
swift_s_down = swift_pre["s_down"]    # [1, 1568, 256] Swift format [B, T, C]

print(f"\nPython s_down stats (after source_downs[0]):")
print(f"  Shape: {python_s_down.shape}")
print(f"  Mean: {python_s_down.mean().item():.8f}")
print(f"  Std: {python_s_down.std().item():.8f}")
print(f"  First 10: {python_s_down[0, :, 0].tolist()[:10]}")

# Transpose Swift to match Python format
swift_s_down_t = swift_s_down.permute(0, 2, 1)  # [1, 1568, 256] -> [1, 256, 1568]

print(f"\nSwift s_down stats (after source_downs[0]):")
print(f"  Shape (transposed): {swift_s_down_t.shape}")
print(f"  Mean: {swift_s_down_t.mean().item():.8f}")
print(f"  Std: {swift_s_down_t.std().item():.8f}")
print(f"  First 10: {swift_s_down_t[0, :, 0].tolist()[:10]}")

# Correlation
py_flat = python_s_down.flatten().numpy()
sw_flat = swift_s_down_t.flatten().numpy()
corr = np.corrcoef(py_flat, sw_flat)[0, 1]

print(f"\n{'✅' if corr > 0.99 else '❌'} source_downs[0] output correlation: {corr:.6f}")

if corr < 0.99:
    print("\n❌ CONCLUSION: Bug is in EITHER:")
    print("   1. STFT implementation (different output)")
    print("   2. source_downs[0] Conv1d (wrong weights or transpose)")
    print("\n📋 NEXT STEP: Save STFT outputs from Python and Swift BEFORE source_downs[0]")
    print("   to isolate whether bug is in STFT or Conv1d")
else:
    print("\n✅ source_downs[0] matches! Bug must be elsewhere.")

print("=" * 80)
