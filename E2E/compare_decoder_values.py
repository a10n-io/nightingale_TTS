"""
Compare actual VALUES from decoder outputs to find transpose or other issues.
Check specific positions to catch subtle bugs.
"""
import torch
import numpy as np
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"

# Load mels
python_mel = load_file(str(FORENSIC_DIR / "python_decoder_mel.safetensors"))["mel"]
swift_mel = load_file(str(FORENSIC_DIR / "swift_decoder_mel.safetensors"))["mel"]

print("=" * 80)
print("DECODER MEL VALUE COMPARISON (checking for transpose/bugs)")
print("=" * 80)

print(f"\nShapes: Python={python_mel.shape}, Swift={swift_mel.shape}")

# Check first time frame, first 10 mel bins
print("\n" + "=" * 80)
print("FIRST TIME FRAME, FIRST 10 MEL BINS [0:10, 0]")
print("=" * 80)
py_vals = python_mel[:10, 0].tolist()
sw_vals = swift_mel[:10, 0].tolist()

print("\nPython [:10, 0]:")
for i, v in enumerate(py_vals):
    print(f"  [{i}]: {v:.8f}")

print("\nSwift [:10, 0]:")
for i, v in enumerate(sw_vals):
    print(f"  [{i}]: {v:.8f}")

print("\nDifferences:")
for i in range(10):
    diff = abs(py_vals[i] - sw_vals[i])
    print(f"  [{i}]: {diff:.8f}")

# Check first mel bin, first 10 time frames
print("\n" + "=" * 80)
print("FIRST MEL BIN, FIRST 10 TIME FRAMES [0, 0:10]")
print("=" * 80)
py_vals_t = python_mel[0, :10].tolist()
sw_vals_t = swift_mel[0, :10].tolist()

print("\nPython [0, :10]:")
for i, v in enumerate(py_vals_t):
    print(f"  [{i}]: {v:.8f}")

print("\nSwift [0, :10]:")
for i, v in enumerate(sw_vals_t):
    print(f"  [{i}]: {v:.8f}")

# Check for transpose - does Swift [i,j] match Python [j,i]?
print("\n" + "=" * 80)
print("TRANSPOSE CHECK")
print("=" * 80)

# Sample a few positions
test_positions = [(0, 0), (5, 10), (10, 20), (40, 50), (79, 100)]
transpose_matches = 0
direct_matches = 0

for i, j in test_positions:
    if j >= python_mel.shape[1]:
        continue

    py_direct = python_mel[i, j].item()
    sw_direct = swift_mel[i, j].item()

    # Check if transposed
    if i < swift_mel.shape[1] and j < swift_mel.shape[0]:
        sw_transposed = swift_mel[j, i].item()

        direct_diff = abs(py_direct - sw_direct)
        transpose_diff = abs(py_direct - sw_transposed)

        print(f"\nPosition [{i},{j}]:")
        print(f"  Python[{i},{j}]:       {py_direct:.8f}")
        print(f"  Swift[{i},{j}]:        {sw_direct:.8f} (diff: {direct_diff:.8f})")
        print(f"  Swift[{j},{i}] (T):    {sw_transposed:.8f} (diff: {transpose_diff:.8f})")

        if direct_diff < 0.01:
            direct_matches += 1
        if transpose_diff < 0.01:
            transpose_matches += 1

print(f"\nDirect matches: {direct_matches}/{len(test_positions)}")
print(f"Transpose matches: {transpose_matches}/{len(test_positions)}")

if transpose_matches > direct_matches:
    print("\n⚠️  WARNING: Swift appears to be TRANSPOSED!")
else:
    print("\n✅ No transpose issue detected")

# Compute element-wise statistics
diff = (python_mel - swift_mel).abs()
print("\n" + "=" * 80)
print("ELEMENT-WISE DIFFERENCE STATISTICS")
print("=" * 80)
print(f"Mean absolute difference: {diff.mean().item():.8f}")
print(f"Max absolute difference:  {diff.max().item():.8f}")
print(f"Median absolute difference: {diff.median().item():.8f}")
print(f"Std of differences: {diff.std().item():.8f}")

# Find positions with largest differences
flat_diff = diff.flatten()
top_indices = torch.topk(flat_diff, 10).indices
print("\n10 positions with largest differences:")
for idx in top_indices:
    idx_val = idx.item()
    i = idx_val // python_mel.shape[1]
    j = idx_val % python_mel.shape[1]
    py_val = python_mel[i, j].item()
    sw_val = swift_mel[i, j].item()
    diff_val = diff[i, j].item()
    print(f"  [{i:2d},{j:3d}]: Python={py_val:8.5f}, Swift={sw_val:8.5f}, diff={diff_val:.5f}")

print("\n" + "=" * 80)
