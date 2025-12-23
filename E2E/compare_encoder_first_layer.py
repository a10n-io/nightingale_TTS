"""
Compare first layer probe outputs between Python and Swift.
This will tell us exactly where the encoder divergence starts.
"""
import torch
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"

print("=" * 80)
print("FIRST LAYER PROBE - COMPARISON")
print("=" * 80)

# Load Python checkpoints
python_path = FORENSIC_DIR / "python_encoder_first_layer_probe.safetensors"
python_data = load_file(str(python_path))

# Load Swift checkpoints
swift_path = FORENSIC_DIR / "swift_encoder_first_layer_probe.safetensors"
swift_data = load_file(str(swift_path))

print(f"\n✅ Loaded both probe outputs")

# ==============================================================================
# CHECKPOINT 1: speech_emb_matrix
# ==============================================================================
print(f"\n" + "=" * 80)
print("CHECKPOINT 1: speech_emb_matrix (raw weights)")
print("=" * 80)

py_emb = python_data["checkpoint1_speech_emb_matrix"]
sw_emb = swift_data["checkpoint1_speech_emb_matrix"]

print(f"\n📊 Python:")
print(f"   Shape: {py_emb.shape}")
print(f"   Range: [{py_emb.min().item():.8f}, {py_emb.max().item():.8f}]")
print(f"   Mean: {py_emb.mean().item():.8f}")
print(f"   Std: {py_emb.std().item():.8f}")
print(f"   First 10 [0,0,:10]: {py_emb[0, 0, :10].tolist()}")

print(f"\n📊 Swift:")
print(f"   Shape: {sw_emb.shape}")
print(f"   Range: [{sw_emb.min().item():.8f}, {sw_emb.max().item():.8f}]")
print(f"   Mean: {sw_emb.mean().item():.8f}")
print(f"   Std: {sw_emb.std().item():.8f}")
print(f"   First 10 [0,0,:10]: {sw_emb[0, 0, :10].tolist()}")

# Compare
diff1 = (py_emb - sw_emb).abs()
print(f"\n📊 Difference:")
print(f"   Mean abs diff: {diff1.mean().item():.10f}")
print(f"   Max abs diff: {diff1.max().item():.10f}")

if diff1.mean().item() < 1e-6:
    print(f"   ✅ CHECKPOINT 1 MATCHES (diff < 1e-6)")
    print(f"   → speech_emb_matrix loaded correctly")
else:
    print(f"   ❌ CHECKPOINT 1 FAILS")
    print(f"   → speech_emb_matrix differs! Check weight loading.")

# Check for transposition by comparing a few elements
print(f"\n🔍 Transposition check:")
print(f"   Python [0,0,0]: {py_emb[0,0,0].item():.6f}")
print(f"   Python [0,1,0]: {py_emb[0,1,0].item():.6f}")
print(f"   Python [0,0,1]: {py_emb[0,0,1].item():.6f}")
print(f"   Swift  [0,0,0]: {sw_emb[0,0,0].item():.6f}")
print(f"   Swift  [0,1,0]: {sw_emb[0,1,0].item():.6f}")
print(f"   Swift  [0,0,1]: {sw_emb[0,0,1].item():.6f}")

# ==============================================================================
# CHECKPOINT 2: After embedding lookup
# ==============================================================================
print(f"\n" + "=" * 80)
print("CHECKPOINT 2: After embedding lookup")
print("=" * 80)

py_after_emb = python_data["checkpoint2_after_embed"]
sw_after_emb = swift_data["checkpoint2_after_embed"]

# Check if they were captured
if py_after_emb.numel() == 1 or sw_after_emb.numel() == 1:
    print(f"\n⚠️  One or both checkpoints not captured properly")
    if py_after_emb.numel() == 1:
        print(f"   Python checkpoint2 not captured")
    if sw_after_emb.numel() == 1:
        print(f"   Swift checkpoint2 not captured")
else:
    print(f"\n📊 Python:")
    print(f"   Shape: {py_after_emb.shape}")
    print(f"   Range: [{py_after_emb.min().item():.8f}, {py_after_emb.max().item():.8f}]")
    print(f"   Mean: {py_after_emb.mean().item():.8f}")
    print(f"   Std: {py_after_emb.std().item():.8f}")
    if py_after_emb.dim() >= 2:
        print(f"   First 10 [0,:10]: {py_after_emb[0, 0, :10].tolist()}")

    print(f"\n📊 Swift:")
    print(f"   Shape: {sw_after_emb.shape}")
    print(f"   Range: [{sw_after_emb.min().item():.8f}, {sw_after_emb.max().item():.8f}]")
    print(f"   Mean: {sw_after_emb.mean().item():.8f}")
    print(f"   Std: {sw_after_emb.std().item():.8f}")
    if sw_after_emb.dim() >= 2:
        print(f"   First 10 [0,:10]: {sw_after_emb[0, 0, :10].tolist()}")

    # Compare
    # Make sure shapes match
    if py_after_emb.shape == sw_after_emb.shape:
        diff2 = (py_after_emb - sw_after_emb).abs()
        print(f"\n📊 Difference:")
        print(f"   Mean abs diff: {diff2.mean().item():.10f}")
        print(f"   Max abs diff: {diff2.max().item():.10f}")

        # Correlation
        py_flat = py_after_emb.flatten()
        sw_flat = sw_after_emb.flatten()
        correlation = torch.corrcoef(torch.stack([py_flat, sw_flat]))[0, 1].item()
        print(f"   Correlation: {correlation:.8f}")

        if diff2.mean().item() < 1e-6:
            print(f"   ✅ CHECKPOINT 2 MATCHES (diff < 1e-6)")
            print(f"   → Embedding lookup works correctly")
        elif diff2.mean().item() < 1e-3:
            print(f"   ⚠️  CHECKPOINT 2 SMALL DIFF (1e-6 < diff < 1e-3)")
            print(f"   → Minor numerical difference in embedding lookup")
        else:
            print(f"   ❌ CHECKPOINT 2 FAILS")
            print(f"   → Embedding lookup diverges!")
            print(f"   → Check: token indices, gather operation, or linear projection")

        if correlation < 0.9:
            print(f"\n   ⚠️  LOW CORRELATION: {correlation:.6f}")
            print(f"   → Suggests systematic error (transpose, wrong indices, etc.)")
    else:
        print(f"\n⚠️  Shape mismatch: Python {py_after_emb.shape} vs Swift {sw_after_emb.shape}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print(f"\n" + "=" * 80)
print("SUMMARY - First Layer Probe")
print("=" * 80)
print(f"\nThis probe isolates where encoder divergence begins:")
print(f"  1. If checkpoint 1 fails → weight loading issue")
print(f"  2. If checkpoint 1 passes but 2 fails → embedding lookup issue")
print(f"  3. If both pass → issue is later in encoder (transformers, residuals)")

print(f"\n" + "=" * 80)
