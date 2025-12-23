"""
Correctly test if Swift's transposition logic works.
"""
import torch
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Load weights and tokens
flow_data = load_file(str(PROJECT_ROOT / "models/chatterbox/s3gen.safetensors"))
tokens_data = load_file(str(PROJECT_ROOT / "test_audio/cross_validate/python_speech_tokens.safetensors"))

input_emb_weight = flow_data["flow.input_embedding.weight"]
weight_pytorch = flow_data["flow.encoder.embed.out.0.weight"]  # [out=512, in=512] PyTorch format
bias = flow_data["flow.encoder.embed.out.0.bias"]
tokens = tokens_data["speech_tokens"]

# Embedding lookup
token_embs = input_emb_weight[tokens]  # [T, 512]

print("=" * 80)
print("WEIGHT TRANSPOSITION TEST")
print("=" * 80)

print(f"\nPyTorch weight shape: {weight_pytorch.shape} = [out={weight_pytorch.shape[0]}, in={weight_pytorch.shape[1]}]")

# ==============================================================================
# Case 1: CORRECT Python behavior
# ==============================================================================
print(f"\n" + "=" * 80)
print("Case 1: Python (CORRECT - uses weight.T)")
print("=" * 80)

output_python = token_embs @ weight_pytorch.T + bias  # PyTorch Linear: x @ W.T
print(f"   Formula: tokenEmbs @ weight_pytorch.T")
print(f"   Output: mean={output_python.mean().item():.6f}, std={output_python.std().item():.6f}")

# ==============================================================================
# Case 2: Swift AFTER remapS3Keys() transposes (CORRECT if used properly)
# ==============================================================================
print(f"\n" + "=" * 80)
print("Case 2: Swift after remapS3Keys() transposes weight")
print("=" * 80)

# remapS3Keys() would transpose: [out, in] -> [in, out]
weight_after_remap = weight_pytorch.T  # [in=512, out=512]
print(f"   After remapS3Keys transpose: shape={weight_after_remap.shape} = [in={weight_after_remap.shape[0]}, out={weight_after_remap.shape[1]}]")

# FixedLinear does: matmul(input, weight)
output_swift_correct = token_embs @ weight_after_remap + bias
print(f"   Formula: tokenEmbs @ weight_transposed")
print(f"   Output: mean={output_swift_correct.mean().item():.6f}, std={output_swift_correct.std().item():.6f}")

diff = (output_python - output_swift_correct).abs().mean()
print(f"\n   Difference from Python: {diff.item():.10f}")
if diff < 1e-5:
    print(f"   ✅ CORRECT! Swift would match Python if using transposed weight")

# ==============================================================================
# Case 3: Swift if NO transpose happens (BUG)
# ==============================================================================
print(f"\n" + "=" * 80)
print("Case 3: Swift if remapS3Keys() does NOT transpose (BUG)")
print("=" * 80)

# If weight is NOT transposed, Swift would use [out, in] directly
output_swift_no_transpose = token_embs @ weight_pytorch + bias
print(f"   Formula: tokenEmbs @ weight_pytorch (wrong!)")
print(f"   Output: mean={output_swift_no_transpose.mean().item():.6f}, std={output_swift_no_transpose.std().item():.6f}")

diff_no_transpose = (output_python - output_swift_no_transpose).abs().mean()
print(f"\n   Difference from Python: {diff_no_transpose.item():.10f}")
if diff_no_transpose > 0.1:
    print(f"   ❌ WRONG! This is what happens without transposition")

# ==============================================================================
# Case 4: Swift if DOUBLE transpose (BUG)
# ==============================================================================
print(f"\n" + "=" * 80)
print("Case 4: Swift if DOUBLE transpose happens (BUG)")
print("=" * 80)

# If remapS3Keys() transposes AND then something else transposes again:
# [out, in] -> [in, out] -> [out, in]  (back to original!)
weight_double_transpose = weight_after_remap.T  # Back to [out, in]
output_swift_double_transpose = token_embs @ weight_double_transpose + bias
print(f"   Formula: tokenEmbs @ weight_double_transposed")
print(f"   Output: mean={output_swift_double_transpose.mean().item():.6f}, std={output_swift_double_transpose.std().item():.6f}")

diff_double = (output_python - output_swift_double_transpose).abs().mean()
print(f"\n   Difference from Python: {diff_double.item():.10f}")
if diff_double > 0.1:
    print(f"   ❌ WRONG! This is what happens with double transposition")

# ==============================================================================
# SUMMARY
# ==============================================================================
print(f"\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nCase 1 (Python correct):        mean={output_python.mean().item():.6f}, std={output_python.std().item():.6f}")
print(f"Case 2 (Swift correct):         mean={output_swift_correct.mean().item():.6f}, std={output_swift_correct.std().item():.6f}, diff={diff.item():.10f}")
print(f"Case 3 (No transpose BUG):      mean={output_swift_no_transpose.mean().item():.6f}, std={output_swift_no_transpose.std().item():.6f}, diff={diff_no_transpose.item():.6f}")
print(f"Case 4 (Double transpose BUG):  mean={output_swift_double_transpose.mean().item():.6f}, std={output_swift_double_transpose.std().item():.6f}, diff={diff_double.item():.6f}")

print(f"\n✅ If Swift matches Case 2 → Transposition is CORRECT")
print(f"❌ If Swift matches Case 3 → NO transposition (bug in remapS3Keys)")
print(f"❌ If Swift matches Case 4 → DOUBLE transposition (bug in update())")

print(f"\n" + "=" * 80)
