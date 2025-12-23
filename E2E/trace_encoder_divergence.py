"""
Trace encoder forward pass step-by-step to find where Swift diverges from Python
"""
import torch
import numpy as np
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"

print("=" * 80)
print("STEP-BY-STEP ENCODER DIVERGENCE ANALYSIS")
print("=" * 80)

# Load Swift encoder outputs
swift_data = load_file(str(FORENSIC_DIR / "swift_encoder_fixed.safetensors"))
swift_input_emb = swift_data["input_embeddings"]  # [348, 512]
swift_encoder_out = swift_data["encoder_output_before_proj"]  # [1, 696, 512]

# Load Python data
python_input_emb = load_file(str(FORENSIC_DIR / "python_input_embeddings.safetensors"))["input_embeddings"]  # [348, 512]
python_encoder_out = load_file(str(FORENSIC_DIR / "python_encoder_before_proj.safetensors"))["encoder_before_proj"]  # [1, 696, 512]

def compute_correlation(a, b):
    """Compute correlation between two tensors"""
    a_flat = a.flatten().numpy()
    b_flat = b.flatten().numpy()
    return np.corrcoef(a_flat, b_flat)[0, 1]

print("\n" + "=" * 80)
print("STAGE 1: Input Embeddings (token lookups)")
print("=" * 80)
print(f"Python shape: {python_input_emb.shape}")
print(f"Swift shape:  {swift_input_emb.shape}")
print(f"Python: mean={python_input_emb.mean().item():.6f}, std={python_input_emb.std().item():.6f}")
print(f"Swift:  mean={swift_input_emb.mean().item():.6f}, std={swift_input_emb.std().item():.6f}")
corr = compute_correlation(python_input_emb, swift_input_emb)
print(f"\n📊 CORRELATION: {corr:.6f}")
if corr > 0.99:
    print("✅ PERFECT! Input embeddings match exactly.")
elif corr > 0.9:
    print("✅ Good correlation - inputs are very similar")
else:
    print("❌ Poor correlation - inputs differ!")

print("\n" + "=" * 80)
print("STAGE 2: Final Encoder Output (before encoder_proj)")
print("=" * 80)
print(f"Python shape: {python_encoder_out.shape}")
print(f"Swift shape:  {swift_encoder_out.shape}")
print(f"Python: mean={python_encoder_out.mean().item():.6f}, std={python_encoder_out.std().item():.6f}")
print(f"Swift:  mean={swift_encoder_out.mean().item():.6f}, std={swift_encoder_out.std().item():.6f}")
print(f"Python range: [{python_encoder_out.min().item():.6f}, {python_encoder_out.max().item():.6f}]")
print(f"Swift range:  [{swift_encoder_out.min().item():.6f}, {swift_encoder_out.max().item():.6f}]")
corr = compute_correlation(python_encoder_out, swift_encoder_out)
print(f"\n📊 CORRELATION: {corr:.6f}")
if corr > 0.99:
    print("✅ PERFECT! Encoder outputs match exactly.")
elif corr > 0.9:
    print("✅ Good correlation - encoder is working")
else:
    print("❌ Poor correlation - divergence happened during encoder!")
    print("\n🔍 DIAGNOSIS:")
    print("   Input embeddings match perfectly (corr ~1.0)")
    print("   But encoder output differs significantly")
    print("   → Divergence happens INSIDE the encoder forward pass")
    print("   → Likely causes:")
    print("     1. LayerNorm epsilon mismatch")
    print("     2. Attention mask differences")
    print("     3. Positional encoding differences")
    print("     4. Dropout (but should be disabled in eval mode)")
    print("     5. Numerical precision differences")

print("\n" + "=" * 80)
