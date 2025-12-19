#!/usr/bin/env python3
"""
Test attention layer outputs with detailed intermediate values.
"""

import torch
import numpy as np
from pathlib import Path
import math

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
MODEL_DIR = PROJECT_ROOT / "models" / "chatterbox"
OUTPUT_DIR = PROJECT_ROOT / "verification_outputs" / "live"


def save_c_order(path, arr):
    """Save numpy array in C order (not Fortran)"""
    np.save(path, np.ascontiguousarray(arr))


def main():
    print("=" * 80)
    print("ATTENTION LAYER DETAILED TEST")
    print("=" * 80)

    # Load the model
    print("Loading model...")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device="cpu")
    s3 = model.s3gen
    estimator = s3.flow.decoder.estimator

    # Load the norm1 output
    norm1_out = torch.from_numpy(np.load(OUTPUT_DIR / "tfmr_test_norm1_out.npy"))  # [B, T, C]
    mask = torch.ones(1, 1, norm1_out.shape[1])  # Full mask

    print(f"norm1_out: {list(norm1_out.shape)}, range [{norm1_out.min().item():.4f}, {norm1_out.max().item():.4f}]")

    # Get the attention layer
    attn = estimator.down_blocks[0][1][0].attn1  # down_blocks[0][1][0] is first transformer, .attn1 is attention

    print("\n" + "-" * 80)
    print("ATTENTION LAYER ANALYSIS")
    print("-" * 80)

    # Print attention config
    print(f"\nAttention config:")
    print(f"  heads: {attn.heads}")
    print(f"  dim_head: {attn.inner_dim // attn.heads}")
    print(f"  inner_dim: {attn.inner_dim}")
    print(f"  to_q weight: {list(attn.to_q.weight.shape)}")
    print(f"  to_k weight: {list(attn.to_k.weight.shape)}")
    print(f"  to_v weight: {list(attn.to_v.weight.shape)}")
    print(f"  to_out[0] weight: {list(attn.to_out[0].weight.shape)}")
    print(f"  to_out[0] bias: {list(attn.to_out[0].bias.shape) if attn.to_out[0].bias is not None else 'None'}")

    # Check for q/k/v biases
    print(f"  to_q bias: {attn.to_q.bias is not None}")
    print(f"  to_k bias: {attn.to_k.bias is not None}")
    print(f"  to_v bias: {attn.to_v.bias is not None}")

    with torch.no_grad():
        # Save input
        save_c_order(OUTPUT_DIR / "attn_test_input.npy", norm1_out.numpy())

        # Manual attention computation with intermediates
        B, T, C = norm1_out.shape
        num_heads = attn.heads
        head_dim = attn.inner_dim // num_heads

        print(f"\n  B={B}, T={T}, C={C}")
        print(f"  num_heads={num_heads}, head_dim={head_dim}")

        # Q, K, V projection
        Q = attn.to_q(norm1_out)  # [B, T, inner_dim]
        K = attn.to_k(norm1_out)
        V = attn.to_v(norm1_out)

        print(f"\n1. Q: {list(Q.shape)}, range [{Q.min().item():.4f}, {Q.max().item():.4f}], mean {Q.mean().item():.6f}")
        print(f"2. K: {list(K.shape)}, range [{K.min().item():.4f}, {K.max().item():.4f}], mean {K.mean().item():.6f}")
        print(f"3. V: {list(V.shape)}, range [{V.min().item():.4f}, {V.max().item():.4f}], mean {V.mean().item():.6f}")

        save_c_order(OUTPUT_DIR / "attn_test_Q.npy", Q.numpy())
        save_c_order(OUTPUT_DIR / "attn_test_K.npy", K.numpy())
        save_c_order(OUTPUT_DIR / "attn_test_V.npy", V.numpy())

        # Reshape for multi-head attention
        # [B, T, H*D] -> [B, T, H, D] -> [B, H, T, D]
        Q_heads = Q.reshape(B, T, num_heads, head_dim).transpose(1, 2)
        K_heads = K.reshape(B, T, num_heads, head_dim).transpose(1, 2)
        V_heads = V.reshape(B, T, num_heads, head_dim).transpose(1, 2)

        print(f"\n4. Q_heads: {list(Q_heads.shape)}")
        print(f"5. K_heads: {list(K_heads.shape)}")
        print(f"6. V_heads: {list(V_heads.shape)}")

        # Compute attention scores
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(Q_heads, K_heads.transpose(-1, -2)) * scale
        print(f"\n7. scores: {list(scores.shape)}, range [{scores.min().item():.4f}, {scores.max().item():.4f}], mean {scores.mean().item():.6f}")
        save_c_order(OUTPUT_DIR / "attn_test_scores.npy", scores.numpy())

        # Softmax (no mask since we use full attention)
        weights = torch.softmax(scores, dim=-1)
        print(f"8. weights: {list(weights.shape)}, range [{weights.min().item():.4f}, {weights.max().item():.4f}], mean {weights.mean().item():.6f}")
        save_c_order(OUTPUT_DIR / "attn_test_weights.npy", weights.numpy())

        # Apply attention to values
        attended = torch.matmul(weights, V_heads)
        print(f"9. attended: {list(attended.shape)}, range [{attended.min().item():.4f}, {attended.max().item():.4f}], mean {attended.mean().item():.6f}")
        save_c_order(OUTPUT_DIR / "attn_test_attended.npy", attended.numpy())

        # Reshape back: [B, H, T, D] -> [B, T, H, D] -> [B, T, H*D]
        attended_reshape = attended.transpose(1, 2).reshape(B, T, -1)
        print(f"10. attended_reshape: {list(attended_reshape.shape)}, range [{attended_reshape.min().item():.4f}, {attended_reshape.max().item():.4f}]")
        save_c_order(OUTPUT_DIR / "attn_test_attended_reshape.npy", attended_reshape.numpy())

        # Output projection
        out = attn.to_out[0](attended_reshape)  # Linear
        print(f"11. out (after to_out): {list(out.shape)}, range [{out.min().item():.4f}, {out.max().item():.4f}], mean {out.mean().item():.6f}")
        save_c_order(OUTPUT_DIR / "attn_test_out.npy", out.numpy())

        # Run actual attention forward for comparison
        actual_out = attn(norm1_out)
        print(f"\n12. Actual attn forward: {list(actual_out.shape)}, range [{actual_out.min().item():.4f}, {actual_out.max().item():.4f}], mean {actual_out.mean().item():.6f}")

        diff = (out - actual_out).abs().max().item()
        print(f"    Diff between manual and actual: {diff:.2e}")

    print("\n" + "=" * 80)
    print("SAVED FILES:")
    for f in sorted(OUTPUT_DIR.glob("attn_test_*.npy")):
        arr = np.load(f)
        print(f"  {f.name}: {arr.shape}")


if __name__ == "__main__":
    main()
