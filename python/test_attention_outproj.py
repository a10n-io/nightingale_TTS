#!/usr/bin/env python3
"""
Test attention output projection specifically.
"""

import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
MODEL_DIR = PROJECT_ROOT / "models" / "chatterbox"
OUTPUT_DIR = PROJECT_ROOT / "verification_outputs" / "live"


def save_c_order(path, arr):
    """Save numpy array in C order (not Fortran)"""
    np.save(path, np.ascontiguousarray(arr))


def main():
    print("=" * 80)
    print("ATTENTION OUTPUT PROJECTION TEST")
    print("=" * 80)

    # Load the model
    print("Loading model...")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device="cpu")
    s3 = model.s3gen
    estimator = s3.flow.decoder.estimator

    # Get the attention layer
    attn = estimator.down_blocks[0][1][0].attn1

    # Load the attended values
    attended = torch.from_numpy(np.load(OUTPUT_DIR / "attn_test_attended.npy"))  # [B, H, T, D]

    print(f"attended: {list(attended.shape)}, range [{attended.min().item():.4f}, {attended.max().item():.4f}]")

    # Print out_proj details
    print("\n" + "-" * 80)
    print("OUTPUT PROJECTION ANALYSIS")
    print("-" * 80)

    print(f"\nto_out structure: {attn.to_out}")
    print(f"to_out[0] weight shape: {list(attn.to_out[0].weight.shape)}")
    print(f"to_out[0] bias shape: {list(attn.to_out[0].bias.shape) if attn.to_out[0].bias is not None else 'None'}")

    # Check if there's a second component in to_out (dropout, etc.)
    print(f"to_out length: {len(attn.to_out)}")
    for i, module in enumerate(attn.to_out):
        print(f"  to_out[{i}]: {type(module).__name__}")

    with torch.no_grad():
        # Reshape attended: [B, H, T, D] -> [B, T, H, D] -> [B, T, H*D]
        B, H, T, D = attended.shape
        attended_reshape = attended.transpose(1, 2).reshape(B, T, H * D)
        print(f"\nattended_reshape: {list(attended_reshape.shape)}, range [{attended_reshape.min().item():.4f}, {attended_reshape.max().item():.4f}]")

        # Apply output projection manually
        out_manual = attn.to_out[0](attended_reshape)  # Linear only
        print(f"out_manual (after to_out[0] Linear): {list(out_manual.shape)}, range [{out_manual.min().item():.4f}, {out_manual.max().item():.4f}], mean {out_manual.mean().item():.6f}")

        # Note: to_out is a ModuleList, can't call directly. The dropout is identity at inference.

        # Save outputs
        save_c_order(OUTPUT_DIR / "attn_out_attended_reshape.npy", attended_reshape.numpy())
        save_c_order(OUTPUT_DIR / "attn_out_manual.npy", out_manual.numpy())

        # Save weights and bias for Swift comparison
        print("\n--- Saving weights for Swift comparison ---")
        weight = attn.to_out[0].weight.detach().numpy()
        bias = attn.to_out[0].bias.detach().numpy()
        save_c_order(OUTPUT_DIR / "attn_out_weight.npy", weight)
        save_c_order(OUTPUT_DIR / "attn_out_bias.npy", bias)

        print(f"Weight: {weight.shape}, range [{weight.min():.4f}, {weight.max():.4f}]")
        print(f"Bias: {bias.shape}, range [{bias.min():.4f}, {bias.max():.4f}]")
        print(f"Weight first 5: {weight[0, :5]}")
        print(f"Bias first 5: {bias[:5]}")


if __name__ == "__main__":
    main()
