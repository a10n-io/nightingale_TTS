#!/usr/bin/env python3
"""
Test transformer block outputs at each step.
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
    print("TRANSFORMER BLOCK DETAILED TEST")
    print("=" * 80)

    # Load the model
    print("Loading model...")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device="cpu")
    s3 = model.s3gen
    estimator = s3.flow.decoder.estimator

    # Load the resnet output
    resnet_out = torch.from_numpy(np.load(OUTPUT_DIR / "step7a_d_down0_resnet.npy"))  # [B, C, T]
    mask = torch.ones(1, 1, resnet_out.shape[2])  # Full mask

    print(f"resnet_out: {list(resnet_out.shape)}, range [{resnet_out.min().item():.4f}, {resnet_out.max().item():.4f}]")

    # Import required functions
    from einops import rearrange
    from chatterbox.models.s3gen.utils.mask import add_optional_chunk_mask
    from chatterbox.models.s3gen.decoder import mask_to_bias

    with torch.no_grad():
        # Transpose to [B, T, C]
        x = rearrange(resnet_out, "b c t -> b t c").contiguous()
        print(f"\nInput x (transposed): {list(x.shape)}, range [{x.min().item():.4f}, {x.max().item():.4f}]")

        # Create attention mask
        attn_mask = add_optional_chunk_mask(x, mask.bool(), False, False, 0, estimator.static_chunk_size, -1)
        attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
        print(f"attn_mask: {list(attn_mask.shape)}, range [{attn_mask.min().item():.4f}, {attn_mask.max().item():.4f}]")

        # Get the first transformer block
        tfmr = estimator.down_blocks[0][1][0]  # down_blocks[0][1] is transformer_blocks list, [0] is first block

        print("\n" + "-" * 80)
        print("TRACING TRANSFORMER BLOCK FORWARD PASS")
        print("-" * 80)

        # Save input
        save_c_order(OUTPUT_DIR / "tfmr_test_input.npy", x.numpy())

        # Step 1: norm1
        h = x
        print(f"\n1. Input h: {list(h.shape)}, range [{h.min().item():.4f}, {h.max().item():.4f}], mean {h.mean().item():.4f}")

        norm1_out = tfmr.norm1(h)
        print(f"2. After norm1: {list(norm1_out.shape)}, range [{norm1_out.min().item():.4f}, {norm1_out.max().item():.4f}], mean {norm1_out.mean().item():.4f}")
        save_c_order(OUTPUT_DIR / "tfmr_test_norm1_out.npy", norm1_out.numpy())

        # Step 2: attn1
        attn_out = tfmr.attn1(
            norm1_out,
            attention_mask=attn_mask,
        )
        print(f"3. After attn1: {list(attn_out.shape)}, range [{attn_out.min().item():.4f}, {attn_out.max().item():.4f}], mean {attn_out.mean().item():.4f}")
        save_c_order(OUTPUT_DIR / "tfmr_test_attn1_out.npy", attn_out.numpy())

        # Step 3: residual 1
        h = attn_out + x
        print(f"4. After residual1: {list(h.shape)}, range [{h.min().item():.4f}, {h.max().item():.4f}], mean {h.mean().item():.4f}")
        save_c_order(OUTPUT_DIR / "tfmr_test_res1_out.npy", h.numpy())

        # Note: attn2/norm2 are None in this config, so skip

        # Step 4: norm3 (maps to norm2 in Swift)
        norm3_out = tfmr.norm3(h)
        print(f"5. After norm3: {list(norm3_out.shape)}, range [{norm3_out.min().item():.4f}, {norm3_out.max().item():.4f}], mean {norm3_out.mean().item():.4f}")
        save_c_order(OUTPUT_DIR / "tfmr_test_norm3_out.npy", norm3_out.numpy())

        # Step 5: ff
        ff_out = tfmr.ff(norm3_out)
        print(f"6. After ff: {list(ff_out.shape)}, range [{ff_out.min().item():.4f}, {ff_out.max().item():.4f}], mean {ff_out.mean().item():.4f}")
        save_c_order(OUTPUT_DIR / "tfmr_test_ff_out.npy", ff_out.numpy())

        # Step 6: residual 2
        final = ff_out + h
        print(f"7. After residual2 (final): {list(final.shape)}, range [{final.min().item():.4f}, {final.max().item():.4f}], mean {final.mean().item():.4f}")
        save_c_order(OUTPUT_DIR / "tfmr_test_final.npy", final.numpy())

        # Also run the full forward pass for comparison
        full_out = tfmr(
            hidden_states=x,
            attention_mask=attn_mask,
        )
        print(f"\n8. Full forward pass: {list(full_out.shape)}, range [{full_out.min().item():.4f}, {full_out.max().item():.4f}], mean {full_out.mean().item():.4f}")

        # Verify full forward matches step-by-step
        diff = (final - full_out).abs().max().item()
        print(f"   Diff between step-by-step and full forward: {diff:.2e}")

    print("\n" + "=" * 80)
    print("SAVED FILES:")
    for f in sorted(OUTPUT_DIR.glob("tfmr_test_*.npy")):
        arr = np.load(f)
        print(f"  {f.name}: {arr.shape}")


if __name__ == "__main__":
    main()
