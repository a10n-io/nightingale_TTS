#!/usr/bin/env python3
"""
Generate detailed Step 7a reference outputs with intermediate checkpoints.
This captures output after each major block in the decoder.
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
    print("STEP 7a DETAILED: DECODER INTERMEDIATE CHECKPOINTS")
    print("=" * 80)

    # Load the model
    print("Loading model...")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device="cpu")
    s3 = model.s3gen
    estimator = s3.flow.decoder.estimator

    # Load existing verification outputs
    print("Loading existing verification outputs...")
    mu = torch.from_numpy(np.load(OUTPUT_DIR / "step6_mu.npy"))  # [B, T, 80]
    x_cond = torch.from_numpy(np.load(OUTPUT_DIR / "step6_x_cond.npy"))  # [B, T, 80]
    spk_emb = torch.from_numpy(np.load(OUTPUT_DIR / "step5_spk_emb.npy"))  # [B, 80]
    mask = torch.from_numpy(np.load(OUTPUT_DIR / "step5_mask.npy"))  # [B, 1, T]

    # Adjust mask to match mu shape if needed
    L_total = mu.shape[1]
    if mask.shape[2] != L_total:
        print(f"Adjusting mask from {mask.shape[2]} to {L_total}")
        mask = torch.ones(1, 1, L_total)

    print(f"mu: {list(mu.shape)}")
    print(f"x_cond: {list(x_cond.shape)}")
    print(f"spk_emb: {list(spk_emb.shape)}")
    print(f"mask: {list(mask.shape)}")

    # Prepare inputs - transpose to NCT format for decoder
    mu_T = mu.transpose(1, 2)  # [B, 80, T]
    cond_T = x_cond.transpose(1, 2)  # [B, 80, T]
    mask_T = mask  # Already [B, 1, T]

    # Generate initial noise (deterministic)
    torch.manual_seed(0)
    initial_noise = torch.randn(1, 80, L_total, dtype=mu.dtype)

    # Time t=0
    t0 = torch.tensor([0.0])

    print("\n" + "=" * 80)
    print("TRACING DECODER FORWARD PASS")
    print("=" * 80)

    from einops import pack, rearrange, repeat
    from chatterbox.models.s3gen.utils.mask import add_optional_chunk_mask
    from chatterbox.models.s3gen.decoder import mask_to_bias

    with torch.no_grad():
        # Step 1: Time embeddings
        t_emb = estimator.time_embeddings(t0).to(t0.dtype)
        t_emb = estimator.time_mlp(t_emb)
        print(f"\n1. Time embedding:")
        print(f"   shape: {list(t_emb.shape)}")
        print(f"   range: [{t_emb.min().item():.4f}, {t_emb.max().item():.4f}]")
        save_c_order(OUTPUT_DIR / "step7a_d_time_emb.npy", t_emb.numpy())

        # Step 2: Input packing
        x = initial_noise
        x = pack([x, mu_T], "b * t")[0]  # [B, 160, T]
        print(f"\n2. After pack([x, mu]):")
        print(f"   shape: {list(x.shape)}")

        spks = repeat(spk_emb, "b c -> b c t", t=x.shape[-1])
        x = pack([x, spks], "b * t")[0]  # [B, 240, T]
        print(f"\n3. After pack with spks:")
        print(f"   shape: {list(x.shape)}")

        x = pack([x, cond_T], "b * t")[0]  # [B, 320, T]
        print(f"\n4. After pack with cond (h_input):")
        print(f"   shape: {list(x.shape)}")
        print(f"   range: [{x.min().item():.4f}, {x.max().item():.4f}]")
        save_c_order(OUTPUT_DIR / "step7a_d_h_input.npy", x.numpy())

        # Step 3: Down blocks
        hiddens = []
        masks = [mask_T]

        for i, (resnet, transformer_blocks, downsample) in enumerate(estimator.down_blocks):
            mask_down = masks[-1]

            # Resnet
            x = resnet(x, mask_down, t_emb)
            print(f"\n5.{i}a. After down_blocks[{i}].resnet:")
            print(f"   shape: {list(x.shape)}")
            print(f"   range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            save_c_order(OUTPUT_DIR / f"step7a_d_down{i}_resnet.npy", x.numpy())

            # Transformer blocks
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_down.bool(), False, False, 0, estimator.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
            print(f"   attn_mask shape: {list(attn_mask.shape)}")

            for j, transformer_block in enumerate(transformer_blocks):
                x_before = x.clone()
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t_emb,
                )
                if j == 0:
                    print(f"\n5.{i}b. After down_blocks[{i}].transformer[0]:")
                    print(f"   shape: {list(x.shape)}")
                    print(f"   range: [{x.min().item():.4f}, {x.max().item():.4f}]")
                    save_c_order(OUTPUT_DIR / f"step7a_d_down{i}_tfmr0.npy", x.numpy())

            x = rearrange(x, "b t c -> b c t").contiguous()
            print(f"\n5.{i}c. After down_blocks[{i}] all transformers (transposed back):")
            print(f"   shape: {list(x.shape)}")
            print(f"   range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            save_c_order(OUTPUT_DIR / f"step7a_d_down{i}_tfmrs.npy", x.numpy())

            hiddens.append(x)

            # Downsample
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
            print(f"\n5.{i}d. After down_blocks[{i}].downsample:")
            print(f"   shape: {list(x.shape)}")
            print(f"   range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            save_c_order(OUTPUT_DIR / f"step7a_d_down{i}_final.npy", x.numpy())

        masks = masks[:-1]
        mask_mid = masks[-1]

        # Step 4: Mid blocks
        for i, (resnet, transformer_blocks) in enumerate(estimator.mid_blocks):
            x = resnet(x, mask_mid, t_emb)
            if i < 3 or i >= len(estimator.mid_blocks) - 2:  # First 3 and last 2
                print(f"\n6.{i}a. After mid_blocks[{i}].resnet:")
                print(f"   shape: {list(x.shape)}")
                print(f"   range: [{x.min().item():.4f}, {x.max().item():.4f}]")
                save_c_order(OUTPUT_DIR / f"step7a_d_mid{i}_resnet.npy", x.numpy())

            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_mid.bool(), False, False, 0, estimator.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)

            for j, transformer_block in enumerate(transformer_blocks):
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t_emb,
                )

            x = rearrange(x, "b t c -> b c t").contiguous()
            if i < 3 or i >= len(estimator.mid_blocks) - 2:
                print(f"\n6.{i}b. After mid_blocks[{i}] complete:")
                print(f"   shape: {list(x.shape)}")
                print(f"   range: [{x.min().item():.4f}, {x.max().item():.4f}]")
                save_c_order(OUTPUT_DIR / f"step7a_d_mid{i}_final.npy", x.numpy())

        print(f"\nAfter all mid blocks:")
        print(f"   shape: {list(x.shape)}")
        print(f"   range: [{x.min().item():.4f}, {x.max().item():.4f}]")
        save_c_order(OUTPUT_DIR / "step7a_d_after_mid.npy", x.numpy())

        # Step 5: Up blocks
        for i, (resnet, transformer_blocks, upsample) in enumerate(estimator.up_blocks):
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]

            print(f"\n7.{i}a. After up_blocks[{i}] concat:")
            print(f"   shape: {list(x.shape)}")
            print(f"   range: [{x.min().item():.4f}, {x.max().item():.4f}]")

            x = resnet(x, mask_up, t_emb)
            print(f"\n7.{i}b. After up_blocks[{i}].resnet:")
            print(f"   shape: {list(x.shape)}")
            print(f"   range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            save_c_order(OUTPUT_DIR / f"step7a_d_up{i}_resnet.npy", x.numpy())

            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_up.bool(), False, False, 0, estimator.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)

            for j, transformer_block in enumerate(transformer_blocks):
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t_emb,
                )

            x = rearrange(x, "b t c -> b c t").contiguous()
            x = upsample(x * mask_up)
            print(f"\n7.{i}c. After up_blocks[{i}] complete:")
            print(f"   shape: {list(x.shape)}")
            print(f"   range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            save_c_order(OUTPUT_DIR / f"step7a_d_up{i}_final.npy", x.numpy())

        # Step 6: Final block and projection
        x = estimator.final_block(x, mask_up)
        print(f"\n8. After final_block:")
        print(f"   shape: {list(x.shape)}")
        print(f"   range: [{x.min().item():.4f}, {x.max().item():.4f}]")
        save_c_order(OUTPUT_DIR / "step7a_d_final_block.npy", x.numpy())

        output = estimator.final_proj(x * mask_up)
        print(f"\n9. After final_proj:")
        print(f"   shape: {list(output.shape)}")
        print(f"   range: [{output.min().item():.4f}, {output.max().item():.4f}]")

        output = output * mask_T
        print(f"\n10. After mask multiply (final output):")
        print(f"   shape: {list(output.shape)}")
        print(f"   range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        save_c_order(OUTPUT_DIR / "step7a_d_output.npy", output.numpy())

    print("\n" + "=" * 80)
    print("SAVED FILES:")
    for f in sorted(OUTPUT_DIR.glob("step7a_d_*.npy")):
        arr = np.load(f)
        print(f"  {f.name}: {arr.shape}")


if __name__ == "__main__":
    main()
