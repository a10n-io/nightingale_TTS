#!/usr/bin/env python3
"""
Generate Step 7a reference output (single decoder forward pass).
This uses existing verification outputs to generate the decoder velocity at t=0.
"""

import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
MODEL_DIR = PROJECT_ROOT / "models" / "chatterbox"
OUTPUT_DIR = PROJECT_ROOT / "verification_outputs" / "live"


def main():
    print("=" * 80)
    print("STEP 7a: GENERATE DECODER SINGLE FORWARD PASS REFERENCE")
    print("=" * 80)

    # Load the model
    print("Loading model...")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device="cpu")
    s3 = model.s3gen

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

    # Get the estimator
    estimator = s3.flow.decoder.estimator

    # Prepare inputs - transpose to NCT format for decoder
    mu_T = mu.transpose(1, 2)  # [B, 80, T]
    cond_T = x_cond.transpose(1, 2)  # [B, 80, T]
    mask_T = mask  # Already [B, 1, T]

    # Generate initial noise (deterministic)
    torch.manual_seed(0)
    initial_noise = torch.randn(1, 80, L_total, dtype=mu.dtype)

    # Time t=0
    t0 = torch.tensor([0.0])

    print(f"\nRunning estimator forward with:")
    print(f"  x (noise): {list(initial_noise.shape)}")
    print(f"  mask: {list(mask_T.shape)}")
    print(f"  mu: {list(mu_T.shape)}")
    print(f"  t: {t0.item()}")
    print(f"  spks: {list(spk_emb.shape)}")
    print(f"  cond: {list(cond_T.shape)}")

    # Run single forward pass with intermediate output capture
    with torch.no_grad():
        decoder = estimator

        # Step 1: Time embeddings (sinusoidal)
        t_sinusoidal = decoder.time_embeddings(t0)
        print(f"\nt_sinusoidal: {list(t_sinusoidal.shape)}")
        print(f"t_sinusoidal range: [{t_sinusoidal.min().item():.4f}, {t_sinusoidal.max().item():.4f}]")
        print(f"t_sinusoidal mean: {t_sinusoidal.mean().item():.4f}")
        np.save(OUTPUT_DIR / "step7a_t_sinusoidal.npy", np.ascontiguousarray(t_sinusoidal.detach().cpu().numpy()))

        # Step 2: Time MLP
        t_emb = decoder.time_mlp(t_sinusoidal)
        print(f"t_emb (after time_mlp): {list(t_emb.shape)}")
        print(f"t_emb range: [{t_emb.min().item():.4f}, {t_emb.max().item():.4f}]")
        print(f"t_emb mean: {t_emb.mean().item():.4f}")
        np.save(OUTPUT_DIR / "step7a_time_emb.npy", np.ascontiguousarray(t_emb.detach().cpu().numpy()))

        # Step 3: Manually trace decoder internals
        from einops import pack, repeat

        # Input packing (matches decoder.forward)
        h = pack([initial_noise, mu_T], "b * t")[0]  # [B, 160, T]
        print(f"\nh after pack([x, mu]): {list(h.shape)}")
        print(f"h range: [{h.min().item():.4f}, {h.max().item():.4f}]")

        spk_repeat = repeat(spk_emb, "b c -> b c t", t=h.shape[-1])
        h = pack([h, spk_repeat], "b * t")[0]  # [B, 240, T]
        print(f"h after pack with spks: {list(h.shape)}")

        h = pack([h, cond_T], "b * t")[0]  # [B, 320, T]
        print(f"h after pack with cond: {list(h.shape)}")
        print(f"h_input range: [{h.min().item():.4f}, {h.max().item():.4f}]")
        np.save(OUTPUT_DIR / "step7a_h_input.npy", np.ascontiguousarray(h.detach().cpu().numpy()))

        # Run first resnet block
        down_block_0 = decoder.down_blocks[0]
        resnet_0 = down_block_0[0]
        h_after_resnet = resnet_0(h, mask_T, t_emb)
        print(f"\nAfter down_blocks[0] resnet: {list(h_after_resnet.shape)}")
        print(f"h_after_resnet range: [{h_after_resnet.min().item():.4f}, {h_after_resnet.max().item():.4f}]")
        print(f"h_after_resnet mean: {h_after_resnet.mean().item():.4f}")
        np.save(OUTPUT_DIR / "step7a_after_resnet0.npy", np.ascontiguousarray(h_after_resnet.detach().cpu().numpy()))

        # Now run the full forward pass
        velocity_t0 = decoder.forward(
            x=initial_noise,
            mask=mask_T,
            mu=mu_T,
            t=t0,
            spks=spk_emb,
            cond=cond_T
        )

    print(f"\nvelocity_t0: {list(velocity_t0.shape)}")
    print(f"velocity_t0 range: [{velocity_t0.min().item():.4f}, {velocity_t0.max().item():.4f}]")
    print(f"velocity_t0 mean: {velocity_t0.mean().item():.4f}")
    print(f"velocity_t0 std: {velocity_t0.std().item():.4f}")

    # Save outputs (use ascontiguousarray to ensure C order, not Fortran)
    def save_c_order(path, arr):
        np.save(path, np.ascontiguousarray(arr))

    save_c_order(OUTPUT_DIR / "step7a_velocity_t0.npy", velocity_t0.detach().cpu().numpy())
    save_c_order(OUTPUT_DIR / "step7a_initial_noise.npy", initial_noise.detach().cpu().numpy())
    save_c_order(OUTPUT_DIR / "step7a_mu_T.npy", mu_T.detach().cpu().numpy())
    save_c_order(OUTPUT_DIR / "step7a_cond_T.npy", cond_T.detach().cpu().numpy())
    save_c_order(OUTPUT_DIR / "step7a_mask_T.npy", mask_T.detach().cpu().numpy())

    print(f"\nSaved Step 7a outputs to {OUTPUT_DIR}")
    print("Files:")
    for f in sorted(OUTPUT_DIR.glob("step7a_*.npy")):
        arr = np.load(f)
        print(f"  {f.name}: {arr.shape} ({arr.dtype})")


if __name__ == "__main__":
    main()
