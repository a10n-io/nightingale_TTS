#!/usr/bin/env python3
"""
Generate Step 7a reference output (single decoder forward pass).
This uses existing verification outputs to generate the decoder velocity at t=0.
"""

import torch
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
MODEL_DIR = PROJECT_ROOT / "models" / "chatterbox"

def main():
    # Get reference directory from command line or use default
    if len(sys.argv) > 1:
        ref_dir = Path(sys.argv[1])
    else:
        ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

    print("=" * 80)
    print("STEP 7a: GENERATE DECODER SINGLE FORWARD PASS REFERENCE")
    print(f"Reference directory: {ref_dir}")
    print("=" * 80)

    # Load the model
    print("Loading model...")
    sys.path.insert(0, str(PROJECT_ROOT / "python"))
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device="cpu")
    s3 = model.s3gen

    # Load existing verification outputs (Step 6 outputs)
    print("Loading existing verification outputs...")
    mu = torch.from_numpy(np.load(ref_dir / "step6_mu.npy"))  # [B, 80, T]
    x_cond = torch.from_numpy(np.load(ref_dir / "step6_x_cond.npy"))  # [B, 80, T]
    spk_emb = torch.from_numpy(np.load(ref_dir / "step5_spk_emb.npy"))  # [B, 80]

    # Note: mu and x_cond from Step 6 are already in [B, 80, T] format (transposed)
    L_total = mu.shape[2]

    # Create mask
    mask = torch.ones(1, 1, L_total)

    print(f"mu: {list(mu.shape)}")
    print(f"x_cond: {list(x_cond.shape)}")
    print(f"spk_emb: {list(spk_emb.shape)}")
    print(f"mask: {list(mask.shape)}")

    # Get the estimator (decoder)
    estimator = s3.flow.decoder.estimator

    # Since mu and x_cond are already transposed [B, 80, T], use directly
    mu_T = mu
    cond_T = x_cond
    mask_T = mask

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

    # Run single forward pass
    with torch.no_grad():
        velocity_t0 = estimator.forward(
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
    print(f"velocity_t0[0,0,:5]: {velocity_t0[0,0,:5].tolist()}")

    # Save outputs (use ascontiguousarray to ensure C order)
    def save_c_order(path, arr):
        np.save(path, np.ascontiguousarray(arr))

    save_c_order(ref_dir / "step7a_velocity_t0.npy", velocity_t0.detach().cpu().numpy())
    save_c_order(ref_dir / "step7a_initial_noise.npy", initial_noise.detach().cpu().numpy())
    save_c_order(ref_dir / "step7a_mu_T.npy", mu_T.detach().cpu().numpy())
    save_c_order(ref_dir / "step7a_cond_T.npy", cond_T.detach().cpu().numpy())
    save_c_order(ref_dir / "step7a_mask_T.npy", mask_T.detach().cpu().numpy())

    print(f"\nSaved Step 7a outputs to {ref_dir}")
    print("Files:")
    for f in sorted(ref_dir.glob("step7a_*.npy")):
        arr = np.load(f)
        print(f"  {f.name}: {arr.shape} ({arr.dtype})")


if __name__ == "__main__":
    main()
