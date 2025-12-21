#!/usr/bin/env python3
"""Generate Step 7 (ODE Solver) reference outputs for Swift verification."""

import torch
import numpy as np
from pathlib import Path
import sys
import math

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
sys.path.insert(0, str(PROJECT_ROOT / "python"))

def main():
    print("=" * 80)
    print("GENERATING STEP 7 (ODE SOLVER) REFERENCE OUTPUTS")
    print("=" * 80)

    # Load Python model
    print("\nLoading Python model...")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model_dir = PROJECT_ROOT / "models" / "chatterbox"
    model = ChatterboxMultilingualTTS.from_local(str(model_dir), device="cpu")
    flow = model.s3gen.flow

    # Reference directory
    ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

    # Load Step 7a inputs (these are the ODE starting point)
    print("\nLoading Step 7a inputs...")
    initial_noise = torch.from_numpy(np.load(ref_dir / "step7a_initial_noise.npy"))
    mu_T = torch.from_numpy(np.load(ref_dir / "step7a_mu_T.npy"))  # [1, 80, T]
    cond_T = torch.from_numpy(np.load(ref_dir / "step7a_cond_T.npy"))  # [1, 80, T]
    mask_T = torch.from_numpy(np.load(ref_dir / "step7a_mask_T.npy"))  # [1, 1, T]
    spk_emb = torch.from_numpy(np.load(ref_dir / "step5_spk_emb.npy"))  # [1, 80]

    print(f"  initial_noise: {initial_noise.shape}")
    print(f"  mu_T: {mu_T.shape}")
    print(f"  cond_T: {cond_T.shape}")
    print(f"  mask_T: {mask_T.shape}")
    print(f"  spk_emb: {spk_emb.shape}")

    # ODE Parameters (from Python's flow.py and flow_matching.py)
    n_timesteps = 10
    cfg_rate = 0.7  # inference_cfg_rate

    # Generate time span with cosine scheduling
    t_span = torch.linspace(0, 1, n_timesteps + 1)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)  # Cosine schedule
    print(f"\nODE Configuration:")
    print(f"  n_timesteps: {n_timesteps}")
    print(f"  cfg_rate: {cfg_rate}")
    print(f"  t_span: {t_span.tolist()}")

    # Get decoder/estimator
    estimator = flow.decoder.estimator

    # Save ODE trace
    trace = {}
    trace["t_span"] = t_span.numpy()
    trace["initial_noise"] = initial_noise.numpy()
    trace["mu_T"] = mu_T.numpy()
    trace["cond_T"] = cond_T.numpy()
    trace["spk_emb"] = spk_emb.numpy()

    # Prepare batch-doubled inputs for CFG (2B total)
    B, C, T = initial_noise.shape
    x = initial_noise.clone()

    # Pre-allocate batch inputs
    x_in = torch.zeros([2 * B, C, T])
    mu_in = torch.zeros([2 * B, C, T])
    mask_in = torch.zeros([2 * B, 1, T])
    spks_in = torch.zeros([2 * B, 80])
    cond_in = torch.zeros([2 * B, C, T])
    t_in = torch.zeros([2 * B])

    print("\n" + "=" * 80)
    print("TRACING ODE STEPS")
    print("=" * 80)

    with torch.no_grad():
        for step, (t, r) in enumerate(zip(t_span[:-1], t_span[1:])):
            t_scalar = t.unsqueeze(0)
            r_scalar = r.unsqueeze(0)

            # Prepare duplicate batch for CFG
            x_in[:B] = x
            x_in[B:] = x
            mask_in[:B] = mask_T
            mask_in[B:] = mask_T
            mu_in[:B] = mu_T  # Only first half gets conditional mu
            # mu_in[B:] stays zeros
            t_in[:B] = t_scalar
            t_in[B:] = t_scalar
            spks_in[:B] = spk_emb  # Only first half gets speaker conditioning
            # spks_in[B:] stays zeros
            cond_in[:B] = cond_T  # Only first half gets conditioning
            # cond_in[B:] stays zeros

            # Decoder call
            dxdt = estimator.forward(
                x=x_in,
                mask=mask_in,
                mu=mu_in,
                t=t_in,
                spks=spks_in,
                cond=cond_in,
            )

            # Split output for CFG
            dxdt_cond = dxdt[:B]
            dxdt_uncond = dxdt[B:]

            # Apply CFG blending
            dxdt_cfg = (1.0 + cfg_rate) * dxdt_cond - cfg_rate * dxdt_uncond

            # Euler step
            dt = r - t
            x_before = x.clone()
            x = x + dt * dxdt_cfg

            print(f"\n--- ODE Step {step + 1}/{n_timesteps} ---")
            print(f"   t = {t.item():.6f}, dt = {dt.item():.6f}")
            print(f"   x before: [{x_before.min().item():.4f}, {x_before.max().item():.4f}]")
            print(f"   dxdt (cond): [{dxdt_cond.min().item():.4f}, {dxdt_cond.max().item():.4f}]")
            print(f"   dxdt (uncond): [{dxdt_uncond.min().item():.4f}, {dxdt_uncond.max().item():.4f}]")
            print(f"   dxdt (after CFG): [{dxdt_cfg.min().item():.4f}, {dxdt_cfg.max().item():.4f}]")
            print(f"   x after: [{x.min().item():.4f}, {x.max().item():.4f}]")

            # Save trace for this step
            trace[f"step{step + 1}_t"] = np.array([t.item()])
            trace[f"step{step + 1}_dt"] = np.array([dt.item()])
            trace[f"step{step + 1}_x_before"] = x_before.numpy()
            trace[f"step{step + 1}_dxdt_cond"] = dxdt_cond.numpy()
            trace[f"step{step + 1}_dxdt_uncond"] = dxdt_uncond.numpy()
            trace[f"step{step + 1}_dxdt_cfg"] = dxdt_cfg.numpy()
            trace[f"step{step + 1}_x_after"] = x.numpy()

    # Final output
    trace["final_mel"] = x.numpy()
    print(f"\nFinal mel: {x.shape}, range=[{x.min().item():.4f}, {x.max().item():.4f}]")

    # Save all traces
    output_dir = ref_dir
    print(f"\nSaving Step 7 reference outputs to {output_dir}...")

    for key, value in trace.items():
        np.save(output_dir / f"step7_{key}.npy", value)
        print(f"  Saved step7_{key}.npy: {value.shape}")

    print("\n✅ Step 7 reference outputs generated successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
