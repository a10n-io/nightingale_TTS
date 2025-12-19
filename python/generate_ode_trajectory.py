#!/usr/bin/env python3
"""
Export full ODE trajectory to debug where divergence happens.
"""

import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
MODEL_DIR = PROJECT_ROOT / "models" / "chatterbox"
OUTPUT_DIR = PROJECT_ROOT / "verification_outputs" / "live"


def save_c_order(path, arr):
    """Save numpy array in C order (not Fortran) as float32"""
    arr_f32 = np.ascontiguousarray(arr).astype(np.float32)
    np.save(path, arr_f32)


def main():
    print("=" * 80)
    print("ODE TRAJECTORY EXPORT")
    print("=" * 80)

    # Load the model
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device="cpu")
    s3gen = model.s3gen
    
    # Load inputs from Step 7
    mu = torch.from_numpy(np.load(OUTPUT_DIR / "step6_mu.npy"))  # [1, T, 80]
    x_cond = torch.from_numpy(np.load(OUTPUT_DIR / "step6_x_cond.npy"))
    initial_noise = torch.from_numpy(np.load(OUTPUT_DIR / "step7_initial_noise.npy"))
    spk_cond = torch.from_numpy(np.load(OUTPUT_DIR / "step5_spk_emb.npy"))  # [1, 80] - already projected
    prompt_feat = torch.from_numpy(np.load(OUTPUT_DIR / "step4_prompt_feat.npy"))

    print(f"mu: {mu.shape}")
    print(f"initial_noise: {initial_noise.shape}")
    print(f"spk_cond: {spk_cond.shape}")

    # Prepare inputs
    mu_t = mu.transpose(1, 2)  # [1, 80, T]
    conds_t = x_cond.transpose(1, 2)  # [1, 80, T]
    xt = initial_noise.clone()

    # Create mask (all ones for full attention)
    T = xt.shape[2]
    mask = torch.ones([1, 1, T], dtype=mu_t.dtype)
    
    # ODE parameters
    n_steps = 10
    cfg_rate = 0.7
    
    # Generate timesteps with cosine scheduler
    t_span_linear = torch.linspace(0, 1, n_steps + 1)
    t_span = 1 - torch.cos(t_span_linear * 0.5 * torch.pi)
    
    print(f"\nRunning ODE loop with {n_steps} steps...")
    print(f"Timesteps: {t_span.numpy()}")
    
    trajectory = []
    trajectory.append(xt.clone())
    
    with torch.no_grad():
        for step in range(1, n_steps + 1):
            t_curr = t_span[step - 1]
            dt = t_span[step] - t_span[step - 1]
            
            t_tensor = torch.tensor([t_curr], dtype=mu_t.dtype)
            
            # CFG: duplicate batch
            x_in = torch.cat([xt, xt], dim=0)
            mu_in = torch.cat([mu_t, torch.zeros_like(mu_t)], dim=0)
            spk_in = torch.cat([spk_cond, torch.zeros_like(spk_cond)], dim=0)
            cond_in = torch.cat([conds_t, torch.zeros_like(conds_t)], dim=0)
            t_in = torch.cat([t_tensor, t_tensor], dim=0)
            mask_in = torch.cat([mask, mask], dim=0)

            # Call decoder (estimator uses positional args)
            v_batch = s3gen.flow.decoder.estimator(x_in, mask_in, mu_in, t_in, spk_in, cond_in)
            
            v_cond = v_batch[0:1]
            v_uncond = v_batch[1:2]
            
            # CFG blending
            v = (1.0 + cfg_rate) * v_cond - cfg_rate * v_uncond

            # Save intermediate values for first step
            if step == 1:
                save_c_order(OUTPUT_DIR / "ode_step1_v_cond.npy", v_cond.numpy())
                save_c_order(OUTPUT_DIR / "ode_step1_v_uncond.npy", v_uncond.numpy())
                save_c_order(OUTPUT_DIR / "ode_step1_v_mix.npy", v.numpy())
                print(f"  [DEBUG] v_cond range: [{v_cond.min().item():.4f}, {v_cond.max().item():.4f}]")
                print(f"  [DEBUG] v_uncond range: [{v_uncond.min().item():.4f}, {v_uncond.max().item():.4f}]")
                print(f"  [DEBUG] v_mix range: [{v.min().item():.4f}, {v.max().item():.4f}]")

            # Euler step
            xt = xt + dt * v

            trajectory.append(xt.clone())

            print(f"  Step {step}/{n_steps}: t={t_curr:.6f}, dt={dt:.6f}, x range=[{xt.min().item():.4f}, {xt.max().item():.4f}]")
    
    print("\n" + "=" * 80)
    print("SAVING TRAJECTORY")
    print("=" * 80)
    
    # Save each step
    for i, x_step in enumerate(trajectory):
        save_c_order(OUTPUT_DIR / f"ode_traj_step{i}.npy", x_step.numpy())
        print(f"  step{i}.npy: {x_step.shape}, range=[{x_step.min().item():.4f}, {x_step.max().item():.4f}]")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
