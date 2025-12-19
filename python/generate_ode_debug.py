#!/usr/bin/env python3
"""
Export ODE solver debug values for first step verification.
This isolates the ODE integration math from the neural network.
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
    print("ODE SOLVER DEBUG: First Step Verification")
    print("=" * 80)

    # Load existing verification outputs from Step 7a
    velocity_t0 = torch.from_numpy(np.load(OUTPUT_DIR / "step7a_d_output.npy"))  # [B, C, T]
    initial_noise = torch.from_numpy(np.load(OUTPUT_DIR / "step7a_d_h_input.npy"))[:, :80, :]  # [B, 80, T]

    print(f"velocity_t0: {list(velocity_t0.shape)}, range [{velocity_t0.min().item():.4f}, {velocity_t0.max().item():.4f}]")
    print(f"initial_noise: {list(initial_noise.shape)}, range [{initial_noise.min().item():.4f}, {initial_noise.max().item():.4f}]")

    # ODE Solver Parameters (must match your actual solver)
    n_steps = 10

    # 1. Generate timesteps
    # Chatterbox uses cosine scheduler by default
    timesteps_linear = torch.linspace(0, 1, n_steps + 1)
    timesteps = 1 - torch.cos(timesteps_linear * 0.5 * torch.pi)
    print(f"\nTimesteps (with cosine scheduler): {timesteps.numpy()}")
    print(f"  shape: {timesteps.shape}")
    print(f"  first 3: {timesteps[:3].numpy()}")

    # 2. Calculate dt for first step
    t_curr = timesteps[0]  # 0.0
    t_next = timesteps[1]  # 0.1
    dt = t_next - t_curr
    print(f"\nStep 0:")
    print(f"  t_curr: {t_curr.item():.6f}")
    print(f"  t_next: {t_next.item():.6f}")
    print(f"  dt: {dt.item():.6f}")

    # 3. Perform Euler update for first step
    # Standard Euler: x_next = x_curr + v * dt
    with torch.no_grad():
        x_curr = initial_noise
        v_pred = velocity_t0
        x_next_step0 = x_curr + v_pred * dt

        print(f"\nEuler Update:")
        print(f"  x_curr: {list(x_curr.shape)}, range [{x_curr.min().item():.4f}, {x_curr.max().item():.4f}]")
        print(f"  v_pred: {list(v_pred.shape)}, range [{v_pred.min().item():.4f}, {v_pred.max().item():.4f}]")
        print(f"  v*dt: range [{(v_pred * dt).min().item():.4f}, {(v_pred * dt).max().item():.4f}]")
        print(f"  x_next: {list(x_next_step0.shape)}, range [{x_next_step0.min().item():.4f}, {x_next_step0.max().item():.4f}]")

    # 4. Export for Swift comparison
    save_c_order(OUTPUT_DIR / "ode_timesteps.npy", timesteps.numpy())
    save_c_order(OUTPUT_DIR / "ode_dt_step0.npy", np.array([dt.item()]))
    save_c_order(OUTPUT_DIR / "ode_x_next_step0.npy", x_next_step0.numpy())

    print("\n" + "=" * 80)
    print("SAVED FILES:")
    print(f"  ode_timesteps.npy: {timesteps.shape}")
    print(f"  ode_dt_step0.npy: scalar value {dt.item():.6f}")
    print(f"  ode_x_next_step0.npy: {x_next_step0.shape}")
    print("=" * 80)


if __name__ == "__main__":
    main()
