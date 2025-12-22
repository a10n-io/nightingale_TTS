#!/usr/bin/env python3
"""Simple trace of Python flow decoder - just load saved tensors and trace first ODE step."""

import numpy as np
import torch

print("Loading saved reference tensors from Python run...")
ref_dir = '/Users/a10n/Projects/nightingale_TTS/E2E/reference_outputs/samantha/expressive_surprise_en'

# Load initial ODE state and first step outputs
try:
    noise = np.load(f'{ref_dir}/step7_initial_noise.npy')
    step1_x_before = np.load(f'{ref_dir}/step7_step1_x_before.npy')
    step1_t = np.load(f'{ref_dir}/step7_step1_t.npy')
    step1_dxdt_cond = np.load(f'{ref_dir}/step7_step1_dxdt_cond.npy')
    step1_dxdt_uncond = np.load(f'{ref_dir}/step7_step1_dxdt_uncond.npy')
    step1_dxdt_cfg = np.load(f'{ref_dir}/step7_step1_dxdt_cfg.npy')
    step1_x_after = np.load(f'{ref_dir}/step7_step1_x_after.npy')
    step1_dt = np.load(f'{ref_dir}/step7_step1_dt.npy')

    print("\n✅ Loaded Python reference tensors for ODE step 1:")
    print(f"  noise shape: {noise.shape}, range: [{noise.min():.4f}, {noise.max():.4f}]")
    print(f"  x_before shape: {step1_x_before.shape}, range: [{step1_x_before.min():.4f}, {step1_x_before.max():.4f}]")
    print(f"  t: {step1_t[0]:.6f}")
    print(f"  dt: {step1_dt[0]:.6f}")
    print(f"  dxdt_cond range: [{step1_dxdt_cond.min():.4f}, {step1_dxdt_cond.max():.4f}], mean: {step1_dxdt_cond.mean():.4f}")
    print(f"  dxdt_uncond range: [{step1_dxdt_uncond.min():.4f}, {step1_dxdt_uncond.max():.4f}], mean: {step1_dxdt_uncond.mean():.4f}")
    print(f"  dxdt_cfg range: [{step1_dxdt_cfg.min():.4f}, {step1_dxdt_cfg.max():.4f}], mean: {step1_dxdt_cfg.mean():.4f}")
    print(f"  x_after shape: {step1_x_after.shape}, range: [{step1_x_after.min():.4f}, {step1_x_after.max():.4f}]")

    # Verify the Euler step manually
    cfg_rate = 0.7
    v_computed = (1.0 + cfg_rate) * step1_dxdt_cond - cfg_rate * step1_dxdt_uncond
    print(f"\n  Computed v (should match dxdt_cfg): range=[{v_computed.min():.4f}, {v_computed.max():.4f}]")
    print(f"  Saved dxdt_cfg:                      range=[{step1_dxdt_cfg.min():.4f}, {step1_dxdt_cfg.max():.4f}]")
    print(f"  Match: {np.allclose(v_computed, step1_dxdt_cfg, atol=1e-5)}")

    x_after_computed = step1_x_before + step1_dxdt_cfg * step1_dt[0]
    print(f"\n  Computed x_after (should match saved): range=[{x_after_computed.min():.4f}, {x_after_computed.max():.4f}]")
    print(f"  Saved x_after:                          range=[{step1_x_after.min():.4f}, {step1_x_after.max():.4f}]")
    print(f"  Match: {np.allclose(x_after_computed, step1_x_after, atol=1e-5)}")

    # Save these for Swift comparison
    np.save('/Users/a10n/Projects/nightingale_TTS/E2E/python_ref_step1_x_before.npy', step1_x_before)
    np.save('/Users/a10n/Projects/nightingale_TTS/E2E/python_ref_step1_dxdt_cond.npy', step1_dxdt_cond)
    np.save('/Users/a10n/Projects/nightingale_TTS/E2E/python_ref_step1_dxdt_uncond.npy', step1_dxdt_uncond)
    np.save('/Users/a10n/Projects/nightingale_TTS/E2E/python_ref_step1_dxdt_cfg.npy', step1_dxdt_cfg)
    np.save('/Users/a10n/Projects/nightingale_TTS/E2E/python_ref_step1_x_after.npy', step1_x_after)

    print("\n✅ Saved Python reference tensors for Swift comparison")
    print("   Use these to compare Swift's ODE step 1 outputs")

    # Load final mel for comparison
    final_mel = np.load(f'{ref_dir}/step7_final_mel.npy')
    print(f"\n  Final mel shape: {final_mel.shape}, range: [{final_mel.min():.4f}, {final_mel.max():.4f}]")
    print(f"  Final mel channel energies:")
    for i in [0, 10, 20, 30, 40, 50, 60, 70, 79]:
        # Mel is [B, C, T]
        energy = final_mel[0, i, :].mean()
        print(f"    Channel {i:2d}: {energy:.4f}")

except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("   Make sure Python reference outputs exist in E2E/reference_outputs/")
