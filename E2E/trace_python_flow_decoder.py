#!/usr/bin/env python3
"""Trace Python flow decoder internals to compare with Swift."""

import numpy as np
import torch
import json

# Load the full S3Gen model
from chatterbox.models.s3gen.s3gen import S3Token2Wav

print("Loading Python S3Gen model...")
s3gen = S3Token2Wav.from_pretrained('/Users/a10n/Projects/nightingale_TTS/models/exp/s3gen')
s3gen.eval()

# Load inputs from saved reference
print("\nLoading reference inputs...")
ref_dir = '/Users/a10n/Projects/nightingale_TTS/E2E/reference_outputs/samantha/expressive_surprise_en'

# These are the inputs to the flow decoder
mu = np.load(f'{ref_dir}/step6_mu.npy')  # Encoder output
cond = np.load(f'{ref_dir}/step6_x_cond.npy')  # Conditioning
spk_emb = np.load(f'{ref_dir}/step4_s3_prompt_feat.npy')  # Speaker embedding

print(f"mu shape: {mu.shape}, range: [{mu.min():.3f}, {mu.max():.3f}]")
print(f"cond shape: {cond.shape}, range: [{cond.min():.3f}, {cond.max():.3f}]")
print(f"spk_emb shape: {spk_emb.shape}, range: [{spk_emb.min():.3f}, {spk_emb.max():.3f}]")

# Convert to torch
mu_torch = torch.from_numpy(mu).float()
cond_torch = torch.from_numpy(cond).float()
spk_emb_torch = torch.from_numpy(spk_emb).float()

# Transpose for decoder: [B, T, C] -> [B, C, T]
mu_T = mu_torch.transpose(1, 2)
cond_T = cond_torch.transpose(1, 2)

print(f"\nAfter transpose:")
print(f"mu_T shape: {mu_T.shape}")
print(f"cond_T shape: {cond_T.shape}")

# Load initial noise (should match Swift's fixed noise)
noise = np.load(f'{ref_dir}/step7_initial_noise.npy')
noise_torch = torch.from_numpy(noise).float()
print(f"noise shape: {noise_torch.shape}, range: [{noise_torch.min():.3f}, {noise_torch.max():.3f}]")

# Prepare speaker conditioning
spk_cond = s3gen.spk_embed_affine(spk_emb_torch)
print(f"spk_cond shape: {spk_cond.shape}, range: [{spk_cond.min():.3f}, {spk_cond.max():.3f}]")

# Create mask
L_total = cond_T.shape[2]
mask = torch.ones(1, 1, L_total)

# ODE solver parameters
n_timesteps = 10
cfg_rate = 0.7

# Cosine time scheduling
t_span = []
for i in range(n_timesteps + 1):
    linear_t = i / n_timesteps
    cosine_t = 1.0 - np.cos(linear_t * 0.5 * np.pi)
    t_span.append(cosine_t)

print(f"\nODE timesteps: {t_span}")

# Run ODE solver with detailed logging
xt = noise_torch.clone()
current_t = t_span[0]

print("\n" + "="*80)
print("PYTHON ODE SOLVER TRACE")
print("="*80)

for step in range(1, n_timesteps + 1):
    dt = t_span[step] - current_t
    t_tensor = torch.tensor([current_t]).float()

    print(f"\n--- Step {step}/{n_timesteps} ---")
    print(f"t={current_t:.6f}, dt={dt:.6f}")
    print(f"xt before: shape={xt.shape}, range=[{xt.min():.3f}, {xt.max():.3f}], mean={xt.mean():.3f}")

    # Prepare batched input for CFG
    x_in = torch.cat([xt, xt], dim=0)
    mu_in = torch.cat([mu_T, torch.zeros_like(mu_T)], dim=0)
    spk_in = torch.cat([spk_cond, torch.zeros_like(spk_cond)], dim=0)
    cond_in = torch.cat([cond_T, torch.zeros_like(cond_T)], dim=0)
    t_in = torch.cat([t_tensor, t_tensor], dim=0)
    mask_in = torch.cat([mask, mask], dim=0)

    print(f"Decoder inputs:")
    print(f"  x_in: {x_in.shape}, range=[{x_in.min():.3f}, {x_in.max():.3f}]")
    print(f"  mu_in: {mu_in.shape}, range=[{mu_in.min():.3f}, {mu_in.max():.3f}]")
    print(f"  cond_in: {cond_in.shape}, range=[{cond_in.min():.3f}, {cond_in.max():.3f}]")
    print(f"  spk_in: {spk_in.shape}, range=[{spk_in.min():.3f}, {spk_in.max():.3f}]")

    # Forward through decoder
    with torch.no_grad():
        v_batch = s3gen.decoder(x_in, mu_in, t_in, spk_in, cond_in, mask_in)

    v_cond = v_batch[0:1]
    v_uncond = v_batch[1:2]

    print(f"Decoder outputs:")
    print(f"  v_cond: range=[{v_cond.min():.3f}, {v_cond.max():.3f}], mean={v_cond.mean():.3f}")
    print(f"  v_uncond: range=[{v_uncond.min():.3f}, {v_uncond.max():.3f}], mean={v_uncond.mean():.3f}")

    # CFG: v = (1 + cfg) * v_cond - cfg * v_uncond
    v = (1.0 + cfg_rate) * v_cond - cfg_rate * v_uncond
    print(f"  v (after CFG): range=[{v.min():.3f}, {v.max():.3f}], mean={v.mean():.3f}")

    # Euler step
    xt = xt + v * dt
    print(f"xt after: range=[{xt.min():.3f}, {xt.max():.3f}], mean={xt.mean():.3f}")

    # Save first step details for comparison
    if step == 1:
        np.save('/Users/a10n/Projects/nightingale_TTS/E2E/python_step1_x_before.npy', xt.numpy())
        np.save('/Users/a10n/Projects/nightingale_TTS/E2E/python_step1_v_cond.npy', v_cond.numpy())
        np.save('/Users/a10n/Projects/nightingale_TTS/E2E/python_step1_v_uncond.npy', v_uncond.numpy())
        np.save('/Users/a10n/Projects/nightingale_TTS/E2E/python_step1_v_cfg.npy', v.numpy())
        np.save('/Users/a10n/Projects/nightingale_TTS/E2E/python_step1_x_after.npy', xt.numpy())
        print("  💾 Saved step 1 tensors for Swift comparison")

    current_t = t_span[step]

# Final mel
print("\n" + "="*80)
print("FINAL OUTPUT")
print("="*80)
L_pm = 50  # Prompt length
final_mel = xt[:, :, L_pm:]
print(f"final_mel shape: {final_mel.shape}")
print(f"final_mel range: [{final_mel.min():.3f}, {final_mel.max():.3f}]")
print(f"final_mel mean: {final_mel.mean():.3f}")

# Check frequency gradient
print("\nPython mel channel energies:")
for i in [0, 10, 20, 30, 40, 50, 60, 70, 79]:
    energy = final_mel[0, i, :].mean().item()
    print(f"  Channel {i:2d}: {energy:.4f}")

# Save final mel
np.save('/Users/a10n/Projects/nightingale_TTS/E2E/python_traced_final_mel.npy', final_mel.numpy())
print("\n✅ Saved python_traced_final_mel.npy")
print("✅ Saved step 1 intermediate tensors for comparison")
