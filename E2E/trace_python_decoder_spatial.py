#!/usr/bin/env python3
"""Trace Python decoder spatial variation during ODE integration."""
import torch
from pathlib import Path
import safetensors.torch as st
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import numpy as np

# Load model
MODELS_DIR = Path("models/chatterbox")
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading Chatterbox model...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load decoder trace to get all inputs
trace_path = Path("test_audio/python_decoder_trace.safetensors")
trace = st.load_file(trace_path)

mu = trace["mu"].to(device)  # [1, 80, 696]
conds = trace["conds"].to(device)  # [1, 80, 696]
speaker_emb = trace["spk_cond"].to(device)  # [1, 80]

print(f"mu shape: {mu.shape}")
print(f"conds shape: {conds.shape}")
print(f"speaker_emb shape: {speaker_emb.shape}")

# Transpose to [B, T, C] for analysis
mu_transposed = mu.transpose(1, 2)  # [1, 696, 80]
conds_transposed = conds.transpose(1, 2)  # [1, 696, 80]

# Find where conds transitions from non-zero to zero to determine L_pm
# Look at the mean across the feature dimension
conds_means = conds_transposed[0, :, :].abs().mean(dim=1)  # [696]
nonzero_mask = conds_means > 0.01
L_pm = nonzero_mask.sum().item()

print(f"\nL_pm (prompt length): {L_pm}")
print(f"L_total: {mu.shape[2]}")
print(f"L_new (generated length): {mu.shape[2] - L_pm}")

# Verify conditioning
print(f"conds prompt region [:, :, :{L_pm}] mean: {conds[:, :, :L_pm].mean():.6f}")
print(f"conds generated region [:, :, {L_pm}:] mean: {conds[:, :, L_pm:].mean():.6f}")

# Get flow decoder estimator (the actual network)
decoder = model.s3gen.flow.decoder.estimator

# ODE solver parameters (matching Swift)
n_timesteps = 10
cfg_rate = 0.7

# Cosine schedule
timesteps = torch.cos(torch.linspace(0, np.pi / 2, n_timesteps + 1, device=device)) ** 2

# Initialize with noise
L_total = mu.shape[2]  # 696
xt = torch.randn(1, L_total, 80, device=device)  # [1, 696, 80]

print(f"\n{'='*80}")
print("Python ODE Solver - Spatial Variation Trace")
print(f"{'='*80}")

for step in range(1, n_timesteps + 1):
    t_curr = timesteps[step - 1]
    t_next = timesteps[step]
    dt = t_next - t_curr

    # Prepare timestep tensor
    t_batch = t_curr.unsqueeze(0).expand(2)  # [2] for CFG

    # Transpose for decoder: [B, T, C] → [B, C, T]
    x_transposed = xt.transpose(1, 2)  # [1, 80, L_total]
    # mu and conds are already in [B, C, T] format from the trace

    # CFG: concatenate conditional and unconditional
    x_batch = torch.cat([x_transposed, x_transposed], dim=0)  # [2, 80, L_total]
    mu_batch = torch.cat([mu, mu], dim=0)  # [2, 80, L_total]
    conds_batch = torch.cat([conds, torch.zeros_like(conds)], dim=0)  # [2, 80, L_total]
    speaker_batch = speaker_emb.expand(2, -1)  # [2, 80]

    # Decoder forward pass
    # Python decoder signature: forward(x, mask, mu, t, spks=None, cond=None, r=None)
    mask_batch = torch.ones(2, 1, L_total, device=device)  # [2, 1, L_total]
    with torch.no_grad():
        v_batch = decoder(
            x=x_batch,
            mask=mask_batch,
            mu=mu_batch,
            t=t_batch,
            spks=speaker_batch,
            cond=conds_batch
        )  # [2, 80, L_total]

    # Apply CFG
    v_cond = v_batch[0]  # [80, L_total]
    v_uncond = v_batch[1]  # [80, L_total]

    # Check decoder output spatial variation BEFORE CFG at step 1
    if step == 1:
        v_cond_prompt = v_cond[:, :L_pm]
        v_cond_generated = v_cond[:, L_pm:]
        print(f"   DECODER OUTPUT vCond SPATIAL - prompt[:, :{L_pm}]: [{v_cond_prompt.min():.6f}, {v_cond_prompt.max():.6f}], mean={v_cond_prompt.mean():.6f}")
        print(f"   DECODER OUTPUT vCond SPATIAL - generated[:, {L_pm}:]: [{v_cond_generated.min():.6f}, {v_cond_generated.max():.6f}], mean={v_cond_generated.mean():.6f}")

    v = (1.0 + cfg_rate) * v_cond - cfg_rate * v_uncond  # [80, L_total]

    # Transpose back: [C, T] → [T, C]
    v = v.transpose(0, 1)  # [L_total, 80]

    # Check spatial variation at key steps
    if step in [1, 5, 10]:
        v_prompt_region = v[:L_pm, :]
        v_generated_region = v[L_pm:, :]

        print(f"\n--- Python ODE Step {step}/{n_timesteps} ---")
        print(f"   t = {t_curr.item():.6f}, dt = {dt.item():.6f}")
        print(f"   v SPATIAL (step {step}) - prompt[:{L_pm}]: [{v_prompt_region.min():.6f}, {v_prompt_region.max():.6f}], mean={v_prompt_region.mean():.6f}")
        print(f"   v SPATIAL (step {step}) - generated[{L_pm}:]: [{v_generated_region.min():.6f}, {v_generated_region.max():.6f}], mean={v_generated_region.mean():.6f}")

    # Euler integration
    xt = xt + v * dt

    # Check xt spatial variation at key steps
    if step in [1, 5, 10]:
        xt_prompt_region = xt[0, :L_pm, :]
        xt_generated_region = xt[0, L_pm:, :]
        print(f"   xt SPATIAL (after step {step}) - prompt[:{L_pm}]: [{xt_prompt_region.min():.6f}, {xt_prompt_region.max():.6f}], mean={xt_prompt_region.mean():.6f}")
        print(f"   xt SPATIAL (after step {step}) - generated[{L_pm}:]: [{xt_generated_region.min():.6f}, {xt_generated_region.max():.6f}], mean={xt_generated_region.mean():.6f}")

print(f"\n{'='*80}")
print("Final ODE output:")
print(f"   xt shape: {xt.shape}")
print(f"   xt prompt region mean: {xt[0, :L_pm, :].mean():.6f}")
print(f"   xt generated region mean: {xt[0, L_pm:, :].mean():.6f}")
print(f"{'='*80}")
