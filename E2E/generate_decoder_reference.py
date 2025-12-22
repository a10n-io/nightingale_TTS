#!/usr/bin/env python3
"""Generate Python reference outputs for encoder and decoder steps.

This script generates step6 (encoder) and step7 (decoder ODE) outputs
for comparison with Swift implementation.
"""

import sys
sys.path.insert(0, '/Users/a10n/Projects/nightingale_TTS/python')

import torch
import numpy as np
from pathlib import Path

# Set deterministic
torch.manual_seed(42)
np.random.seed(42)
if hasattr(torch.mps, 'manual_seed'):
    torch.mps.manual_seed(42)

# Paths
PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
REF_DIR = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

print("="*80)
print("GENERATE PYTHON DECODER REFERENCE")
print("="*80)

# Load chatterbox model
print("\nLoading model...")
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals

model_dir = PROJECT_ROOT / "models" / "chatterbox"
device = "cpu"

model = ChatterboxMultilingualTTS.from_local(str(model_dir), device=device)

# Load voice
voice_path = PROJECT_ROOT / "baked_voices" / "samantha" / "baked_voice.pt"
model.conds = Conditionals.load(str(voice_path), map_location=device)
print(f"  Model loaded: {device}")

# Load step5 outputs (input to encoder)
print("\nLoading step5 outputs...")
token_emb = torch.from_numpy(np.load(REF_DIR / "step5_token_emb.npy")).to(device)
spk_emb = torch.from_numpy(np.load(REF_DIR / "step5_spk_emb.npy")).to(device)
mask = torch.from_numpy(np.load(REF_DIR / "step5_mask.npy")).to(device)
prompt_feat = torch.from_numpy(np.load(REF_DIR / "step4_s3_prompt_feat.npy")).to(device)
prompt_token_len = int(np.load(REF_DIR / "step5_prompt_token_len.npy"))
speech_token_len = int(np.load(REF_DIR / "step5_speech_token_len.npy"))

print(f"  token_emb: {token_emb.shape}")
print(f"  spk_emb: {spk_emb.shape}")
print(f"  mask: {mask.shape}")
print(f"  prompt_feat: {prompt_feat.shape}")
print(f"  prompt_token_len: {prompt_token_len}")
print(f"  speech_token_len: {speech_token_len}")

# =========================================================================
# Step 6: Encoder
# =========================================================================
print("\n" + "="*80)
print("STEP 6: ENCODER")
print("="*80)

mask_float = mask.unsqueeze(-1)
token_emb_masked = token_emb * mask_float
total_len = prompt_token_len + speech_token_len
token_len = torch.tensor([total_len], dtype=torch.long, device=device)

print(f"\nRunning encoder...")
print(f"  Input: {token_emb_masked.shape}")
print(f"  Token length: {total_len}")

with torch.no_grad():
    encoder_out, encoder_masks = model.s3gen.flow.encoder(token_emb_masked, token_len)

    print(f"  Encoder output: {encoder_out.shape}")
    print(f"  Range: [{encoder_out.min().item():.4f}, {encoder_out.max().item():.4f}]")

    # Save encoder output
    np.save(REF_DIR / "step6_encoder_out.npy", encoder_out.cpu().numpy())
    print(f"\n✅ Saved: step6_encoder_out.npy")

    # Project encoder output to mel conditioning (mu)
    encoder_proj = model.s3gen.flow.encoder_proj(encoder_out)
    mu = encoder_proj.transpose(1, 2).contiguous()  # [B, 80, T]

print(f"\nEncoder projection (mu):")
print(f"  Shape: {mu.shape}")
print(f"  Range: [{mu.min().item():.4f}, {mu.max().item():.4f}]")
print(f"  Mean: {mu.mean().item():.4f}")

np.save(REF_DIR / "step6_mu.npy", mu.cpu().numpy())
print(f"✅ Saved: step6_mu.npy")

# Prepare x_cond (prompt + zeros for generated part)
mel_len1 = prompt_feat.shape[1]
mel_len2 = encoder_out.shape[1] - mel_len1
x_cond = torch.zeros([1, mel_len1 + mel_len2, 80], device=device, dtype=encoder_proj.dtype)
x_cond[:, :mel_len1] = prompt_feat
x_cond = x_cond.transpose(1, 2).contiguous()  # [B, 80, T]

print(f"\nConditioning (x_cond):")
print(f"  Shape: {x_cond.shape}")
print(f"  Prompt len: {mel_len1}, Generated len: {mel_len2}")

np.save(REF_DIR / "step6_x_cond.npy", x_cond.cpu().numpy())
print(f"✅ Saved: step6_x_cond.npy")

# =========================================================================
# Step 7: ODE Solver (Decoder)
# =========================================================================
print("\n" + "="*80)
print("STEP 7: ODE SOLVER (DECODER)")
print("="*80)

# ODE parameters
n_timesteps = 10
cfg_weight = 0.5
B, C, T = mu.shape

print(f"\nODE parameters:")
print(f"  Timesteps: {n_timesteps}")
print(f"  CFG weight: {cfg_weight}")
print(f"  Output shape: {mu.shape}")

# Get decoder and its estimator (the actual network that estimates velocity)
decoder = model.s3gen.flow.decoder
estimator = decoder.estimator

# Create mask for decoder
cond_mask = torch.ones(B, 1, T, device=device, dtype=torch.float32)
cond_mask[:, :, :mel_len1] = 0  # Mask out prompt (don't generate)

# Expand for CFG (conditional + unconditional)
cond_T = torch.cat([cond_mask, torch.zeros_like(cond_mask)], dim=0)
mu_T = torch.cat([mu, mu], dim=0)
spk_emb_T = torch.cat([spk_emb, spk_emb], dim=0)
x_cond_T = torch.cat([x_cond, torch.zeros_like(x_cond)], dim=0)

print(f"\nCFG setup:")
print(f"  cond_T: {cond_T.shape}, sum={cond_T.sum().item()}")
print(f"  mu_T: {mu_T.shape}")
print(f"  spk_emb_T: {spk_emb_T.shape}")

# Save these
np.save(REF_DIR / "step7_cond_T.npy", cond_T.cpu().numpy())
np.save(REF_DIR / "step7_mu_T.npy", mu_T.cpu().numpy())
np.save(REF_DIR / "step7_spk_emb.npy", spk_emb_T.cpu().numpy())

# Initial noise
initial_noise = torch.randn(B, C, T, device=device, dtype=mu.dtype)
np.save(REF_DIR / "step7_initial_noise.npy", initial_noise.cpu().numpy())
print(f"\nInitial noise: {initial_noise.shape}, range=[{initial_noise.min().item():.4f}, {initial_noise.max().item():.4f}]")

# Expand for CFG
xt = torch.cat([initial_noise, initial_noise], dim=0)

# Time scheduling (cosine)
import math
t_span = torch.linspace(0, 1, n_timesteps + 1, device=device)
t_span = 1.0 - torch.cos(t_span * 0.5 * math.pi)
np.save(REF_DIR / "step7_t_span.npy", t_span.cpu().numpy())
print(f"Time span: {t_span.cpu().numpy()}")

print(f"\nRunning ODE solver...")
with torch.no_grad():
    for step in range(n_timesteps):
        t = t_span[step]
        dt = t_span[step + 1] - t

        # Save state before this step
        np.save(REF_DIR / f"step7_step{step+1}_t.npy", t.cpu().numpy())
        np.save(REF_DIR / f"step7_step{step+1}_dt.npy", dt.cpu().numpy())
        np.save(REF_DIR / f"step7_step{step+1}_x_before.npy", xt.cpu().numpy())

        # Get velocity
        t_batch = t.unsqueeze(0).expand(xt.shape[0])

        dxdt = estimator(
            x=xt,
            mask=cond_T,
            mu=mu_T,
            t=t_batch,
            spks=spk_emb_T,
            cond=x_cond_T,
        )

        # Split CFG
        dxdt_cond = dxdt[:B]
        dxdt_uncond = dxdt[B:]

        # Save velocities
        np.save(REF_DIR / f"step7_step{step+1}_dxdt_cond.npy", dxdt_cond.cpu().numpy())
        np.save(REF_DIR / f"step7_step{step+1}_dxdt_uncond.npy", dxdt_uncond.cpu().numpy())

        # CFG
        dxdt_cfg = (1.0 + cfg_weight) * dxdt_cond - cfg_weight * dxdt_uncond
        np.save(REF_DIR / f"step7_step{step+1}_dxdt_cfg.npy", dxdt_cfg.cpu().numpy())

        # Integrate
        xt_single = xt[:B] + dxdt_cfg * dt
        xt = torch.cat([xt_single, xt_single], dim=0)

        # Save state after this step
        np.save(REF_DIR / f"step7_step{step+1}_x_after.npy", xt.cpu().numpy())

        print(f"  Step {step+1}/{n_timesteps}: t={t.item():.4f}, dt={dt.item():.4f}, "
              f"x_range=[{xt.min().item():.4f}, {xt.max().item():.4f}]")

# Final mel
mel = xt[:B]
np.save(REF_DIR / "step7_mel.npy", mel.cpu().numpy())
np.save(REF_DIR / "step7_final_mel.npy", mel.cpu().numpy())

print(f"\n✅ Final mel: {mel.shape}")
print(f"   Range: [{mel.min().item():.4f}, {mel.max().item():.4f}]")
print(f"   Mean: {mel.mean().item():.4f}")
print(f"   Negative: {(mel < 0).sum().item()}/{mel.numel()} ({100*(mel < 0).sum().item()/mel.numel():.1f}%)")

print("\n" + "="*80)
print("✅ ALL REFERENCE OUTPUTS SAVED")
print("="*80)
print(f"\nSaved to: {REF_DIR}")
print("\nFiles created:")
print("  - step6_encoder_out.npy")
print("  - step6_mu.npy")
print("  - step6_x_cond.npy")
print("  - step7_*.npy (ODE step outputs)")
print("  - step7_final_mel.npy")
