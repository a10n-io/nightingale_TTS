#!/usr/bin/env python3
"""Trace Python decoder layer-by-layer to create reference outputs for Swift verification."""

import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

print("="*80)
print("PYTHON DECODER LAYER-BY-LAYER TRACE")
print("="*80)

# Load decoder weights
model_path = PROJECT_ROOT / "models" / "chatterbox" / "s3gen.pt"
state_dict = torch.load(str(model_path), map_location='cpu')

# Create decoder
from torch import nn

# Load inputs
x = torch.from_numpy(np.load(ref_dir / "step7_step1_x_before.npy")[[0]])  # [1, 80, 696]
mu = torch.from_numpy(np.load(ref_dir / "step7_mu_T.npy")[[0]])           # [1, 80, 696]
spk_emb = torch.from_numpy(np.load(ref_dir / "step7_spk_emb.npy")[[0]])   # [1, 80]
x_cond = torch.from_numpy(np.load(ref_dir / "step6_x_cond.npy"))          # [1, 80, 696]
mask = torch.from_numpy(np.load(ref_dir / "step7_cond_T.npy")[[0]])       # [1, 1, 696]
t = torch.from_numpy(np.load(ref_dir / "step7_step1_t.npy"))              # []

print(f"\nInputs loaded:")
print(f"  x: {x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")
print(f"  mu: {mu.shape}, range=[{mu.min():.4f}, {mu.max():.4f}]")
print(f"  spk_emb: {spk_emb.shape}, range=[{spk_emb.min():.4f}, {spk_emb.max():.4f}]")
print(f"  x_cond: {x_cond.shape}, range=[{x_cond.min():.4f}, {x_cond.max():.4f}]")
print(f"  mask: {mask.shape}, sum={mask.sum().item()}/{mask.numel()} ones")
print(f"  t: {t.shape}, value={t.item():.6f}")

# Check concatenation
from einops import pack
h = pack([x, mu], "b * t")[0]
print(f"\nAfter [x, mu] concat: {h.shape}")

spks_expanded = spk_emb.unsqueeze(-1).expand(-1, -1, h.shape[-1])
h = pack([h, spks_expanded], "b * t")[0]
print(f"After [h, spks] concat: {h.shape}")

h = pack([h, x_cond], "b * t")[0]
print(f"After [h, x_cond] concat: {h.shape}, range=[{h.min():.4f}, {h.max():.4f}]")

# Save concatenated input
np.save(ref_dir / "decoder_h_concat.npy", h.numpy())
print(f"✅ Saved: decoder_h_concat.npy")

# Check time embedding
print(f"\nTime embedding:")
# Time embeddings use sinusoidal position encoding
dim = 320
half_dim = dim // 2
emb = np.log(10000) / (half_dim - 1)
emb = torch.exp(torch.arange(half_dim) * -emb)
# t is scalar, need to reshape for broadcasting
t_batch = t.view(1) if t.dim() == 0 else t  # [1]
emb = t_batch.unsqueeze(1) * emb.unsqueeze(0)
time_emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [1, 320]
print(f"  time_emb: {time_emb.shape}, range=[{time_emb.min():.4f}, {time_emb.max():.4f}]")
print(f"  time_emb first 10 values: {time_emb[0, :10].numpy()}")
print(f"  time_emb last 10 values: {time_emb[0, -10:].numpy()}")

# Apply time MLP
# Load time MLP weights
time_mlp_w1 = state_dict['flow.decoder.estimator.time_mlp.linear_1.weight']  # [1024, 320]
time_mlp_b1 = state_dict['flow.decoder.estimator.time_mlp.linear_1.bias']    # [1024]
time_mlp_w2 = state_dict['flow.decoder.estimator.time_mlp.linear_2.weight']  # [1024, 1024]
time_mlp_b2 = state_dict['flow.decoder.estimator.time_mlp.linear_2.bias']    # [1024]

t_emb = torch.nn.functional.linear(time_emb, time_mlp_w1, time_mlp_b1)
t_emb = torch.nn.functional.silu(t_emb)
t_emb = torch.nn.functional.linear(t_emb, time_mlp_w2, time_mlp_b2)
print(f"  t_emb after MLP: {t_emb.shape}, range=[{t_emb.min():.4f}, {t_emb.max():.4f}]")

np.save(ref_dir / "decoder_t_emb.npy", t_emb.numpy())
print(f"✅ Saved: decoder_t_emb.npy")

print("\n" + "="*80)
print("✅ LAYER-BY-LAYER TRACE COMPLETE")
print("="*80)
