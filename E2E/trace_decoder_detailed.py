#!/usr/bin/env python3
"""Detailed decoder trace to find exact divergence point with Swift."""

import torch
import numpy as np
from pathlib import Path
from einops import pack

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

print("="*80)
print("DETAILED PYTHON DECODER TRACE")
print("="*80)

# Load inputs
x = torch.from_numpy(np.load(ref_dir / "step7_step1_x_before.npy")[[0]])  # [1, 80, 696]
mu = torch.from_numpy(np.load(ref_dir / "step7_mu_T.npy")[[0]])           # [1, 80, 696]
spk_emb = torch.from_numpy(np.load(ref_dir / "step7_spk_emb.npy")[[0]])   # [1, 80]
x_cond = torch.from_numpy(np.load(ref_dir / "step6_x_cond.npy"))          # [1, 80, 696]
mask = torch.from_numpy(np.load(ref_dir / "step7_cond_T.npy")[[0]])       # [1, 1, 696]
t = torch.from_numpy(np.load(ref_dir / "step7_step1_t.npy"))              # []

print(f"\n1. INPUTS:")
print(f"  x: {x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")
print(f"  mask sum: {mask.sum()}/{mask.numel()} ones")

# Concatenate
spks_expanded = spk_emb.unsqueeze(-1).expand(-1, -1, x.shape[-1])
h = pack([x, mu, spks_expanded, x_cond], "b * t")[0]
print(f"\n2. CONCATENATION:")
print(f"  h: {h.shape}, range=[{h.min():.4f}, {h.max():.4f}]")
np.save(ref_dir / "detailed_step2_h_concat.npy", h.numpy())

# Load model
model_path = PROJECT_ROOT / "models" / "chatterbox" / "s3gen.pt"
state_dict = torch.load(str(model_path), map_location='cpu')

# Time embedding
from chatterbox.models.s3gen.decoder import SinusoidalPosEmb
time_emb_fn = SinusoidalPosEmb(320)
time_emb = time_emb_fn(t)
print(f"\n3. TIME EMBEDDING (sinusoidal):")
print(f"  time_emb: {time_emb.shape}, range=[{time_emb.min():.4f}, {time_emb.max():.4f}]")
np.save(ref_dir / "detailed_step3_time_emb.npy", time_emb.numpy())

# Time MLP
w1 = state_dict['flow.decoder.estimator.time_mlp.linear_1.weight']
b1 = state_dict['flow.decoder.estimator.time_mlp.linear_1.bias']
w2 = state_dict['flow.decoder.estimator.time_mlp.linear_2.weight']
b2 = state_dict['flow.decoder.estimator.time_mlp.linear_2.bias']

t_emb = torch.nn.functional.linear(time_emb, w1, b1)
t_emb = torch.nn.functional.silu(t_emb)
t_emb = torch.nn.functional.linear(t_emb, w2, b2)
print(f"\n4. TIME MLP:")
print(f"  t_emb: {t_emb.shape}, range=[{t_emb.min():.4f}, {t_emb.max():.4f}]")
print(f"  t_emb[:5]: {t_emb[0, :5].numpy()}")
np.save(ref_dir / "detailed_step4_t_emb.npy", t_emb.numpy())

# Down ResNet Block 1
print(f"\n5. DOWN RESNET BLOCK:")
print(f"  Input h: {h.shape}, range=[{h.min():.4f}, {h.max():.4f}]")

# Block1 conv (with mask)
from chatterbox.models.s3gen.decoder import CausalConv1d
conv1 = CausalConv1d(320, 256, kernel_size=3)
conv1.weight.data = state_dict['flow.decoder.estimator.down_blocks.0.0.block1.block.0.weight']
conv1.bias.data = state_dict['flow.decoder.estimator.down_blocks.0.0.block1.block.0.bias']

h_masked = h * mask
print(f"  h * mask: range=[{h_masked.min():.4f}, {h_masked.max():.4f}]")
np.save(ref_dir / "detailed_step5a_h_masked.npy", h_masked.numpy())

h1 = conv1(h_masked)
print(f"  After block1 conv: {h1.shape}, range=[{h1.min():.4f}, {h1.max():.4f}]")
np.save(ref_dir / "detailed_step5b_after_conv1.npy", h1.detach().numpy())

# LayerNorm
from torch import nn
norm1 = nn.LayerNorm(256)
norm1.weight.data = state_dict['flow.decoder.estimator.down_blocks.0.0.block1.block.2.weight']
norm1.bias.data = state_dict['flow.decoder.estimator.down_blocks.0.0.block1.block.2.bias']

# Transpose for norm
h1_t = h1.transpose(1, 2)  # [B, T, C]
h1_norm = norm1(h1_t)
h1_norm = h1_norm.transpose(1, 2)  # Back to [B, C, T]
print(f"  After block1 norm: range=[{h1_norm.min():.4f}, {h1_norm.max():.4f}]")
np.save(ref_dir / "detailed_step5c_after_norm1.npy", h1_norm.detach().numpy())

# Mish
h1_mish = nn.functional.mish(h1_norm)
print(f"  After block1 mish: range=[{h1_mish.min():.4f}, {h1_mish.max():.4f}]")

# Apply mask again
h1_out = h1_mish * mask
print(f"  After block1 output mask: range=[{h1_out.min():.4f}, {h1_out.max():.4f}]")
np.save(ref_dir / "detailed_step5d_block1_output.npy", h1_out.detach().numpy())

print("\n" + "="*80)
print("✅ DETAILED TRACE COMPLETE")
print("   Saved intermediate values to reference_outputs/")
print("="*80)
