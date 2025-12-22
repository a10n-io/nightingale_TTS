#!/usr/bin/env python3
"""Trace decoder down block step by step to find divergence."""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path.home() / "Library/Python/3.9/lib/python/site-packages"))

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

print("="*80)
print("DECODER DOWN BLOCK TRACE")
print("="*80)

# Load decoder reference data
x_batched = torch.from_numpy(np.load(ref_dir / "step7_step1_x_before.npy"))  # [2, 80, 696]
mu_batched = torch.from_numpy(np.load(ref_dir / "step7_mu_T.npy"))           # [2, 80, 696]
cond = torch.from_numpy(np.load(ref_dir / "step6_x_cond.npy"))               # [1, 80, 696]
spk_batched = torch.from_numpy(np.load(ref_dir / "step7_spk_emb.npy"))       # [2, 80]
mask_batched = torch.from_numpy(np.load(ref_dir / "step7_cond_T.npy"))       # [2, 1, 696]

# Extract conditional pass (index 0)
x = x_batched[[0]]      # [1, 80, 696]
mu = mu_batched[[0]]    # [1, 80, 696]
spk = spk_batched[[0]]  # [1, 80]
mask = mask_batched[[0]]  # [1, 1, 696]

print(f"\n📥 Loaded inputs:")
print(f"  x shape: {x.shape}")
print(f"  mu shape: {mu.shape}")
print(f"  cond shape: {cond.shape}")
print(f"  spk shape: {spk.shape}")
print(f"  mask shape: {mask.shape}, sum={mask.sum().item()}")

# Load state dict
model_path = PROJECT_ROOT / "models" / "chatterbox" / "s3gen.pt"
state_dict = torch.load(str(model_path), map_location='cpu')

def get_weight(key):
    full_key = f'flow.decoder.estimator.{key}'
    if full_key in state_dict:
        return state_dict[full_key]
    raise KeyError(f"Key not found: {full_key}")

# Step 1: Concatenate inputs
print(f"\n{'='*80}")
print("STEP 1: Concatenate h")
print("="*80)
spk_expanded = spk.unsqueeze(-1).expand(-1, -1, x.shape[2])  # [1, 80, 696]
h = torch.cat([x, mu, spk_expanded, cond], dim=1)  # [1, 320, 696]
print(f"  h shape: {h.shape}")
print(f"  h range: [{h.min():.6f}, {h.max():.6f}]")
print(f"  h mean: {h.mean():.6f}")
np.save(ref_dir / "dec_down_step1_h_concat.npy", h.numpy())

# Step 2: ResNet block (down_blocks.0.0)
print(f"\n{'='*80}")
print("STEP 2: ResNet Block")
print("="*80)

# Load ResNet weights
conv1_w = get_weight('down_blocks.0.0.block1.block.0.weight')  # [256, 320, 3]
conv1_b = get_weight('down_blocks.0.0.block1.block.0.bias')    # [256]
norm1_w = get_weight('down_blocks.0.0.block1.block.2.weight')  # [256]
norm1_b = get_weight('down_blocks.0.0.block1.block.2.bias')    # [256]

print(f"  conv1 weight shape: {conv1_w.shape}")
print(f"  conv1 bias shape: {conv1_b.shape}")

# Apply first conv (with masking)
import torch.nn.functional as F

# Pad for causal conv (kernel=3, pad left by 2)
h_padded = F.pad(h, (2, 0))  # Pad 2 on left
print(f"  After padding: {h_padded.shape}")

# Conv1d
h_conv1 = F.conv1d(h_padded, conv1_w, conv1_b)
print(f"  After conv1: shape={h_conv1.shape}, range=[{h_conv1.min():.6f}, {h_conv1.max():.6f}]")
np.save(ref_dir / "dec_down_step2_after_conv1.npy", h_conv1.numpy())

# Norm1 (GroupNorm with 8 groups)
h_norm1 = F.group_norm(h_conv1, num_groups=8, weight=norm1_w, bias=norm1_b, eps=1e-5)
print(f"  After norm1: shape={h_norm1.shape}, range=[{h_norm1.min():.6f}, {h_norm1.max():.6f}]")
np.save(ref_dir / "dec_down_step2_after_norm1.npy", h_norm1.numpy())

# Mish activation
h_mish = h_norm1 * torch.tanh(F.softplus(h_norm1))
print(f"  After mish: shape={h_mish.shape}, range=[{h_mish.min():.6f}, {h_mish.max():.6f}]")
np.save(ref_dir / "dec_down_step2_after_mish.npy", h_mish.numpy())

# Load time embedding (should be computed from t=0.999)
time_emb = torch.from_numpy(np.load(ref_dir / "dec_trace_time_emb.npy"))
print(f"\n  Time embedding shape: {time_emb.shape}, range=[{time_emb.min():.6f}, {time_emb.max():.6f}]")

# Time MLP projection
mlp1_w = get_weight('down_blocks.0.0.mlp.1.weight')  # [256, 1024]
mlp1_b = get_weight('down_blocks.0.0.mlp.1.bias')    # [256]
t_emb = F.linear(F.mish(time_emb), mlp1_w, mlp1_b)
t_emb = t_emb.unsqueeze(-1)  # [1, 256, 1]
print(f"  After time mlp: shape={t_emb.shape}, range=[{t_emb.min():.6f}, {t_emb.max():.6f}]")
np.save(ref_dir / "dec_down_step2_t_emb_proj.npy", t_emb.numpy())

# Add time embedding
h_with_time = h_mish + t_emb
print(f"  After adding time: shape={h_with_time.shape}, range=[{h_with_time.min():.6f}, {h_with_time.max():.6f}]")
np.save(ref_dir / "dec_down_step2_with_time.npy", h_with_time.numpy())

# Second conv
conv2_w = get_weight('down_blocks.0.0.block2.block.0.weight')  # [256, 256, 3]
conv2_b = get_weight('down_blocks.0.0.block2.block.0.bias')    # [256]
norm2_w = get_weight('down_blocks.0.0.block2.block.2.weight')  # [256]
norm2_b = get_weight('down_blocks.0.0.block2.block.2.bias')    # [256]

h_padded2 = F.pad(h_with_time, (2, 0))
h_conv2 = F.conv1d(h_padded2, conv2_w, conv2_b)
h_norm2 = F.group_norm(h_conv2, num_groups=8, weight=norm2_w, bias=norm2_b, eps=1e-5)
h_mish2 = h_norm2 * torch.tanh(F.softplus(h_norm2))
print(f"  After block2: shape={h_mish2.shape}, range=[{h_mish2.min():.6f}, {h_mish2.max():.6f}]")
np.save(ref_dir / "dec_down_step2_after_block2.npy", h_mish2.numpy())

# Apply mask before residual
if mask is not None:
    h_masked = h_mish2 * mask
    print(f"  After masking: shape={h_masked.shape}, range=[{h_masked.min():.6f}, {h_masked.max():.6f}]")
    np.save(ref_dir / "dec_down_step2_masked.npy", h_masked.numpy())
else:
    h_masked = h_mish2

# Residual connection
res_conv_w = get_weight('down_blocks.0.0.res_conv.weight')  # [256, 320, 1]
res_conv_b = get_weight('down_blocks.0.0.res_conv.bias')    # [256]
h_res = F.conv1d(h, res_conv_w, res_conv_b)
print(f"  Residual conv: shape={h_res.shape}, range=[{h_res.min():.6f}, {h_res.max():.6f}]")
np.save(ref_dir / "dec_down_step2_res_conv.npy", h_res.numpy())

h_out = h_masked + h_res
print(f"  After residual: shape={h_out.shape}, range=[{h_out.min():.6f}, {h_out.max():.6f}]")
print(f"  Mean: {h_out.mean():.6f}")
np.save(ref_dir / "dec_down_step2_resnet_out.npy", h_out.numpy())

# Step 3: Transformers (4 of them)
print(f"\n{'='*80}")
print("STEP 3: Transformers")
print("="*80)

# Transpose for transformers: [B, C, T] -> [B, T, C]
h_tfmr = h_out.transpose(1, 2)  # [1, 696, 256]
print(f"  Input to transformers: shape={h_tfmr.shape}, range=[{h_tfmr.min():.6f}, {h_tfmr.max():.6f}]")
np.save(ref_dir / "dec_down_step3_tfmr_input.npy", h_tfmr.numpy())

# Create attention mask (just for the valid tokens)
mask_2d = mask.squeeze(1)  # [1, 696]
print(f"  Mask shape: {mask_2d.shape}, sum={mask_2d.sum().item()}")

# For now, just note where transformers would run
print(f"  Would run 4 transformers here...")
print(f"  (Transformer parity already verified separately)")

print(f"\n{'='*80}")
print("✅ Saved all down block intermediate steps")
print("="*80)
