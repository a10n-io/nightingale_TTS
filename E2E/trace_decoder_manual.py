#!/usr/bin/env python3
"""Manual decoder trace to capture all intermediate values."""

import torch
import numpy as np
from pathlib import Path
from einops import pack, rearrange
import sys

sys.path.insert(0, str(Path.home() / "Library/Python/3.9/lib/python/site-packages"))
from chatterbox.models.s3gen.matcha.decoder import SinusoidalPosEmb, Block1D, ResnetBlock1D

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

print("="*80)
print("MANUAL DECODER TRACE (FIRST DOWN BLOCK ONLY)")
print("="*80)

# Load inputs
x = torch.from_numpy(np.load(ref_dir / "step7_step1_x_before.npy")[[0]])  # [1, 80, 696]
mu = torch.from_numpy(np.load(ref_dir / "step7_mu_T.npy")[[0]])           # [1, 80, 696]
spk_emb = torch.from_numpy(np.load(ref_dir / "step7_spk_emb.npy")[[0]])   # [1, 80]
x_cond = torch.from_numpy(np.load(ref_dir / "step6_x_cond.npy"))          # [1, 80, 696]
mask = torch.from_numpy(np.load(ref_dir / "step7_cond_T.npy")[[0]])       # [1, 1, 696]
t = torch.from_numpy(np.load(ref_dir / "step7_step1_t.npy"))              # []

print(f"\n📥 Inputs:")
print(f"  x: {x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")
print(f"  mu: {mu.shape}, range=[{mu.min():.4f}, {mu.max():.4f}]")
print(f"  mask: {mask.shape}, sum={mask.sum().item()}")

# Load state dict
model_path = PROJECT_ROOT / "models" / "chatterbox" / "s3gen.pt"
state_dict = torch.load(str(model_path), map_location='cpu')

def get_weight(key):
    return state_dict[f'flow.decoder.estimator.{key}']

# 1. Time embedding
print("\n⏱ Time Embedding:")
time_emb_func = SinusoidalPosEmb(320)
time_emb_raw = time_emb_func(t)
print(f"  time_emb_raw: {time_emb_raw.shape}, range=[{time_emb_raw.min():.4f}, {time_emb_raw.max():.4f}]")
np.save(ref_dir / "dec_trace_time_emb_raw.npy", time_emb_raw.detach().numpy())

w1 = get_weight('time_mlp.linear_1.weight')
b1 = get_weight('time_mlp.linear_1.bias')
w2 = get_weight('time_mlp.linear_2.weight')
b2 = get_weight('time_mlp.linear_2.bias')

t_emb = torch.nn.functional.linear(time_emb_raw, w1, b1)
t_emb = torch.nn.functional.silu(t_emb)
t_emb = torch.nn.functional.linear(t_emb, w2, b2)
print(f"  t_emb: {t_emb.shape}, range=[{t_emb.min():.4f}, {t_emb.max():.4f}]")
np.save(ref_dir / "dec_trace_time_emb.npy", t_emb.detach().numpy())

# 2. Concatenate inputs
print("\n🔗 Concatenation:")
spks_expanded = spk_emb.unsqueeze(-1).expand(-1, -1, x.shape[-1])
h = pack([x, mu, spks_expanded, x_cond], "b * t")[0]
print(f"  h_concat: {h.shape}, range=[{h.min():.4f}, {h.max():.4f}]")
np.save(ref_dir / "dec_trace_h_concat.npy", h.detach().numpy())

# 3. Down block 0 ResNet
print("\n🔽 Down Block 0 ResNet:")

# Create ResNet block and load weights
resnet = ResnetBlock1D(dim=320, dim_out=256, time_emb_dim=1024, groups=8)

# Load block1 weights
prefix = 'down_blocks.0.0'
block1_conv_w = get_weight(f'{prefix}.block1.block.0.weight')
block1_conv_b = get_weight(f'{prefix}.block1.block.0.bias')
block1_norm_w = get_weight(f'{prefix}.block1.block.2.weight')  # Index 2 not 1!
block1_norm_b = get_weight(f'{prefix}.block1.block.2.bias')

resnet.block1.block[0].weight.data = block1_conv_w
resnet.block1.block[0].bias.data = block1_conv_b
resnet.block1.block[1].weight.data = block1_norm_w
resnet.block1.block[1].bias.data = block1_norm_b

# Load mlp weights
mlp_w = get_weight(f'{prefix}.mlp.1.weight')
mlp_b = get_weight(f'{prefix}.mlp.1.bias')
resnet.mlp[1].weight.data = mlp_w
resnet.mlp[1].bias.data = mlp_b

# Load block2 weights
block2_conv_w = get_weight(f'{prefix}.block2.block.0.weight')
block2_conv_b = get_weight(f'{prefix}.block2.block.0.bias')
block2_norm_w = get_weight(f'{prefix}.block2.block.2.weight')  # Index 2 not 1!
block2_norm_b = get_weight(f'{prefix}.block2.block.2.bias')

resnet.block2.block[0].weight.data = block2_conv_w
resnet.block2.block[0].bias.data = block2_conv_b
resnet.block2.block[1].weight.data = block2_norm_w
resnet.block2.block[1].bias.data = block2_norm_b

# Load res_conv weights
res_conv_w = get_weight(f'{prefix}.res_conv.weight')
res_conv_b = get_weight(f'{prefix}.res_conv.bias')
resnet.res_conv.weight.data = res_conv_w
resnet.res_conv.bias.data = res_conv_b

# Forward pass with tracing
resnet.eval()
with torch.no_grad():
    # Block1
    h_block1_input = h * mask
    print(f"  block1 input (h*mask): range=[{h_block1_input.min():.4f}, {h_block1_input.max():.4f}]")
    np.save(ref_dir / "dec_trace_down0_block1_input.npy", h_block1_input.detach().numpy())

    h_block1 = resnet.block1(h, mask)
    print(f"  after block1: range=[{h_block1.min():.4f}, {h_block1.max():.4f}]")
    np.save(ref_dir / "dec_trace_down0_after_block1.npy", h_block1.detach().numpy())

    # MLP
    t_mlp = resnet.mlp(t_emb)
    print(f"  mlp output: range=[{t_mlp.min():.4f}, {t_mlp.max():.4f}]")
    np.save(ref_dir / "dec_trace_down0_mlp_out.npy", t_mlp.detach().numpy())

    # Add time
    h_with_time = h_block1 + t_mlp.unsqueeze(-1)
    print(f"  after adding time: range=[{h_with_time.min():.4f}, {h_with_time.max():.4f}]")
    np.save(ref_dir / "dec_trace_down0_with_time.npy", h_with_time.detach().numpy())

    # Block2
    h_block2 = resnet.block2(h_with_time, mask)
    print(f"  after block2: range=[{h_block2.min():.4f}, {h_block2.max():.4f}]")
    np.save(ref_dir / "dec_trace_down0_after_block2.npy", h_block2.detach().numpy())

    # Res conv
    h_res_input = h * mask
    print(f"  res_conv input (h*mask): range=[{h_res_input.min():.4f}, {h_res_input.max():.4f}]")
    np.save(ref_dir / "dec_trace_down0_res_input.npy", h_res_input.detach().numpy())

    h_res = resnet.res_conv(h * mask)
    print(f"  after res_conv: range=[{h_res.min():.4f}, {h_res.max():.4f}]")
    np.save(ref_dir / "dec_trace_down0_res_conv.npy", h_res.detach().numpy())

    # Final
    h_resnet = h_block2 + h_res
    print(f"  final resnet output: range=[{h_resnet.min():.4f}, {h_resnet.max():.4f}]")
    np.save(ref_dir / "dec_trace_down0_resnet_out.npy", h_resnet.detach().numpy())

print("\n✅ Saved trace files")
print("="*80)
