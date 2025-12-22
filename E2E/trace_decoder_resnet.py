#!/usr/bin/env python3
"""Trace Python decoder ResNet block to compare with Swift."""

import torch
import numpy as np
from pathlib import Path
import sys

# Add chatterbox to path
sys.path.insert(0, str(Path.home() / "Library/Python/3.9/lib/python/site-packages"))

from chatterbox.models.s3gen.flow_matching import FlowMatchingDecoder
from chatterbox.models.s3gen.config import S3GenConfig

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

print("="*80)
print("PYTHON DECODER RESNET TRACE")
print("="*80)

# Load decoder
config = S3GenConfig()
decoder = FlowMatchingDecoder(config)

# Load weights
model_path = PROJECT_ROOT / "models" / "chatterbox" / "s3gen.pt"
state_dict = torch.load(str(model_path), map_location='cpu')

# Filter decoder weights
decoder_state = {}
for k, v in state_dict.items():
    if k.startswith('flow.decoder'):
        new_key = k.replace('flow.decoder.', '')
        decoder_state[new_key] = v

decoder.load_state_dict(decoder_state, strict=False)
decoder.eval()

# Load inputs
x = torch.from_numpy(np.load(ref_dir / "step7_step1_x_before.npy")[[0]])  # [1, 80, 696]
mu = torch.from_numpy(np.load(ref_dir / "step7_mu_T.npy")[[0]])           # [1, 80, 696]
spk_emb = torch.from_numpy(np.load(ref_dir / "step7_spk_emb.npy")[[0]])   # [1, 80]
x_cond = torch.from_numpy(np.load(ref_dir / "step6_x_cond.npy"))          # [1, 80, 696]
mask = torch.from_numpy(np.load(ref_dir / "step7_cond_T.npy")[[0]])       # [1, 1, 696]
t = torch.from_numpy(np.load(ref_dir / "step7_step1_t.npy"))              # []

print(f"\nInputs:")
print(f"  x: {x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")
print(f"  mu: {mu.shape}, range=[{mu.min():.4f}, {mu.max():.4f}]")
print(f"  spk_emb: {spk_emb.shape}, range=[{spk_emb.min():.4f}, {spk_emb.max():.4f}]")
print(f"  x_cond: {x_cond.shape}, range=[{x_cond.min():.4f}, {x_cond.max():.4f}]")
print(f"  mask: {mask.shape}, sum={mask.sum().item()}/{mask.numel()}")
print(f"  t: {t.shape}, value={t.item():.6f}")

# Monkey-patch to trace intermediate values
original_forward = decoder.forward

def traced_forward(x, mu, t, spks, cond, mask=None):
    # Concatenate inputs
    from einops import pack

    # Expand speaker embedding
    spks_expanded = spks.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    h = pack([x, mu, spks_expanded, cond], "b * t")[0]
    print(f"\nAfter concat: {h.shape}, range=[{h.min():.4f}, {h.max():.4f}]")

    # Time embedding
    t_emb = decoder.estimator.time_mlp(t)
    print(f"Time embedding: {t_emb.shape}, range=[{t_emb.min():.4f}, {t_emb.max():.4f}]")
    print(f"  First 5: {t_emb[0, :5].numpy()}")

    # Down block ResNet
    down_block = decoder.estimator.down_blocks[0]
    resnet = down_block.resnets[0]

    print(f"\nBefore ResNet: h {h.shape}, range=[{h.min():.4f}, {h.max():.4f}]")

    # ResNet forward with tracing
    # Block 1
    h_res = resnet.block1(h)
    print(f"  After block1: range=[{h_res.min():.4f}, {h_res.max():.4f}]")

    # Time embedding projection
    t_proj = resnet.time_mlp_linear(torch.nn.functional.mish(t_emb))
    print(f"  Time proj: range=[{t_proj.min():.4f}, {t_proj.max():.4f}]")
    t_proj = t_proj.unsqueeze(-1)

    # Add time embedding
    h_res = h_res + t_proj
    print(f"  After adding time: range=[{h_res.min():.4f}, {h_res.max():.4f}]")

    # Block 2
    h_res = resnet.block2(h_res)
    print(f"  After block2: range=[{h_res.min():.4f}, {h_res.max():.4f}]")

    # Residual connection
    if resnet.res_conv is not None:
        h_skip = resnet.res_conv(h)
        print(f"  Skip connection: range=[{h_skip.min():.4f}, {h_skip.max():.4f}]")
        h_res = h_res + h_skip

    print(f"  Final ResNet output: range=[{h_res.min():.4f}, {h_res.max():.4f}]")

    return torch.zeros_like(x)  # Dummy return

# Trace
with torch.no_grad():
    traced_forward(x, mu, t, spk_emb, x_cond, mask)

print("\n" + "="*80)
print("✅ RESNET TRACE COMPLETE")
print("="*80)
