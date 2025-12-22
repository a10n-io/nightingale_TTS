#!/usr/bin/env python3
"""Test how Python decoder applies masking."""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path.home() / "Library/Python/3.9/lib/python/site-packages"))

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

print("="*80)
print("PYTHON DECODER MASKING TEST")
print("="*80)

# Load model
model_path = PROJECT_ROOT / "models" / "chatterbox" / "s3gen.pt"
state = torch.load(str(model_path), map_location='cpu')

# Extract decoder
from chatterbox.models.s3gen import S3Gen
model = S3Gen.from_pretrained(model_path.parent)
decoder = model.s3_gen.flow.decoder.estimator

# Load inputs
x_batched = torch.from_numpy(np.load(ref_dir / "step7_step1_x_before.npy"))
mu_batched = torch.from_numpy(np.load(ref_dir / "step7_mu_T.npy"))
cond = torch.from_numpy(np.load(ref_dir / "step6_x_cond.npy"))
spk_batched = torch.from_numpy(np.load(ref_dir / "step7_spk_emb.npy"))
mask_batched = torch.from_numpy(np.load(ref_dir / "step7_cond_T.npy"))

print(f"\nMask shape: {mask_batched.shape}")
print(f"Mask sum per sample: {mask_batched.sum(dim=-1)}")
print(f"Mask[0]: {mask_batched[0,0,:10].tolist()}")

# Check if decoder.down_blocks[0].resnet.block1 applies mask
print("\n" + "="*80)
print("Checking down_blocks[0].0.block1")
print("="*80)

x = x_batched[[0]]
mu = mu_batched[[0]]
spk = spk_batched[[0]]
mask = mask_batched[[0]]

spk_exp = spk.unsqueeze(-1).expand(-1, -1, x.shape[2])
h_in = torch.cat([x, mu, spk_exp, cond], dim=1)
print(f"h_in shape: {h_in.shape}, range=[{h_in.min():.4f}, {h_in.max():.4f}]")

# Check CausalBlock1D behavior
block1 = decoder.down_blocks[0].resnet.block1

# Check source of block1's forward method
import inspect
try:
    source = inspect.getsource(block1.__class__.forward)
    print("\nCausalBlock1D.forward source:")
    print(source)
except:
    print("\nCould not get source for CausalBlock1D.forward")
    print(f"block1 class: {block1.__class__}")
    print(f"block1 module: {block1.__class__.__module__}")

print("\n" + "="*80)
