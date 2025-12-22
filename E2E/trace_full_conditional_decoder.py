#!/usr/bin/env python3
"""Trace full conditional decoder pass step-by-step."""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path.home() / "Library/Python/3.9/lib/python/site-packages"))

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

# Load decoder state dict
model_path = PROJECT_ROOT / "models" / "chatterbox" / "s3gen.pt"
state_dict = torch.load(str(model_path), map_location='cpu')

# Import decoder class
from chatterbox.models.s3gen.decoder import Decoder

# Create decoder
decoder = Decoder(
    channels=[256],
    attention_head_dim=64,
    num_mid_blocks=12,
    num_res_blocks=1,
    num_heads=8,
    spk_emb_dim=80,
    dropout=0,
)

# Load weights
decoder_state = {}
for k, v in state_dict.items():
    if k.startswith('flow.decoder.estimator.'):
        new_k = k.replace('flow.decoder.estimator.', '')
        decoder_state[new_k] = v

decoder.load_state_dict(decoder_state)
decoder.eval()

# Load inputs
x = torch.from_numpy(np.load(ref_dir / "step7_step1_x_before.npy"))[[0]]  # [1, 80, 696]
mu = torch.from_numpy(np.load(ref_dir / "step7_mu_T.npy"))[[0]]  # [1, 80, 696]
cond = torch.from_numpy(np.load(ref_dir / "step6_x_cond.npy"))  # [1, 80, 696]
spk = torch.from_numpy(np.load(ref_dir / "step7_spk_emb.npy"))[[0]]  # [1, 80]
mask = torch.from_numpy(np.load(ref_dir / "step7_cond_T.npy"))[[0]]  # [1, 1, 696]
t = torch.from_numpy(np.load(ref_dir / "step7_step1_t.npy"))  # []

print("=" * 80)
print("FULL CONDITIONAL DECODER TRACE")
print("=" * 80)

print(f"\nInputs:")
print(f"  x:    {x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")
print(f"  mu:   {mu.shape}, range=[{mu.min():.4f}, {mu.max():.4f}]")
print(f"  cond: {cond.shape}, range=[{cond.min():.4f}, {cond.max():.4f}]")
print(f"  spk:  {spk.shape}, range=[{spk.min():.4f}, {spk.max():.4f}]")
print(f"  mask: {mask.shape}, sum={mask.sum().item()}")
print(f"  t:    {t.item():.6f}")

# Call decoder
with torch.no_grad():
    output = decoder(x=x, mask=mask, mu=mu, t=t.unsqueeze(0), spks=spk, cond=cond)

print(f"\nOutput:")
print(f"  shape: {output.shape}")
print(f"  range: [{output.min():.4f}, {output.max():.4f}]")
print(f"  mean:  {output.mean():.4f}")

# Save for Swift comparison
np.save(ref_dir / "../../python_full_decoder_cond_output.npy", output.numpy())
print(f"\n✅ Saved Python full decoder output")

# Check unmasked vs masked
unmasked = output[:, :, mask[0, 0] > 0]
masked = output[:, :, mask[0, 0] == 0]
print(f"\nUnmasked positions: {unmasked.shape[2]}/{output.shape[2]}")
print(f"  range: [{unmasked.min():.4f}, {unmasked.max():.4f}]")
print(f"  mean:  {unmasked.mean():.4f}")
print(f"\nMasked positions: {masked.shape[2]}/{output.shape[2]}")
print(f"  range: [{masked.min():.4f}, {masked.max():.4f}]")
print(f"  mean:  {masked.mean():.4f}")
