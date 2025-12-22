#!/usr/bin/env python3
"""Check LayerNorm weight values."""

import torch
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
model_path = PROJECT_ROOT / "models" / "chatterbox" / "s3gen.pt"
state_dict = torch.load(str(model_path), map_location='cpu')

prefix = 'flow.decoder.estimator.down_blocks.0.1.0.norm1'

norm1_weight = state_dict[f'{prefix}.weight']
norm1_bias = state_dict[f'{prefix}.bias']

print("LayerNorm weights:")
print(f"  weight shape: {norm1_weight.shape}")
print(f"  weight range: [{norm1_weight.min().item():.6f}, {norm1_weight.max().item():.6f}]")
print(f"  weight mean: {norm1_weight.mean().item():.6f}")
print(f"  weight std: {norm1_weight.std().item():.6f}")
print(f"  bias shape: {norm1_bias.shape}")
print(f"  bias range: [{norm1_bias.min().item():.6f}, {norm1_bias.max().item():.6f}]")
print(f"  bias mean: {norm1_bias.mean().item():.6f}")
print(f"  bias std: {norm1_bias.std().item():.6f}")

print(f"\nFirst 10 weight values:")
print(norm1_weight[:10])
print(f"\nFirst 10 bias values:")
print(norm1_bias[:10])
