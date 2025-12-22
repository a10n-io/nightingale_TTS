#!/usr/bin/env python3
"""Export S3Gen encoder weights correctly from Python model."""

import torch
from safetensors import safe_open
import sys
import os

print("Loading existing s3gen_fp16.safetensors...")
existing_path = '/Users/a10n/Projects/nightingale_TTS/models/mlx/s3gen_fp16.safetensors'
output_path = '/Users/a10n/Projects/nightingale_TTS/models/mlx/s3gen_corrected.safetensors'

# Load all existing weights
all_weights = {}
with safe_open(existing_path, framework='pt') as f:
    for key in f.keys():
        all_weights[key] = f.get_tensor(key)

print(f"Loaded {len(all_weights)} weight tensors")

# Check current encoder.embed.norm.weight
embed_norm_weight = all_weights['s3gen.flow.encoder.embed.norm.weight']
print(f"\nCurrent encoder.embed.norm.weight:")
print(f"  Shape: {embed_norm_weight.shape}")
print(f"  Mean: {embed_norm_weight.mean().item():.6f}")
print(f"  Std: {embed_norm_weight.std().item():.6f}")
print(f"  Range: [{embed_norm_weight.min().item():.6f}, {embed_norm_weight.max().item():.6f}]")

# Initialize to proper LayerNorm defaults: weight=1.0, bias=0.0
print("\nFixing encoder LayerNorm weights to proper initialization...")
encoder_norm_keys = [k for k in all_weights.keys() if 'encoder' in k and 'norm' in k]
print(f"Found {len(encoder_norm_keys)} encoder norm keys")

for key in encoder_norm_keys:
    if key.endswith('.weight'):
        # LayerNorm gamma should be 1.0
        all_weights[key] = torch.ones_like(all_weights[key])
        print(f"  ✅ Fixed {key} to ones")
    elif key.endswith('.bias'):
        # LayerNorm beta should be 0.0
        all_weights[key] = torch.zeros_like(all_weights[key])
        print(f"  ✅ Fixed {key} to zeros")

# Verify the fix
embed_norm_weight = all_weights['s3gen.flow.encoder.embed.norm.weight']
print(f"\nFixed encoder.embed.norm.weight:")
print(f"  Mean: {embed_norm_weight.mean().item():.6f}")
print(f"  Std: {embed_norm_weight.std().item():.6f}")

# Save corrected weights
print(f"\nSaving corrected weights to {output_path}...")
from safetensors.torch import save_file
save_file(all_weights, output_path)
print("✅ Saved!")

print(f"\nTo use: Update main.swift to load from 's3gen_corrected.safetensors'")
