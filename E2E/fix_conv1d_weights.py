#!/usr/bin/env python3
"""Fix all Conv1d weights in s3gen_fp16.safetensors to consistent MLX format.

PyTorch Conv1d format: [out_channels, in_channels, kernel_size]
MLX Conv1d format:     [out_channels, kernel_size, in_channels]

This script detects which format each Conv1d weight is in and converts to MLX format.
"""

import torch
from safetensors import safe_open
from safetensors.torch import save_file
import numpy as np

print("="*80)
print("FIXING CONV1D WEIGHTS TO MLX FORMAT")
print("="*80)

existing_path = '/Users/a10n/Projects/nightingale_TTS/models/mlx/s3gen_fp16.safetensors'
output_path = '/Users/a10n/Projects/nightingale_TTS/models/mlx/s3gen_fp16_fixed.safetensors'

print(f"\nLoading weights from: {existing_path}")

# Load all weights
all_weights = {}
with safe_open(existing_path, framework='pt') as f:
    for key in f.keys():
        all_weights[key] = f.get_tensor(key)

print(f"Loaded {len(all_weights)} weight tensors")

# Find all Conv1d weights (3D tensors ending in .weight)
conv1d_weights = []
for key, tensor in all_weights.items():
    if key.endswith('.weight') and tensor.ndim == 3:
        conv1d_weights.append((key, tensor))

print(f"\nFound {len(conv1d_weights)} Conv1d weights to check")

# Analyze and convert each Conv1d weight
conversions = []
already_mlx = []

for key, tensor in conv1d_weights:
    shape = list(tensor.shape)

    # Heuristic to detect format:
    # Conv1d kernels are typically small (1, 3, 5, 7, sometimes up to 30)
    # PyTorch: [out, in, kernel] - kernel is last and usually smallest
    # MLX:     [out, kernel, in] - kernel is middle and usually smallest

    # Compare middle dimension (shape[1]) vs last dimension (shape[2])
    # If shape[1] < shape[2], likely already MLX format (kernel < in)
    # If shape[2] < shape[1], likely PyTorch format (kernel < in)

    # Special case: ConvTranspose1d uses different transpose
    is_conv_transpose = 'ups.' in key or 'upsample' in key.lower()

    if shape[1] < shape[2]:
        # Middle dimension is smaller - likely MLX [out, kernel, in]
        already_mlx.append((key, shape))
    else:
        # Last dimension is smaller - likely PyTorch [out, in, kernel]
        if is_conv_transpose:
            # ConvTranspose1d: transpose (1, 2, 0)
            transposed = tensor.permute(1, 2, 0).contiguous()
            all_weights[key] = transposed
            conversions.append((key, shape, list(transposed.shape), 'ConvTranspose'))
        else:
            # Regular Conv1d: transpose (0, 2, 1)
            transposed = tensor.permute(0, 2, 1).contiguous()
            all_weights[key] = transposed
            conversions.append((key, shape, list(transposed.shape), 'Conv1d'))

print(f"\n✅ Already in MLX format: {len(already_mlx)} weights")
if already_mlx and len(already_mlx) <= 10:
    for key, shape in already_mlx[:10]:
        print(f"  {key}: {shape}")
elif already_mlx:
    print(f"  (showing first 5)")
    for key, shape in already_mlx[:5]:
        print(f"  {key}: {shape}")

print(f"\n🔄 Converted to MLX format: {len(conversions)} weights")
for key, old_shape, new_shape, conv_type in conversions:
    print(f"  {key}:")
    print(f"    {old_shape} -> {new_shape} ({conv_type})")

# Verify no shape mismatches
print("\n🔍 Verification:")
for key, tensor in all_weights.items():
    if key.endswith('.weight') and tensor.ndim == 3:
        shape = list(tensor.shape)
        # After conversion, ALL Conv1d should have kernel in middle (smallest dimension)
        if shape[1] > shape[2]:
            print(f"  ⚠️  WARNING: {key} may still be in wrong format: {shape}")

# Save corrected weights
print(f"\n💾 Saving corrected weights to: {output_path}")
save_file(all_weights, output_path)
print("✅ Done!")

print(f"\n📝 Summary:")
print(f"  Total weights: {len(all_weights)}")
print(f"  Conv1d weights found: {len(conv1d_weights)}")
print(f"  Already MLX format: {len(already_mlx)}")
print(f"  Converted to MLX: {len(conversions)}")
print(f"\n✅ All Conv1d weights are now in MLX format: [out_channels, kernel_size, in_channels]")
print(f"\nTo use the fixed weights, update main.swift to load from:")
print(f"  '{output_path}'")
