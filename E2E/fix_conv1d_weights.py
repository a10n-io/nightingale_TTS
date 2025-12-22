#!/usr/bin/env python3
"""
DEPRECATED - DO NOT RUN

This script was used to create s3gen_fp16_fixed.safetensors but it created a corrupted
file with mixed PyTorch/MLX formats. The correct file to use is:

    models/mlx/s3gen_fp16.safetensors

Which already has:
- Clean Swift key format (down_blocks_0.resnet.block1.conv.conv.weight)
- All Conv1d weights in MLX format [out, kernel, in]
- No duplicate keys

If you need to recreate weight files, use the proper export scripts in python/chatterbox/
"""

raise RuntimeError(
    "DEPRECATED: This script created corrupted weights. "
    "Use models/mlx/s3gen_fp16.safetensors instead."
)

# Original code preserved for reference only - DO NOT UNCOMMENT
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
... (rest of original code)
"""
