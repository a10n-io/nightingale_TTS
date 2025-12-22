#!/usr/bin/env python3
"""
DEPRECATED - DO NOT RUN

This script merged weights into s3gen_fp16_fixed.safetensors which created a corrupted
file with mixed key formats (both Python and Swift naming conventions).

The correct file to use is:

    models/mlx/s3gen_fp16.safetensors

Which has clean Swift key format and MLX Conv1d format.
"""

raise RuntimeError(
    "DEPRECATED: This script created corrupted weights with mixed formats. "
    "Use models/mlx/s3gen_fp16.safetensors instead."
)

# Original code preserved for reference only - DO NOT UNCOMMENT
"""
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
... (rest of original code)
"""
