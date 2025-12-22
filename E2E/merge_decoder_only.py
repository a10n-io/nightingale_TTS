#!/usr/bin/env python3
"""
DEPRECATED - DO NOT RUN

This script merged decoder weights into s3gen_fp16_fixed.safetensors which created
a corrupted file with duplicate keys in both Python and Swift formats.

The correct file to use is:

    models/mlx/s3gen_fp16.safetensors

Which has clean Swift key format and MLX Conv1d format.
"""

raise RuntimeError(
    "DEPRECATED: This script created corrupted weights with duplicate keys. "
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
