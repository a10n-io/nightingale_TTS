#!/usr/bin/env python3
"""Check what's in safetensors for LayerNorm."""

import sys
sys.path.insert(0, "/Users/a10n/Library/Python/3.9/lib/python/site-packages")
from safetensors import safe_open
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
safetensors_path = PROJECT_ROOT / "models" / "chatterbox" / "decoder_weights.safetensors"

print("Searching for norm1 in decoder_weights.safetensors:")
print("="*80)

with safe_open(safetensors_path, framework="pt", device="cpu") as f:
    for key in sorted(f.keys()):
        if "norm1" in key and "down_blocks.0.1.0" in key:
            tensor = f.get_tensor(key)
            print(f"\n{key}")
            print(f"  Shape: {tensor.shape}")
            print(f"  Range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
            print(f"  First 10 values: {tensor[:10].tolist()}")
