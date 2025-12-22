#!/usr/bin/env python3
"""Export full S3Gen weights from s3gen.pt to safetensors."""

import torch
from pathlib import Path
from safetensors.torch import save_file

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

print("="*80)
print("EXPORTING FULL S3GEN WEIGHTS TO SAFETENSORS")
print("="*80)

# Load s3gen.pt
model_path = PROJECT_ROOT / "models" / "chatterbox" / "s3gen.pt"
print(f"\nLoading {model_path}...")
state_dict = torch.load(str(model_path), map_location='cpu')
print(f"Loaded {len(state_dict)} keys")

# Add s3gen prefix to all keys (ChatterboxEngine expects this)
s3gen_weights = {}
for key, value in state_dict.items():
    prefixed_key = f"s3gen.{key}"
    s3gen_weights[prefixed_key] = value
    
print(f"\nPrefixed {len(s3gen_weights)} weights with 's3gen.'")

# Save to safetensors
output_path = PROJECT_ROOT / "models" / "chatterbox" / "s3gen_fp16.safetensors"
print(f"\nSaving to {output_path}...")
save_file(s3gen_weights, str(output_path))

file_size_mb = output_path.stat().st_size / (1024 * 1024)
print(f"✅ Exported {len(s3gen_weights)} weights ({file_size_mb:.1f} MB)")
print(f"   {output_path}")
print("="*80)
