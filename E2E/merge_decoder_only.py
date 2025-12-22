#!/usr/bin/env python3
"""Merge ONLY decoder weights from python_flow_weights into s3gen_fp16_fixed.safetensors."""

import torch
from pathlib import Path
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

print("="*80)
print("MERGING DECODER WEIGHTS ONLY")
print("="*80)

# Load existing s3gen_fp16_fixed (before merge)
# First restore it by loading original
orig_path = PROJECT_ROOT / "models" / "mlx" / "s3gen_fp16.safetensors"
print(f"\n1. Loading original {orig_path}...")
s3gen_weights = load_file(str(orig_path))
print(f"   Loaded {len(s3gen_weights)} keys")

# Load python_flow_weights
flow_path = PROJECT_ROOT / "models" / "mlx" / "python_flow_weights.safetensors"
print(f"\n2. Loading {flow_path}...")
flow_weights = load_file(str(flow_path))
print(f"   Loaded {len(flow_weights)} keys")

# Extract ONLY decoder weights
decoder_weights = {k: v for k, v in flow_weights.items() if "decoder" in k}
print(f"\n3. Filtered to {len(decoder_weights)} decoder weights")

# Merge: decoder weights overwrite s3gen weights
merged = dict(s3gen_weights)
overwritten = 0
added = 0
for key, value in decoder_weights.items():
    if key in merged:
        overwritten += 1
    else:
        added += 1
    merged[key] = value

print(f"   Overwritten: {overwritten} existing keys")
print(f"   Added: {added} new keys")
print(f"   Total keys: {len(merged)}")

# Save merged
output_path = PROJECT_ROOT / "models" / "mlx" / "s3gen_fp16_fixed.safetensors"
print(f"\n4. Saving to {output_path}...")
save_file(merged, str(output_path))

file_size_mb = output_path.stat().st_size / (1024 * 1024)
print(f"   ✅ Saved {len(merged)} weights ({file_size_mb:.1f} MB)")
print("="*80)
