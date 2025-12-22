#!/usr/bin/env python3
"""Merge python_flow_weights into s3gen_fp16_fixed.safetensors."""

import torch
from pathlib import Path
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

print("="*80)
print("MERGING PYTHON FLOW WEIGHTS INTO s3gen_fp16_fixed.safetensors")
print("="*80)

# Load existing s3gen_fp16_fixed
s3gen_path = PROJECT_ROOT / "models" / "mlx" / "s3gen_fp16_fixed.safetensors"
print(f"\n1. Loading {s3gen_path}...")
s3gen_weights = load_file(str(s3gen_path))
print(f"   Loaded {len(s3gen_weights)} keys")

# Load python_flow_weights
flow_path = PROJECT_ROOT / "models" / "mlx" / "python_flow_weights.safetensors"
print(f"\n2. Loading {flow_path}...")
flow_weights = load_file(str(flow_path))
print(f"   Loaded {len(flow_weights)} keys")

# Merge: flow weights overwrite s3gen weights
print(f"\n3. Merging...")
merged = dict(s3gen_weights)
overwritten = 0
for key, value in flow_weights.items():
    if key in merged:
        overwritten += 1
    merged[key] = value

print(f"   Merged {len(flow_weights)} flow weights")
print(f"   Overwritten: {overwritten} existing keys")
print(f"   Total keys: {len(merged)}")

# Save merged
output_path = PROJECT_ROOT / "models" / "mlx" / "s3gen_fp16_fixed.safetensors"
print(f"\n4. Saving to {output_path}...")
save_file(merged, str(output_path))

file_size_mb = output_path.stat().st_size / (1024 * 1024)
print(f"   ✅ Saved {len(merged)} weights ({file_size_mb:.1f} MB)")
print("="*80)
