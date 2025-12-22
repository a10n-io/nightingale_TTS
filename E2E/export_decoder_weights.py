#!/usr/bin/env python3
"""Export S3Gen decoder weights to safetensors for Swift verification."""

import torch
from pathlib import Path
from safetensors.torch import save_file

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

print("Loading S3Gen state dict...")
model_path = PROJECT_ROOT / "models" / "chatterbox" / "s3gen.pt"
state_dict = torch.load(str(model_path), map_location='cpu')

print(f"Loaded {len(state_dict)} keys from s3gen.pt")

# Extract decoder weights
print("\nExtracting decoder weights...")
decoder_weights = {}
for key, value in state_dict.items():
    # Keys in s3gen.pt are like: "flow.decoder.down_blocks.0.conv_1d.weight"
    # We want them prefixed as: "s3gen.flow.decoder.down_blocks.0.conv_1d.weight"
    if "flow.decoder" in key:
        prefixed_key = f"s3gen.{key}"
        decoder_weights[prefixed_key] = value
        print(f"  {prefixed_key}: {tuple(value.shape)}")

# Save to safetensors
output_path = PROJECT_ROOT / "models" / "chatterbox" / "decoder_weights.safetensors"
save_file(decoder_weights, str(output_path))

print(f"\n✅ Exported {len(decoder_weights)} decoder weights to:")
print(f"   {output_path}")
