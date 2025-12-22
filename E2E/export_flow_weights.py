#!/usr/bin/env python3
"""Export flow (encoder + decoder) weights from s3gen.pt to safetensors."""

import torch
from pathlib import Path
from safetensors.torch import save_file

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

print("="*80)
print("EXPORTING FLOW WEIGHTS (Encoder + Decoder)")
print("="*80)

# Load s3gen.pt
model_path = PROJECT_ROOT / "models" / "chatterbox" / "s3gen.pt"
print(f"\nLoading {model_path}...")
state_dict = torch.load(str(model_path), map_location='cpu')
print(f"Loaded {len(state_dict)} keys")

# Extract flow.* keys (encoder + decoder, but not vocoder)
flow_weights = {}
for key, value in state_dict.items():
    if key.startswith("flow."):
        flow_weights[key] = value

print(f"\nExtracted {len(flow_weights)} flow keys")

# Count encoder vs decoder
encoder_count = sum(1 for k in flow_weights.keys() if "encoder" in k)
decoder_count = sum(1 for k in flow_weights.keys() if "decoder" in k)
print(f"  Encoder keys: {encoder_count}")
print(f"  Decoder keys: {decoder_count}")

# Save to safetensors in both locations
for output_dir in [
    PROJECT_ROOT / "models" / "chatterbox",
    PROJECT_ROOT / "models" / "mlx"
]:
    output_path = output_dir / "python_flow_weights.safetensors"
    print(f"\nSaving to {output_path}...")
    save_file(flow_weights, str(output_path))
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved {len(flow_weights)} weights ({file_size_mb:.1f} MB)")

print("="*80)
