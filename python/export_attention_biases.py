#!/usr/bin/env python3
"""
Export missing attention output projection biases to safetensors.
These biases are NOT in the main safetensors file and need to be loaded separately.
"""

import torch
from pathlib import Path
from safetensors.torch import save_file

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
MODEL_DIR = PROJECT_ROOT / "models" / "chatterbox"
OUTPUT_FILE = MODEL_DIR / "decoder_attention_biases.safetensors"


def main():
    print("=" * 80)
    print("EXPORTING DECODER ATTENTION BIASES")
    print("=" * 80)

    # Load the model
    print("Loading model...")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device="cpu")
    s3 = model.s3gen

    # Get state dict
    state = s3.state_dict()

    # Filter for attention to_out biases
    bias_dict = {}
    for k, v in state.items():
        if 'to_out.0.bias' in k and ('down_blocks' in k or 'mid_blocks' in k or 'up_blocks' in k):
            bias_dict[k] = v
            print(f"  {k}: {v.shape}")

    print(f"\nTotal biases to export: {len(bias_dict)}")

    # Save to safetensors
    save_file(bias_dict, OUTPUT_FILE)
    print(f"\n✅ Saved to: {OUTPUT_FILE}")
    print(f"   File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
