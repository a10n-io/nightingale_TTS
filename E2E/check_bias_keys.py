#!/usr/bin/env python3
"""Check keys in decoder_attention_biases.safetensors."""

from pathlib import Path
from safetensors import safe_open

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

def main():
    bias_file = PROJECT_ROOT / "models" / "chatterbox" / "decoder_attention_biases.safetensors"
    print(f"Keys in {bias_file.name}:")
    with safe_open(bias_file, framework="pt", device="cpu") as f:
        for key in sorted(f.keys()):
            tensor = f.get_tensor(key)
            print(f"  {key}: {tensor.shape}, first={tensor[0].item():.6f}")

if __name__ == "__main__":
    main()
