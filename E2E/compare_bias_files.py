#!/usr/bin/env python3
"""Compare biases in different safetensors files vs Python live model."""

import torch
import numpy as np
from pathlib import Path
import sys
from safetensors import safe_open

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

def main():
    print("=" * 80)
    print("COMPARING BIAS FILES VS PYTHON LIVE MODEL")
    print("=" * 80)

    # Load Python model
    sys.path.insert(0, str(PROJECT_ROOT / "python"))
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model_dir = PROJECT_ROOT / "models" / "chatterbox"
    model = ChatterboxMultilingualTTS.from_local(str(model_dir), device="cpu")
    s3 = model.s3gen
    estimator = s3.flow.decoder.estimator

    # Get Python live model bias for first transformer
    first_attn = estimator.down_blocks[0][1][0].attn1
    py_bias = first_attn.to_out[0].bias
    print(f"\nPython live model - down_blocks[0].transformer[0].attn.to_out[0].bias:")
    print(f"  shape: {py_bias.shape}")
    print(f"  first 5: {py_bias[:5].tolist()}")
    print(f"  last 5: {py_bias[-5:].tolist()}")
    print(f"  mean: {py_bias.mean().item():.6f}")

    # Check decoder_attention_biases.safetensors
    print("\n" + "-" * 40)
    bias_file = PROJECT_ROOT / "models" / "chatterbox" / "decoder_attention_biases.safetensors"
    if bias_file.exists():
        print(f"decoder_attention_biases.safetensors:")
        with safe_open(bias_file, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"  Total keys: {len(keys)}")
            # Find the matching key
            for key in keys:
                if "down_blocks.0" in key and "transformers.0" in key and "to_out.0.bias" in key:
                    bias = f.get_tensor(key)
                    print(f"  Key: {key}")
                    print(f"  shape: {bias.shape}")
                    print(f"  first 5: {bias[:5].tolist()}")
                    print(f"  last 5: {bias[-5:].tolist()}")
                    print(f"  mean: {bias.mean().item():.6f}")
                    diff = (py_bias - bias).abs().max().item()
                    print(f"  MAX DIFF vs Python: {diff}")
                    break
    else:
        print("decoder_attention_biases.safetensors: NOT FOUND")

    # Check python_flow_weights.safetensors
    print("\n" + "-" * 40)
    flow_file = PROJECT_ROOT / "models" / "mlx" / "python_flow_weights.safetensors"
    if flow_file.exists():
        print(f"python_flow_weights.safetensors:")
        with safe_open(flow_file, framework="pt", device="cpu") as f:
            # Find the matching key
            key = "decoder.down_blocks_0.transformer_0.attn.out_proj.bias"
            if key in f.keys():
                bias = f.get_tensor(key)
                print(f"  Key: {key}")
                print(f"  shape: {bias.shape}")
                print(f"  first 5: {bias[:5].tolist()}")
                print(f"  last 5: {bias[-5:].tolist()}")
                print(f"  mean: {bias.mean().item():.6f}")
                diff = (py_bias - bias).abs().max().item()
                print(f"  MAX DIFF vs Python: {diff}")
            else:
                print(f"  Key '{key}' NOT FOUND")
    else:
        print("python_flow_weights.safetensors: NOT FOUND")

    # Solution: we need to either:
    # 1. Re-export python_flow_weights.safetensors with correct biases, OR
    # 2. Load decoder_attention_biases.safetensors to override the wrong biases

if __name__ == "__main__":
    main()
