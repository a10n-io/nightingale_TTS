#!/usr/bin/env python3
"""
Fix the attention biases in python_flow_weights.safetensors.
The original export had incorrect biases. This script updates them from the live Python model.
"""

import torch
from pathlib import Path
from safetensors.torch import load_file, save_file
import sys

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
FLOW_WEIGHTS_FILE = PROJECT_ROOT / "models" / "mlx" / "python_flow_weights.safetensors"

def main():
    print("=" * 80)
    print("FIXING ATTENTION BIASES IN python_flow_weights.safetensors")
    print("=" * 80)

    # Load Python model
    print("\nLoading Python model...")
    sys.path.insert(0, str(PROJECT_ROOT / "python"))
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model_dir = PROJECT_ROOT / "models" / "chatterbox"
    model = ChatterboxMultilingualTTS.from_local(str(model_dir), device="cpu")
    s3 = model.s3gen
    estimator = s3.flow.decoder.estimator

    # Load existing safetensors
    print(f"\nLoading {FLOW_WEIGHTS_FILE.name}...")
    weights = load_file(FLOW_WEIGHTS_FILE)
    print(f"Total keys: {len(weights)}")

    # Count and fix biases
    bias_keys_updated = 0

    # Build mapping from safetensors key to Python model path
    for key in weights.keys():
        if "out_proj.bias" in key:
            # Parse the key to get block type, block idx, transformer idx
            # Key format: decoder.down_blocks_0.transformer_0.attn.out_proj.bias
            parts = key.split(".")
            if "down_blocks" in key:
                block_type = "down_blocks"
                block_idx = int(parts[1].split("_")[2])  # down_blocks_0 -> 0
                tfmr_idx = int(parts[2].split("_")[1])   # transformer_0 -> 0
                py_attn = estimator.down_blocks[block_idx][1][tfmr_idx].attn1
            elif "mid_blocks" in key:
                block_type = "mid_blocks"
                block_idx = int(parts[1].split("_")[2])  # mid_blocks_0 -> 0
                tfmr_idx = int(parts[2].split("_")[1])   # transformer_0 -> 0
                py_attn = estimator.mid_blocks[block_idx][1][tfmr_idx].attn1
            elif "up_blocks" in key:
                block_type = "up_blocks"
                block_idx = int(parts[1].split("_")[2])  # up_blocks_0 -> 0
                tfmr_idx = int(parts[2].split("_")[1])   # transformer_0 -> 0
                py_attn = estimator.up_blocks[block_idx][1][tfmr_idx].attn1
            else:
                print(f"  Unknown block type in key: {key}")
                continue

            # Get the correct bias from Python model
            correct_bias = py_attn.to_out[0].bias.detach().clone()

            # Check the difference
            old_bias = weights[key]
            diff = (correct_bias - old_bias).abs().max().item()

            if diff > 0.001:
                print(f"  {key}: diff={diff:.6f} -> FIXING")
                weights[key] = correct_bias
                bias_keys_updated += 1
            else:
                print(f"  {key}: diff={diff:.6f} (OK)")

    print(f"\n✅ Updated {bias_keys_updated} biases")

    # Save back
    print(f"\nSaving updated weights to {FLOW_WEIGHTS_FILE.name}...")
    save_file(weights, FLOW_WEIGHTS_FILE)
    print(f"✅ Saved successfully")

    # Verify
    print("\nVerifying fix...")
    weights2 = load_file(FLOW_WEIGHTS_FILE)
    first_key = "decoder.down_blocks_0.transformer_0.attn.out_proj.bias"
    py_bias = estimator.down_blocks[0][1][0].attn1.to_out[0].bias
    st_bias = weights2[first_key]
    diff = (py_bias - st_bias).abs().max().item()
    print(f"  {first_key}: max_diff = {diff:.9f}")
    if diff < 1e-6:
        print("✅ Verification PASSED")
    else:
        print("❌ Verification FAILED")

if __name__ == "__main__":
    main()
