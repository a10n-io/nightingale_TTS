#!/usr/bin/env python3
"""Check all attention biases in Python model vs safetensors."""

import torch
import numpy as np
from pathlib import Path
import sys
from safetensors import safe_open

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

def main():
    print("=" * 80)
    print("CHECKING ALL ATTENTION BIASES: PYTHON MODEL VS SAFETENSORS")
    print("=" * 80)

    # Load Python model
    sys.path.insert(0, str(PROJECT_ROOT / "python"))
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model_dir = PROJECT_ROOT / "models" / "chatterbox"
    model = ChatterboxMultilingualTTS.from_local(str(model_dir), device="cpu")
    s3 = model.s3gen
    estimator = s3.flow.decoder.estimator

    safetensor_path = PROJECT_ROOT / "models" / "mlx" / "python_flow_weights.safetensors"

    total_bias_mismatch = 0
    total_weight_mismatch = 0

    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        # Check down_blocks
        for block_idx in range(1):  # Only 1 down block
            for tfmr_idx in range(4):
                tfmr = estimator.down_blocks[block_idx][1][tfmr_idx]
                attn = tfmr.attn1

                py_out_bias = attn.to_out[0].bias
                py_out_weight = attn.to_out[0].weight

                st_key_bias = f"decoder.down_blocks_{block_idx}.transformer_{tfmr_idx}.attn.out_proj.bias"
                st_key_weight = f"decoder.down_blocks_{block_idx}.transformer_{tfmr_idx}.attn.out_proj.weight"

                if st_key_bias in f.keys():
                    st_bias = f.get_tensor(st_key_bias)
                    bias_diff = (py_out_bias - st_bias).abs().max().item()
                    if bias_diff > 0.001:
                        print(f"down_blocks[{block_idx}].transformer[{tfmr_idx}].attn.out_proj.bias: diff={bias_diff:.6f}")
                        total_bias_mismatch += 1

                if st_key_weight in f.keys():
                    st_weight = f.get_tensor(st_key_weight)
                    weight_diff = (py_out_weight - st_weight).abs().max().item()
                    if weight_diff > 0.001:
                        print(f"down_blocks[{block_idx}].transformer[{tfmr_idx}].attn.out_proj.weight: diff={weight_diff:.6f}")
                        total_weight_mismatch += 1

        # Check mid_blocks
        for block_idx in range(12):
            for tfmr_idx in range(4):
                tfmr = estimator.mid_blocks[block_idx][1][tfmr_idx]
                attn = tfmr.attn1

                py_out_bias = attn.to_out[0].bias
                py_out_weight = attn.to_out[0].weight

                st_key_bias = f"decoder.mid_blocks_{block_idx}.transformer_{tfmr_idx}.attn.out_proj.bias"
                st_key_weight = f"decoder.mid_blocks_{block_idx}.transformer_{tfmr_idx}.attn.out_proj.weight"

                if st_key_bias in f.keys():
                    st_bias = f.get_tensor(st_key_bias)
                    bias_diff = (py_out_bias - st_bias).abs().max().item()
                    if bias_diff > 0.001:
                        print(f"mid_blocks[{block_idx}].transformer[{tfmr_idx}].attn.out_proj.bias: diff={bias_diff:.6f}")
                        total_bias_mismatch += 1

                if st_key_weight in f.keys():
                    st_weight = f.get_tensor(st_key_weight)
                    weight_diff = (py_out_weight - st_weight).abs().max().item()
                    if weight_diff > 0.001:
                        print(f"mid_blocks[{block_idx}].transformer[{tfmr_idx}].attn.out_proj.weight: diff={weight_diff:.6f}")
                        total_weight_mismatch += 1

        # Check up_blocks
        for block_idx in range(1):
            for tfmr_idx in range(4):
                tfmr = estimator.up_blocks[block_idx][1][tfmr_idx]
                attn = tfmr.attn1

                py_out_bias = attn.to_out[0].bias
                py_out_weight = attn.to_out[0].weight

                st_key_bias = f"decoder.up_blocks_{block_idx}.transformer_{tfmr_idx}.attn.out_proj.bias"
                st_key_weight = f"decoder.up_blocks_{block_idx}.transformer_{tfmr_idx}.attn.out_proj.weight"

                if st_key_bias in f.keys():
                    st_bias = f.get_tensor(st_key_bias)
                    bias_diff = (py_out_bias - st_bias).abs().max().item()
                    if bias_diff > 0.001:
                        print(f"up_blocks[{block_idx}].transformer[{tfmr_idx}].attn.out_proj.bias: diff={bias_diff:.6f}")
                        total_bias_mismatch += 1

                if st_key_weight in f.keys():
                    st_weight = f.get_tensor(st_key_weight)
                    weight_diff = (py_out_weight - st_weight).abs().max().item()
                    if weight_diff > 0.001:
                        print(f"up_blocks[{block_idx}].transformer[{tfmr_idx}].attn.out_proj.weight: diff={weight_diff:.6f}")
                        total_weight_mismatch += 1

    print("\n" + "=" * 80)
    print(f"SUMMARY: {total_bias_mismatch} bias mismatches, {total_weight_mismatch} weight mismatches")
    print("=" * 80)

if __name__ == "__main__":
    main()
