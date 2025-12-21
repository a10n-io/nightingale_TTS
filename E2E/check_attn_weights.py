#!/usr/bin/env python3
"""Check attention weights in Python model vs safetensors."""

import torch
import numpy as np
from pathlib import Path
import sys
from safetensors import safe_open

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

def main():
    print("=" * 80)
    print("CHECKING ATTENTION WEIGHTS: PYTHON MODEL VS SAFETENSORS")
    print("=" * 80)

    # Load Python model
    sys.path.insert(0, str(PROJECT_ROOT / "python"))
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model_dir = PROJECT_ROOT / "models" / "chatterbox"
    model = ChatterboxMultilingualTTS.from_local(str(model_dir), device="cpu")
    s3 = model.s3gen

    # Get the first transformer's attention
    estimator = s3.flow.decoder.estimator
    first_tfmr = estimator.down_blocks[0][1][0]
    attn = first_tfmr.attn1

    print("\n" + "=" * 80)
    print("PYTHON LIVE MODEL - first transformer attn weights:")
    print("=" * 80)

    py_q = attn.to_q.weight
    py_k = attn.to_k.weight
    py_v = attn.to_v.weight
    py_out = attn.to_out[0].weight
    py_out_bias = attn.to_out[0].bias

    print(f"\nto_q.weight: shape={py_q.shape}")
    print(f"  row 0, first 5: {py_q[0, :5].tolist()}")
    print(f"  col 0, first 5: {py_q[:5, 0].tolist()}")

    print(f"\nto_out[0].weight: shape={py_out.shape}")
    print(f"  row 0, first 5: {py_out[0, :5].tolist()}")
    print(f"  col 0, first 5: {py_out[:5, 0].tolist()}")

    print(f"\nto_out[0].bias: shape={py_out_bias.shape if py_out_bias is not None else None}")
    if py_out_bias is not None:
        print(f"  first 5: {py_out_bias[:5].tolist()}")

    # Now check safetensors
    print("\n" + "=" * 80)
    print("SAFETENSORS FILE - corresponding weights:")
    print("=" * 80)

    safetensor_path = PROJECT_ROOT / "models" / "mlx" / "python_flow_weights.safetensors"
    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        st_q = f.get_tensor("decoder.down_blocks_0.transformer_0.attn.query_proj.weight")
        st_k = f.get_tensor("decoder.down_blocks_0.transformer_0.attn.key_proj.weight")
        st_v = f.get_tensor("decoder.down_blocks_0.transformer_0.attn.value_proj.weight")
        st_out = f.get_tensor("decoder.down_blocks_0.transformer_0.attn.out_proj.weight")
        st_out_bias = f.get_tensor("decoder.down_blocks_0.transformer_0.attn.out_proj.bias")

        print(f"\nquery_proj.weight: shape={st_q.shape}")
        print(f"  row 0, first 5: {st_q[0, :5].tolist()}")
        print(f"  col 0, first 5: {st_q[:5, 0].tolist()}")

        print(f"\nout_proj.weight: shape={st_out.shape}")
        print(f"  row 0, first 5: {st_out[0, :5].tolist()}")
        print(f"  col 0, first 5: {st_out[:5, 0].tolist()}")

        print(f"\nout_proj.bias: shape={st_out_bias.shape}")
        print(f"  first 5: {st_out_bias[:5].tolist()}")

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON:")
    print("=" * 80)

    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        st_q = f.get_tensor("decoder.down_blocks_0.transformer_0.attn.query_proj.weight")
        st_out = f.get_tensor("decoder.down_blocks_0.transformer_0.attn.out_proj.weight")
        st_out_bias = f.get_tensor("decoder.down_blocks_0.transformer_0.attn.out_proj.bias")

        # Q weights comparison
        print(f"\nQ weights:")
        print(f"  Python shape: {py_q.shape}")
        print(f"  Safetensors shape: {st_q.shape}")
        if py_q.shape == st_q.shape:
            diff = (py_q - st_q).abs().max().item()
            print(f"  Max diff (direct): {diff}")
        if py_q.shape[0] == st_q.shape[1] and py_q.shape[1] == st_q.shape[0]:
            diff_t = (py_q - st_q.T).abs().max().item()
            print(f"  Max diff (transposed): {diff_t}")

        # Out weights comparison
        print(f"\nOut_proj weights:")
        print(f"  Python shape: {py_out.shape}")
        print(f"  Safetensors shape: {st_out.shape}")
        if py_out.shape == st_out.shape:
            diff = (py_out - st_out).abs().max().item()
            print(f"  Max diff (direct): {diff}")
        if py_out.shape[0] == st_out.shape[1] and py_out.shape[1] == st_out.shape[0]:
            diff_t = (py_out - st_out.T).abs().max().item()
            print(f"  Max diff (transposed): {diff_t}")

        # Bias comparison
        print(f"\nOut_proj bias:")
        print(f"  Python shape: {py_out_bias.shape}")
        print(f"  Safetensors shape: {st_out_bias.shape}")
        if py_out_bias is not None and py_out_bias.shape == st_out_bias.shape:
            diff = (py_out_bias - st_out_bias).abs().max().item()
            print(f"  Max diff: {diff}")

    # Check attention config
    print("\n" + "=" * 80)
    print("ATTENTION CONFIG:")
    print("=" * 80)
    print(f"attn.heads: {attn.heads}")
    print(f"attn.dim_head: {attn.dim_head if hasattr(attn, 'dim_head') else 'N/A'}")
    print(f"attn.inner_dim: {attn.inner_dim if hasattr(attn, 'inner_dim') else 'N/A'}")
    print(f"attn.scale: {attn.scale}")

if __name__ == "__main__":
    main()
