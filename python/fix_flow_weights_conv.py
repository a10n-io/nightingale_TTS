#!/usr/bin/env python3
"""
Fix the res_conv weight shapes in python_flow_weights.safetensors.
MLX Conv1d expects: [out_channels, kernel_size, in_channels]
PyTorch Conv1d has: [out_channels, in_channels, kernel_size]

We need to TRANSPOSE the PyTorch weights to MLX format.
"""

import torch
from pathlib import Path
from safetensors.torch import load_file, save_file
import sys

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
FLOW_WEIGHTS_FILE = PROJECT_ROOT / "models" / "mlx" / "python_flow_weights.safetensors"

def main():
    print("=" * 80)
    print("FIXING RES_CONV WEIGHT SHAPES FOR MLX FORMAT")
    print("=" * 80)

    # Load Python model
    print("\nLoading Python model...")
    sys.path.insert(0, str(PROJECT_ROOT / "python"))
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model_dir = PROJECT_ROOT / "models" / "chatterbox"
    model = ChatterboxMultilingualTTS.from_local(str(model_dir), device="cpu")
    estimator = model.s3gen.flow.decoder.estimator

    # Load existing safetensors
    print(f"\nLoading {FLOW_WEIGHTS_FILE.name}...")
    weights = load_file(FLOW_WEIGHTS_FILE)

    fixed = 0

    # Fix res_conv weights - need to transpose from PyTorch to MLX format
    res_conv_keys = [k for k in weights.keys() if "res_conv" in k or "final_proj" in k]
    print(f"\nFound {len(res_conv_keys)} res_conv/final_proj keys")

    for key in res_conv_keys:
        st_tensor = weights[key]
        # Get correct tensor from Python model
        py_key = key_to_py(key)
        if py_key:
            py_tensor = get_tensor(estimator, py_key)
            if py_tensor is not None:
                # Only transpose weight tensors, not biases
                if len(py_tensor.shape) == 3:
                    # PyTorch: [out_channels, in_channels, kernel_size]
                    # MLX: [out_channels, kernel_size, in_channels]
                    # Transpose: swap dims 1 and 2
                    mlx_tensor = py_tensor.transpose(1, 2).contiguous()
                else:
                    # Bias - no transpose needed
                    mlx_tensor = py_tensor.clone()

                if st_tensor.shape != mlx_tensor.shape:
                    print(f"  {key}: {st_tensor.shape} -> {mlx_tensor.shape}")
                    weights[key] = mlx_tensor.clone()
                    fixed += 1
                else:
                    diff = (st_tensor - mlx_tensor).abs().max().item()
                    if diff > 1e-6:
                        print(f"  {key}: same shape but values differ by {diff:.6f}, replacing")
                        weights[key] = mlx_tensor.clone()
                        fixed += 1

    print(f"\n✅ Fixed {fixed} weights (transposed to MLX format)")

    # Save
    print(f"\nSaving to {FLOW_WEIGHTS_FILE.name}...")
    save_file(weights, FLOW_WEIGHTS_FILE)
    print("✅ Saved successfully")

    # Verify
    print("\nVerifying shapes are now in MLX format:")
    weights2 = load_file(FLOW_WEIGHTS_FILE)
    for key in ["decoder.down_blocks_0.resnet.res_conv.weight", "decoder.final_proj.weight"]:
        if key in weights2:
            print(f"  {key}: {weights2[key].shape}")

def key_to_py(key):
    """Map safetensors key to Python model path."""
    import re

    if "down_blocks_" in key:
        m = re.match(r"decoder\.down_blocks_(\d+)\.resnet\.res_conv\.(\w+)", key)
        if m:
            return f"down_blocks[{m.group(1)}][0].res_conv.{m.group(2)}"

    if "mid_blocks_" in key:
        m = re.match(r"decoder\.mid_blocks_(\d+)\.resnet\.res_conv\.(\w+)", key)
        if m:
            return f"mid_blocks[{m.group(1)}][0].res_conv.{m.group(2)}"

    if "up_blocks_" in key:
        m = re.match(r"decoder\.up_blocks_(\d+)\.resnet\.res_conv\.(\w+)", key)
        if m:
            return f"up_blocks[{m.group(1)}][0].res_conv.{m.group(2)}"

    if "final_proj" in key:
        m = re.match(r"decoder\.final_proj\.(\w+)", key)
        if m:
            return f"final_proj.{m.group(1)}"

    return None

def get_tensor(estimator, py_path):
    """Get tensor from estimator using path string."""
    import re

    if py_path.startswith("down_blocks"):
        m = re.match(r"down_blocks\[(\d+)\]\[0\]\.res_conv\.(\w+)", py_path)
        if m:
            block = estimator.down_blocks[int(m.group(1))][0]
            return getattr(block.res_conv, m.group(2))

    if py_path.startswith("mid_blocks"):
        m = re.match(r"mid_blocks\[(\d+)\]\[0\]\.res_conv\.(\w+)", py_path)
        if m:
            block = estimator.mid_blocks[int(m.group(1))][0]
            return getattr(block.res_conv, m.group(2))

    if py_path.startswith("up_blocks"):
        m = re.match(r"up_blocks\[(\d+)\]\[0\]\.res_conv\.(\w+)", py_path)
        if m:
            block = estimator.up_blocks[int(m.group(1))][0]
            return getattr(block.res_conv, m.group(2))

    if py_path.startswith("final_proj"):
        m = re.match(r"final_proj\.(\w+)", py_path)
        if m:
            return getattr(estimator.final_proj, m.group(1))

    return None

if __name__ == "__main__":
    main()
