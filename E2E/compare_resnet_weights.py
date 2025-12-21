#!/usr/bin/env python3
"""Compare ResNet weights between Python model and s3gen_fp16.safetensors."""

import torch
from pathlib import Path
import sys
from safetensors import safe_open

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

def main():
    print("=" * 80)
    print("COMPARING RESNET WEIGHTS: PYTHON MODEL VS S3GEN_FP16")
    print("=" * 80)

    # Load Python model
    sys.path.insert(0, str(PROJECT_ROOT / "python"))
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model_dir = PROJECT_ROOT / "models" / "chatterbox"
    model = ChatterboxMultilingualTTS.from_local(str(model_dir), device="cpu")
    estimator = model.s3gen.flow.decoder.estimator

    # Compare first resnet block weights
    resnet = estimator.down_blocks[0][0]

    print("\n" + "=" * 40)
    print("PYTHON LIVE MODEL - down_blocks[0].resnet:")
    print("=" * 40)

    # block1
    block1_conv = resnet.block1.block[0]  # Conv1d
    block1_norm = resnet.block1.block[2]  # LayerNorm
    print(f"\nblock1.conv (block.0): weight shape = {block1_conv.weight.shape}")
    print(f"  weight[:,:,0].flatten()[:5] = {block1_conv.weight[:,:,0].flatten()[:5].tolist()}")
    print(f"block1.norm (block.2): weight shape = {block1_norm.weight.shape}")
    print(f"  weight[:5] = {block1_norm.weight[:5].tolist()}")

    # block2
    block2_conv = resnet.block2.block[0]
    block2_norm = resnet.block2.block[2]
    print(f"\nblock2.conv (block.0): weight shape = {block2_conv.weight.shape}")
    print(f"  weight[:,:,0].flatten()[:5] = {block2_conv.weight[:,:,0].flatten()[:5].tolist()}")

    # mlp
    mlp = resnet.mlp[1]  # Sequential: [SiLU, Linear]
    print(f"\nmlp[1] (Linear): weight shape = {mlp.weight.shape}")
    print(f"  weight[0,:5] = {mlp.weight[0,:5].tolist()}")

    # Now check s3gen_fp16.safetensors
    print("\n" + "=" * 40)
    print("S3GEN_FP16.SAFETENSORS:")
    print("=" * 40)

    st_path = PROJECT_ROOT / "models" / "mlx" / "s3gen_fp16.safetensors"
    with safe_open(st_path, framework="pt", device="cpu") as f:
        # block1
        block1_w = f.get_tensor("s3gen.flow.decoder.estimator.down_blocks_0.resnet.block1.conv.conv.weight")
        block1_n = f.get_tensor("s3gen.flow.decoder.estimator.down_blocks_0.resnet.block1.norm.weight")
        print(f"\nblock1.conv.conv.weight shape = {block1_w.shape}")
        print(f"  weight[:,:,0].flatten()[:5] = {block1_w[:,:,0].flatten()[:5].tolist()}")
        print(f"block1.norm.weight shape = {block1_n.shape}")
        print(f"  weight[:5] = {block1_n[:5].tolist()}")

        # block2
        block2_w = f.get_tensor("s3gen.flow.decoder.estimator.down_blocks_0.resnet.block2.conv.conv.weight")
        print(f"\nblock2.conv.conv.weight shape = {block2_w.shape}")
        print(f"  weight[:,:,0].flatten()[:5] = {block2_w[:,:,0].flatten()[:5].tolist()}")

        # mlp
        mlp_w = f.get_tensor("s3gen.flow.decoder.estimator.down_blocks_0.resnet.mlp_linear.weight")
        print(f"\nmlp_linear.weight shape = {mlp_w.shape}")
        print(f"  weight[0,:5] = {mlp_w[0,:5].tolist()}")

    # Now compare
    print("\n" + "=" * 40)
    print("COMPARISON:")
    print("=" * 40)

    with safe_open(st_path, framework="pt", device="cpu") as f:
        # Block1 conv weight
        st_block1_w = f.get_tensor("s3gen.flow.decoder.estimator.down_blocks_0.resnet.block1.conv.conv.weight")
        py_block1_w = block1_conv.weight

        # Check if shapes match (may need transpose for MLX format)
        print(f"\nblock1.conv weight: Python {py_block1_w.shape} vs Safetensors {st_block1_w.shape}")
        if py_block1_w.shape == st_block1_w.shape:
            diff = (py_block1_w - st_block1_w).abs().max().item()
            print(f"  Max diff (direct): {diff}")
        else:
            # Try transpose
            if py_block1_w.shape[0] == st_block1_w.shape[0] and py_block1_w.shape[1] == st_block1_w.shape[2]:
                st_transposed = st_block1_w.transpose(1, 2)
                diff = (py_block1_w - st_transposed).abs().max().item()
                print(f"  Max diff (after transpose): {diff}")

        # MLP weight
        st_mlp_w = f.get_tensor("s3gen.flow.decoder.estimator.down_blocks_0.resnet.mlp_linear.weight")
        py_mlp_w = mlp.weight
        print(f"\nmlp_linear weight: Python {py_mlp_w.shape} vs Safetensors {st_mlp_w.shape}")
        if py_mlp_w.shape == st_mlp_w.shape:
            diff = (py_mlp_w - st_mlp_w).abs().max().item()
            print(f"  Max diff (direct): {diff}")
        elif py_mlp_w.shape[0] == st_mlp_w.shape[1] and py_mlp_w.shape[1] == st_mlp_w.shape[0]:
            diff = (py_mlp_w - st_mlp_w.T).abs().max().item()
            print(f"  Max diff (transposed): {diff}")

if __name__ == "__main__":
    main()
