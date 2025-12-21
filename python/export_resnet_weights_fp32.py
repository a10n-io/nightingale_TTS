#!/usr/bin/env python3
"""
Export the 148 missing ResNet weights (block1, block2, mlp) from Python model
to python_flow_weights.safetensors in FP32 format.

These weights are currently loaded from s3gen_fp16.safetensors in FP16,
which may be causing the 5.44 max_diff in Step 7a verification.
"""

import torch
from pathlib import Path
from safetensors.torch import load_file, save_file
import sys
import re

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
FLOW_WEIGHTS_FILE = PROJECT_ROOT / "models" / "mlx" / "python_flow_weights.safetensors"

def main():
    print("=" * 80)
    print("EXPORTING RESNET WEIGHTS (block1, block2, mlp) IN FP32")
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
    original_count = len(weights)
    print(f"Original key count: {original_count}")

    added = 0

    # Export down_blocks ResNet weights
    for i, block in enumerate(estimator.down_blocks):
        resnet = block[0]  # First element is ResNetBlock
        added += export_resnet_block(weights, resnet, f"decoder.down_blocks_{i}.resnet")

    # Export mid_blocks ResNet weights
    for i, block in enumerate(estimator.mid_blocks):
        resnet = block[0]
        added += export_resnet_block(weights, resnet, f"decoder.mid_blocks_{i}.resnet")

    # Export up_blocks ResNet weights
    for i, block in enumerate(estimator.up_blocks):
        resnet = block[0]
        added += export_resnet_block(weights, resnet, f"decoder.up_blocks_{i}.resnet")

    print(f"\n✅ Added {added} new weights in FP32")
    print(f"New total key count: {len(weights)}")

    # Save
    print(f"\nSaving to {FLOW_WEIGHTS_FILE.name}...")
    save_file(weights, FLOW_WEIGHTS_FILE)
    print("✅ Saved successfully")

    # Verify a few weights
    print("\nVerifying exported weights:")
    weights2 = load_file(FLOW_WEIGHTS_FILE)
    sample_keys = [
        "decoder.down_blocks_0.resnet.block1.conv.conv.weight",
        "decoder.down_blocks_0.resnet.block1.norm.weight",
        "decoder.down_blocks_0.resnet.mlp_linear.weight",
    ]
    for key in sample_keys:
        if key in weights2:
            t = weights2[key]
            print(f"  {key}: shape={t.shape}, dtype={t.dtype}")

def export_resnet_block(weights, resnet, prefix):
    """Export a ResNet block's weights (block1, block2, mlp) with proper MLX format."""
    added = 0

    # block1: Sequential(Conv1d, SiLU, LayerNorm)
    # block1.block[0] = Conv1d, block1.block[2] = LayerNorm
    block1_conv = resnet.block1.block[0]
    block1_norm = resnet.block1.block[2]

    # Conv1d weight: PyTorch [out, in, kernel] -> MLX [out, kernel, in]
    # Swift expects .conv.conv.weight (nested Conv1d wrapper)
    key = f"{prefix}.block1.conv.conv.weight"
    w = block1_conv.weight.detach().float()
    weights[key] = w.transpose(1, 2).contiguous()
    added += 1

    key = f"{prefix}.block1.conv.conv.bias"
    weights[key] = block1_conv.bias.detach().float()
    added += 1

    key = f"{prefix}.block1.norm.weight"
    weights[key] = block1_norm.weight.detach().float()
    added += 1

    key = f"{prefix}.block1.norm.bias"
    weights[key] = block1_norm.bias.detach().float()
    added += 1

    # block2: Sequential(Conv1d, SiLU, LayerNorm)
    block2_conv = resnet.block2.block[0]
    block2_norm = resnet.block2.block[2]

    key = f"{prefix}.block2.conv.conv.weight"
    w = block2_conv.weight.detach().float()
    weights[key] = w.transpose(1, 2).contiguous()
    added += 1

    key = f"{prefix}.block2.conv.conv.bias"
    weights[key] = block2_conv.bias.detach().float()
    added += 1

    key = f"{prefix}.block2.norm.weight"
    weights[key] = block2_norm.weight.detach().float()
    added += 1

    key = f"{prefix}.block2.norm.bias"
    weights[key] = block2_norm.bias.detach().float()
    added += 1

    # mlp: Sequential(SiLU, Linear)
    # mlp[1] = Linear
    mlp_linear = resnet.mlp[1]

    # Linear weight: keep in PyTorch format [out, in] - Swift will transpose during loading
    key = f"{prefix}.mlp_linear.weight"
    weights[key] = mlp_linear.weight.detach().float()
    added += 1

    key = f"{prefix}.mlp_linear.bias"
    weights[key] = mlp_linear.bias.detach().float()
    added += 1

    return added

if __name__ == "__main__":
    main()
