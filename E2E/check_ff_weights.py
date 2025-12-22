#!/usr/bin/env python3
"""Check FF layer weights in mid block 11, transformer 3 - where explosion happens."""

import torch
import numpy as np
from pathlib import Path
from safetensors import safe_open

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
MODELS_DIR = PROJECT_ROOT / "models" / "mlx"

def main():
    print("=" * 80)
    print("CHECKING FF LAYER WEIGHTS - MID BLOCK 11, TRANSFORMER 3")
    print("=" * 80)

    # Check python_flow_weights.safetensors
    python_flow_path = MODELS_DIR / "python_flow_weights.safetensors"
    if python_flow_path.exists():
        print(f"\n1. Python flow weights: {python_flow_path}")
        with safe_open(python_flow_path, framework="pt") as f:
            keys = list(f.keys())

            # Find mid_blocks_11.transformers_3.ff keys
            ff_keys = [k for k in keys if "mid_blocks_11" in k and "transformers_3" in k and "ff" in k]
            print(f"   Found {len(ff_keys)} FF keys for mid_blocks_11.transformers_3:")

            for k in sorted(ff_keys):
                tensor = f.get_tensor(k)
                print(f"     {k}")
                print(f"       shape: {tensor.shape}")
                print(f"       range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
                print(f"       mean: {tensor.mean().item():.4f}")
                if "weight" in k:
                    print(f"       first 5 values: {tensor.flatten()[:5].tolist()}")
    else:
        print(f"\n1. Python flow weights NOT FOUND: {python_flow_path}")

    # Check s3gen_fp16.safetensors
    s3gen_path = MODELS_DIR / "s3gen_fp16.safetensors"
    if s3gen_path.exists():
        print(f"\n2. S3Gen FP16 weights: {s3gen_path}")
        with safe_open(s3gen_path, framework="pt") as f:
            keys = list(f.keys())

            # Find flow.decoder.estimator.mid_blocks.11.transformers.3.ff keys
            ff_keys = [k for k in keys if "mid_blocks.11" in k and "transformers.3" in k and "ff" in k]
            print(f"   Found {len(ff_keys)} FF keys for mid_blocks.11.transformers.3:")

            for k in sorted(ff_keys):
                tensor = f.get_tensor(k)
                print(f"     {k}")
                print(f"       shape: {tensor.shape}")
                print(f"       range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
                print(f"       mean: {tensor.mean().item():.4f}")
                if "weight" in k:
                    print(f"       first 5 values: {tensor.flatten()[:5].tolist()}")
    else:
        print(f"\n2. S3Gen FP16 weights NOT FOUND: {s3gen_path}")

    # Also load the model and check actual weights
    print("\n3. Checking live model weights...")
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "python"))
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    model_dir = PROJECT_ROOT / "models" / "chatterbox"
    model = ChatterboxMultilingualTTS.from_local(str(model_dir), device="cpu")

    # Access mid_blocks[11].transformers[3].ff
    estimator = model.s3gen.flow.decoder.estimator
    print(f"\n   Estimator type: {type(estimator)}")
    print(f"   Estimator children: {list(estimator.named_children())[:5]}")

    mid_blocks = estimator.mid_blocks
    print(f"\n   mid_blocks type: {type(mid_blocks)}")
    print(f"   mid_blocks length: {len(mid_blocks)}")

    mid_block_11 = mid_blocks[11]
    print(f"\n   mid_block_11 type: {type(mid_block_11)}")
    print(f"   mid_block_11 length: {len(mid_block_11)}")

    # mid_block is a ModuleList of [resnet, transformers_list]
    resnet = mid_block_11[0]
    transformers_list = mid_block_11[1]
    print(f"\n   transformers_list type: {type(transformers_list)}")
    print(f"   transformers_list length: {len(transformers_list)}")

    transformer_3 = transformers_list[3]
    print(f"\n   transformer_3 type: {type(transformer_3)}")
    print(f"   transformer_3 children: {list(transformer_3.named_children())}")

    ff = transformer_3.ff

    print(f"\n   ff.net structure: {ff.net}")

    # ff.net is ModuleList with:
    # [0]: GELU (which has .proj = Linear 256->1024)
    # [1]: Dropout
    # [2]: LoRACompatibleLinear 1024->256
    gelu_layer = ff.net[0]
    linear1 = gelu_layer.proj  # Linear inside GELU
    linear2 = ff.net[2]  # Second linear layer

    print(f"\n   gelu_layer type: {type(gelu_layer)}")
    print(f"   linear1 (gelu_layer.proj) type: {type(linear1)}")

    print(f"\n   ff.net[0] (Linear1):")
    print(f"     weight shape: {linear1.weight.shape}")
    print(f"     weight range: [{linear1.weight.min().item():.4f}, {linear1.weight.max().item():.4f}]")
    print(f"     weight mean: {linear1.weight.mean().item():.4f}")
    print(f"     weight[:5,0]: {linear1.weight[:5, 0].tolist()}")
    if linear1.bias is not None:
        print(f"     bias shape: {linear1.bias.shape}")
        print(f"     bias range: [{linear1.bias.min().item():.4f}, {linear1.bias.max().item():.4f}]")

    print(f"\n   ff.net[2] (Linear2):")
    print(f"     weight shape: {linear2.weight.shape}")
    print(f"     weight range: [{linear2.weight.min().item():.4f}, {linear2.weight.max().item():.4f}]")
    print(f"     weight mean: {linear2.weight.mean().item():.4f}")
    if linear2.bias is not None:
        print(f"     bias shape: {linear2.bias.shape}")
        print(f"     bias range: [{linear2.bias.min().item():.4f}, {linear2.bias.max().item():.4f}]")

    # Save reference weights for comparison
    ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"
    np.save(ref_dir / "debug_ff_linear1_weight.npy", linear1.weight.detach().numpy())
    np.save(ref_dir / "debug_ff_linear2_weight.npy", linear2.weight.detach().numpy())
    if linear1.bias is not None:
        np.save(ref_dir / "debug_ff_linear1_bias.npy", linear1.bias.detach().numpy())
    if linear2.bias is not None:
        np.save(ref_dir / "debug_ff_linear2_bias.npy", linear2.bias.detach().numpy())
    print(f"\n   Saved reference FF weights to {ref_dir}")

if __name__ == "__main__":
    main()
