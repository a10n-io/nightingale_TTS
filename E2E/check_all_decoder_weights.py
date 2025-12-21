#!/usr/bin/env python3
"""Comprehensive check of all decoder weights vs safetensors."""

import torch
from pathlib import Path
import sys
from safetensors import safe_open

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
FLOW_WEIGHTS_FILE = PROJECT_ROOT / "models" / "mlx" / "python_flow_weights.safetensors"

def main():
    print("=" * 80)
    print("CHECKING ALL DECODER WEIGHTS VS SAFETENSORS")
    print("=" * 80)

    # Load Python model
    sys.path.insert(0, str(PROJECT_ROOT / "python"))
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model_dir = PROJECT_ROOT / "models" / "chatterbox"
    model = ChatterboxMultilingualTTS.from_local(str(model_dir), device="cpu")

    # Get decoder state dict
    py_state = model.s3gen.state_dict()

    # Load safetensors
    with safe_open(FLOW_WEIGHTS_FILE, framework="pt", device="cpu") as f:
        st_keys = set(f.keys())

        mismatches = []
        missing_in_st = []
        missing_in_py = []

        # Check all Python decoder keys
        for py_key in py_state.keys():
            if not ("decoder" in py_key or "flow.decoder" in py_key):
                continue

            # Map Python key to safetensors key
            # Python: flow.decoder.estimator.down_blocks.0.1.0.attn1.to_q.weight
            # Safetensors: decoder.down_blocks_0.transformer_0.attn.query_proj.weight
            st_key = map_py_to_st_key(py_key)

            if st_key is None:
                continue

            if st_key not in st_keys:
                missing_in_st.append((py_key, st_key))
                continue

            py_tensor = py_state[py_key]
            st_tensor = f.get_tensor(st_key)

            # Compare
            if py_tensor.shape != st_tensor.shape:
                mismatches.append((py_key, st_key, "shape mismatch", py_tensor.shape, st_tensor.shape))
            else:
                diff = (py_tensor - st_tensor).abs().max().item()
                if diff > 1e-5:
                    mismatches.append((py_key, st_key, "value mismatch", diff, None))

        # Report
        print(f"\nTotal Python decoder keys checked: {len([k for k in py_state if 'decoder' in k])}")
        print(f"Total safetensors keys: {len(st_keys)}")

        if mismatches:
            print(f"\n❌ Found {len(mismatches)} mismatches:")
            for item in mismatches[:20]:
                if item[2] == "shape mismatch":
                    print(f"  {item[0]}: {item[2]} {item[3]} vs {item[4]}")
                else:
                    print(f"  {item[0]}: {item[2]} diff={item[3]:.6f}")
        else:
            print("\n✅ All weights match!")

        if missing_in_st:
            print(f"\n⚠️  {len(missing_in_st)} keys missing in safetensors:")
            for py_key, st_key in missing_in_st[:10]:
                print(f"  {py_key} -> {st_key}")

def map_py_to_st_key(py_key):
    """Map Python state dict key to safetensors key format."""
    # Python format: flow.decoder.estimator.down_blocks.0.1.0.attn1.to_q.weight
    # Safetensors format: decoder.down_blocks_0.transformer_0.attn.query_proj.weight

    if "flow.decoder.estimator" not in py_key:
        return None

    # Remove prefix
    key = py_key.replace("flow.decoder.estimator.", "decoder.")

    # Handle block indexing (0.1.0 -> _0.transformer_0)
    import re

    # down_blocks.0.1.0 -> down_blocks_0.transformer_0
    key = re.sub(r'down_blocks\.(\d+)\.1\.(\d+)', r'down_blocks_\1.transformer_\2', key)
    # mid_blocks.0.1.0 -> mid_blocks_0.transformer_0
    key = re.sub(r'mid_blocks\.(\d+)\.1\.(\d+)', r'mid_blocks_\1.transformer_\2', key)
    # up_blocks.0.1.0 -> up_blocks_0.transformer_0
    key = re.sub(r'up_blocks\.(\d+)\.1\.(\d+)', r'up_blocks_\1.transformer_\2', key)

    # Handle resnet blocks
    # down_blocks.0.0 -> down_blocks_0.resnet
    key = re.sub(r'down_blocks\.(\d+)\.0\.', r'down_blocks_\1.resnet.', key)
    key = re.sub(r'mid_blocks\.(\d+)\.0\.', r'mid_blocks_\1.resnet.', key)
    key = re.sub(r'up_blocks\.(\d+)\.0\.', r'up_blocks_\1.resnet.', key)

    # Handle downsample/upsample
    key = re.sub(r'down_blocks\.(\d+)\.2\.', r'down_blocks_\1.downsample.', key)
    key = re.sub(r'up_blocks\.(\d+)\.2\.', r'up_blocks_\1.upsample.', key)

    # Handle attention projections
    key = key.replace("attn1.to_q.", "attn.query_proj.")
    key = key.replace("attn1.to_k.", "attn.key_proj.")
    key = key.replace("attn1.to_v.", "attn.value_proj.")
    key = key.replace("attn1.to_out.0.", "attn.out_proj.")

    # Handle FF
    key = key.replace("ff.net.0.proj.", "ff.layers.0.")
    key = key.replace("ff.net.2.", "ff.layers.1.")

    # final_block, final_proj
    key = re.sub(r'final_block\.(\d+)\.', r'final_block_\1.', key)

    return key

if __name__ == "__main__":
    main()
