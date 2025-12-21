#!/usr/bin/env python3
"""Find weights that exist in Python but are missing from safetensors."""

import torch
from pathlib import Path
import sys
from safetensors import safe_open

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
FLOW_WEIGHTS_FILE = PROJECT_ROOT / "models" / "mlx" / "python_flow_weights.safetensors"

def main():
    print("=" * 80)
    print("FINDING MISSING WEIGHTS IN SAFETENSORS")
    print("=" * 80)

    # Load Python model
    sys.path.insert(0, str(PROJECT_ROOT / "python"))
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model_dir = PROJECT_ROOT / "models" / "chatterbox"
    model = ChatterboxMultilingualTTS.from_local(str(model_dir), device="cpu")

    # Get decoder state dict
    py_state = model.s3gen.state_dict()
    decoder_keys = [k for k in py_state.keys() if "decoder" in k or "flow.decoder" in k]

    # Load safetensors
    with safe_open(FLOW_WEIGHTS_FILE, framework="pt", device="cpu") as f:
        st_keys = set(f.keys())

    # Find the Python keys that have no corresponding safetensors key
    missing = []
    for py_key in decoder_keys:
        # Check various possible mappings
        found = False
        for st_key in st_keys:
            if is_match(py_key, st_key):
                found = True
                break
        if not found:
            missing.append(py_key)

    print(f"\nPython decoder keys: {len(decoder_keys)}")
    print(f"Safetensors keys: {len(st_keys)}")
    print(f"Missing: {len(missing)}")

    if missing:
        print(f"\n❌ Missing weights ({len(missing)} total):")
        # Group by pattern
        grouped = {}
        for k in missing:
            parts = k.split(".")
            if "mlp" in k:
                key = "mlp"
            elif "block1" in k:
                key = "block1"
            elif "block2" in k:
                key = "block2"
            else:
                key = "other"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(k)

        for group_name, keys in grouped.items():
            print(f"\n  [{group_name}] ({len(keys)} keys):")
            for k in keys[:5]:
                tensor = py_state[k]
                print(f"    {k}: {tensor.shape}")
            if len(keys) > 5:
                print(f"    ... and {len(keys) - 5} more")

def is_match(py_key, st_key):
    """Check if a Python key matches a safetensors key."""
    # Normalize both keys
    py_norm = normalize_py(py_key)
    st_norm = normalize_st(st_key)
    return py_norm == st_norm

def normalize_py(key):
    """Normalize Python key to common format."""
    import re

    # Remove prefix
    key = key.replace("flow.decoder.estimator.", "")

    # Normalize block indices
    key = re.sub(r'down_blocks\.(\d+)\.1\.(\d+)', r'down_blocks.\1.transformer.\2', key)
    key = re.sub(r'mid_blocks\.(\d+)\.1\.(\d+)', r'mid_blocks.\1.transformer.\2', key)
    key = re.sub(r'up_blocks\.(\d+)\.1\.(\d+)', r'up_blocks.\1.transformer.\2', key)
    key = re.sub(r'down_blocks\.(\d+)\.0\.', r'down_blocks.\1.resnet.', key)
    key = re.sub(r'mid_blocks\.(\d+)\.0\.', r'mid_blocks.\1.resnet.', key)
    key = re.sub(r'up_blocks\.(\d+)\.0\.', r'up_blocks.\1.resnet.', key)
    key = re.sub(r'down_blocks\.(\d+)\.2\.', r'down_blocks.\1.downsample.', key)
    key = re.sub(r'up_blocks\.(\d+)\.2\.', r'up_blocks.\1.upsample.', key)

    # Normalize attention
    key = key.replace("attn1.to_q.", "attn.query_proj.")
    key = key.replace("attn1.to_k.", "attn.key_proj.")
    key = key.replace("attn1.to_v.", "attn.value_proj.")
    key = key.replace("attn1.to_out.0.", "attn.out_proj.")

    # Normalize FF
    key = key.replace("ff.net.0.proj.", "ff.layers.0.")
    key = key.replace("ff.net.2.", "ff.layers.1.")

    return key

def normalize_st(key):
    """Normalize safetensors key to common format."""
    import re

    # Remove prefix
    key = key.replace("decoder.", "")

    # Normalize block indices
    key = re.sub(r'down_blocks_(\d+)\.transformer_(\d+)', r'down_blocks.\1.transformer.\2', key)
    key = re.sub(r'mid_blocks_(\d+)\.transformer_(\d+)', r'mid_blocks.\1.transformer.\2', key)
    key = re.sub(r'up_blocks_(\d+)\.transformer_(\d+)', r'up_blocks.\1.transformer.\2', key)
    key = re.sub(r'down_blocks_(\d+)\.resnet\.', r'down_blocks.\1.resnet.', key)
    key = re.sub(r'mid_blocks_(\d+)\.resnet\.', r'mid_blocks.\1.resnet.', key)
    key = re.sub(r'up_blocks_(\d+)\.resnet\.', r'up_blocks.\1.resnet.', key)
    key = re.sub(r'down_blocks_(\d+)\.downsample\.', r'down_blocks.\1.downsample.', key)
    key = re.sub(r'up_blocks_(\d+)\.upsample\.', r'up_blocks.\1.upsample.', key)

    # Normalize attention
    key = key.replace(".attn.query_proj.", ".attn.query_proj.")
    key = key.replace(".attn.key_proj.", ".attn.key_proj.")
    key = key.replace(".attn.value_proj.", ".attn.value_proj.")
    key = key.replace(".attn.out_proj.", ".attn.out_proj.")

    return key

if __name__ == "__main__":
    main()
