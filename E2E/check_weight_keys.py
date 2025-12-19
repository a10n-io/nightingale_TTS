#!/Users/a10n/Projects/nightingale_TTS/python/venv/bin/python
"""Check what keys exist in the model weight files."""

from safetensors import safe_open
from pathlib import Path

MODEL_DIR = Path("/Users/a10n/Projects/nightingale_TTS/models/mlx")

def check_keys(filepath, pattern):
    """Print keys matching pattern."""
    print(f"\n{'='*60}")
    print(f"File: {filepath.name}")
    print(f"Pattern: {pattern}")
    print(f"{'='*60}")

    try:
        with safe_open(filepath, framework="pt") as f:
            keys = list(f.keys())
            matching = [k for k in keys if pattern in k]
            print(f"Total keys: {len(keys)}")
            print(f"Matching '{pattern}': {len(matching)}")
            for k in sorted(matching)[:30]:
                tensor = f.get_tensor(k)
                print(f"  {k}: {list(tensor.shape)}")
            if len(matching) > 30:
                print(f"  ... and {len(matching) - 30} more")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check all safetensor files for pos_enc.pe
    files = [
        "chatterbox_hf.safetensors",
        "s3gen_fp16.safetensors",
        "python_flow_weights.safetensors",
    ]

    for fname in files:
        fpath = MODEL_DIR / fname
        if fpath.exists():
            print("\n" + "="*60)
            print(f"FILE: {fname}")
            print("="*60)
            with safe_open(fpath, framework="pt") as f:
                for k in sorted(f.keys()):
                    if "pos_enc" in k.lower() or "pos_bias" in k.lower():
                        tensor = f.get_tensor(k)
                        print(f"  {k}: {list(tensor.shape)}")
