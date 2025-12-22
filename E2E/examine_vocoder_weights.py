#!/usr/bin/env python3
"""Examine vocoder weight structure for key remapping."""
import safetensors.torch as st
from pathlib import Path
from collections import defaultdict

# Load s3gen weights
s3gen_path = Path("models/chatterbox/s3gen.safetensors")
weights = st.load_file(s3gen_path)

# Find all vocoder keys
vocoder_keys = [k for k in weights.keys() if "mel2wav" in k]
print(f"Found {len(vocoder_keys)} vocoder keys\n")

# Group by structure
groups = defaultdict(list)
for key in sorted(vocoder_keys):
    shape = list(weights[key].shape)

    # Extract base structure
    if "parametrizations" in key:
        # Extract what's being parametrized
        parts = key.split(".")
        param_idx = parts.index("parametrizations")
        base = ".".join(parts[:param_idx+2])  # e.g., mel2wav.resblocks.0.convs1.0.parametrizations
        groups["weight_norm"].append(f"{key}: {shape}")
    elif "conv_pre" in key:
        groups["conv_pre"].append(f"{key}: {shape}")
    elif "conv_post" in key:
        groups["conv_post"].append(f"{key}: {shape}")
    elif "f0_predictor" in key:
        groups["f0_predictor"].append(f"{key}: {shape}")
    elif "resblocks" in key:
        groups["resblocks"].append(f"{key}: {shape}")
    elif "ups" in key:
        groups["ups"].append(f"{key}: {shape}")
    else:
        groups["other"].append(f"{key}: {shape}")

# Print groups
for group_name, keys in sorted(groups.items()):
    print(f"\n=== {group_name} ({len(keys)} keys) ===")
    for key in keys[:10]:  # Show first 10
        print(f"  {key}")
    if len(keys) > 10:
        print(f"  ... and {len(keys) - 10} more")
