#!/usr/bin/env python3
"""Check the actual input data."""

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

# Load input
input_data = np.load(ref_dir / "tfmr_trace_input.npy")

print(f"Input shape: {input_data.shape}")
print(f"Input dtype: {input_data.dtype}")

# Check first sequence
first_seq = input_data[0, 0, :]  # [256]
print(f"\nFirst sequence shape: {first_seq.shape}")
print(f"First sequence range: [{first_seq.min():.6f}, {first_seq.max():.6f}]")
print(f"First sequence mean: {first_seq.mean():.6f}")
print(f"First 10 values: {first_seq[:10]}")
print(f"Are all values the same? {np.all(first_seq == first_seq[0])}")

# Check another sequence
second_seq = input_data[0, 1, :]  # [256]
print(f"\nSecond sequence range: [{second_seq.min():.6f}, {second_seq.max():.6f}]")
print(f"Second sequence mean: {second_seq.mean():.6f}")

# Check overall input
print(f"\nOverall input range: [{input_data.min():.6f}, {input_data.max():.6f}]")
print(f"Overall input mean: {input_data.mean():.6f}")
