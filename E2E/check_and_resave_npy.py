#!/usr/bin/env python3
"""Check NPY file format and resave if needed."""

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

# Load input
input_data = np.load(ref_dir / "tfmr_trace_input.npy")

print(f"Loaded shape: {input_data.shape}")
print(f"Loaded dtype: {input_data.dtype}")
print(f"Loaded order: {'C' if input_data.flags['C_CONTIGUOUS'] else 'F' if input_data.flags['F_CONTIGUOUS'] else 'neither'}")
print(f"First 10 values: {input_data[0, 0, :10]}")

# Resave with explicit C-order and no allow_pickle
print("\nResaving with C-order...")
input_c_order = np.ascontiguousarray(input_data)
np.save(ref_dir / "tfmr_trace_input_c_order.npy", input_c_order)

# Verify
reloaded = np.load(ref_dir / "tfmr_trace_input_c_order.npy")
print(f"\nReloaded shape: {reloaded.shape}")
print(f"Reloaded dtype: {reloaded.dtype}")
print(f"Reloaded order: {'C' if reloaded.flags['C_CONTIGUOUS'] else 'F' if reloaded.flags['F_CONTIGUOUS'] else 'neither'}")
print(f"First 10 values: {reloaded[0, 0, :10]}")
print(f"Data matches: {np.allclose(input_data, reloaded)}")
