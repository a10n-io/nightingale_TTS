#!/usr/bin/env python3
"""Save Python mean and variance for comparison."""

import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

# Load input
input_data = torch.from_numpy(np.load(ref_dir / "tfmr_trace_input.npy"))

print("Input:", input_data.shape)

# Compute mean and variance over last dimension
mean = input_data.mean(dim=-1, keepdim=True)
var = input_data.var(dim=-1, unbiased=False, keepdim=True)

print(f"Mean: {mean.shape}, range=[{mean.min():.6f}, {mean.max():.6f}]")
print(f"Variance: {var.shape}, range=[{var.min():.6f}, {var.max():.6f}]")

# Save for Swift comparison
np.save(ref_dir / "tfmr_ln_mean.npy", mean.numpy())
np.save(ref_dir / "tfmr_ln_var.npy", var.numpy())

print("\nSaved mean and variance reference files")
