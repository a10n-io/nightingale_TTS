#!/usr/bin/env python3
"""Test what the Python encoder actually outputs with the trained weights."""

import torch
import numpy as np

# Load the PyTorch model
print("Loading PyTorch S3Gen model...")
model_path = '/Users/a10n/Projects/nightingale_TTS/models/chatterbox/s3gen.pt'
state_dict = torch.load(model_path, map_location='cpu')

# Check encoder.embed.out.1 (LayerNorm) weights
weight = state_dict['flow.encoder.embed.out.1.weight']
print(f"\nEncoder LayerNorm weight: mean={weight.mean().item():.6f}, std={weight.std().item():.6f}")
print(f"  This is CORRECT - these are the trained values")
print(f"  The small values are intentional!")

# Load a reference encoder output from the saved runs
ref_dir = '/Users/a10n/Projects/nightingale_TTS/E2E/reference_outputs/samantha/expressive_surprise_en'

# The Python reference run should have saved the encoder output
# Let me check what we have
import os
files = os.listdir(ref_dir)
encoder_files = [f for f in files if 'encoder' in f.lower() or 'step6' in f]
print(f"\nEncoder-related files in reference_outputs:")
for f in sorted(encoder_files):
    print(f"  {f}")

# Load step6_mu.npy which is the encoder output
mu = np.load(f'{ref_dir}/step6_mu.npy')
print(f"\nPython encoder output (mu):")
print(f"  Shape: {mu.shape}")
print(f"  Mean: {mu.mean():.6f}")
print(f"  Std: {mu.std():.6f}")
print(f"  Range: [{mu.min():.6f}, {mu.max():.6f}]")

# Check the frequency structure of the encoder output
print(f"\nEncoder output channel energies:")
for i in [0, 10, 20, 30, 40, 50, 60, 70, 79]:
    energy = mu[0, :, i].mean()
    print(f"  Channel {i:2d}: {energy:.4f}")
