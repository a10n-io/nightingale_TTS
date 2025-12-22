#!/usr/bin/env python3
"""Test LayerNorm parity between PyTorch and manual computation."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

# Load the Python-generated input
input_data = torch.from_numpy(np.load(ref_dir / "tfmr_trace_input.npy"))  # [1, 696, 256]

# Load LayerNorm weights
model_path = PROJECT_ROOT / "models" / "chatterbox" / "s3gen.pt"
state_dict = torch.load(str(model_path), map_location='cpu')
norm1_weight = state_dict['flow.decoder.estimator.down_blocks.0.1.0.norm1.weight']
norm1_bias = state_dict['flow.decoder.estimator.down_blocks.0.1.0.norm1.bias']

print("="*80)
print("LAYERNORM PARITY TEST")
print("="*80)

print(f"\nInput shape: {input_data.shape}")
print(f"Input range: [{input_data.min():.4f}, {input_data.max():.4f}]")

# Test 1: PyTorch LayerNorm
print(f"\n1. PyTorch LayerNorm (epsilon=1e-5):")
norm_pytorch = nn.LayerNorm(256, eps=1e-5)
norm_pytorch.weight.data = norm1_weight
norm_pytorch.bias.data = norm1_bias
with torch.no_grad():
    output_pytorch = norm_pytorch(input_data)
print(f"  Output range: [{output_pytorch.min():.4f}, {output_pytorch.max():.4f}]")

# Test 2: Manual LayerNorm computation (epsilon=1e-5)
print(f"\n2. Manual LayerNorm (epsilon=1e-5):")
eps = 1e-5
mean = input_data.mean(dim=-1, keepdim=True)
var = input_data.var(dim=-1, unbiased=False, keepdim=True)
normalized = (input_data - mean) / torch.sqrt(var + eps)
output_manual = normalized * norm1_weight + norm1_bias
print(f"  Output range: [{output_manual.min():.4f}, {output_manual.max():.4f}]")
print(f"  Difference from PyTorch: max abs = {(output_manual - output_pytorch).abs().max():.6f}")

# Test 3: Try different epsilon values
print(f"\n3. Testing different epsilon values:")
for eps_test in [1e-6, 1e-5, 1e-4, 1e-3]:
    norm_test = nn.LayerNorm(256, eps=eps_test)
    norm_test.weight.data = norm1_weight
    norm_test.bias.data = norm1_bias
    with torch.no_grad():
        output_test = norm_test(input_data)
    print(f"  eps={eps_test}: range=[{output_test.min():.4f}, {output_test.max():.4f}]")

# Check the variance and mean after normalization
print(f"\n4. Normalized statistics (before weight/bias):")
mean = input_data.mean(dim=-1, keepdim=True)
var = input_data.var(dim=-1, unbiased=False, keepdim=True)
normalized = (input_data - mean) / torch.sqrt(var + 1e-5)
print(f"  Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
print(f"  Normalized mean: {normalized.mean():.6f} (should be ~0)")
print(f"  Normalized var: {normalized.var(unbiased=False):.6f} (should be ~1)")

print("="*80)
