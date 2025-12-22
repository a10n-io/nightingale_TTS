#!/usr/bin/env python3
"""Test variance formula used by PyTorch LayerNorm."""

import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

# Load input
input_data = torch.from_numpy(np.load(ref_dir / "tfmr_trace_input.npy"))

print("="*80)
print("VARIANCE FORMULA TEST")
print("="*80)

print(f"\nInput: {input_data.shape}")

# Test different variance formulas
print("\n1. PyTorch variance (unbiased=False, divides by N):")
var_unbiased_false = input_data.var(dim=-1, unbiased=False, keepdim=True)
print(f"  Range: [{var_unbiased_false.min():.6f}, {var_unbiased_false.max():.6f}]")

print("\n2. PyTorch variance (unbiased=True, divides by N-1):")
var_unbiased_true = input_data.var(dim=-1, unbiased=True, keepdim=True)
print(f"  Range: [{var_unbiased_true.min():.6f}, {var_unbiased_true.max():.6f}]")

# Compute LayerNorm manually with both
eps = 1e-5
mean = input_data.mean(dim=-1, keepdim=True)

print("\n3. Normalized with unbiased=False:")
norm_false = (input_data - mean) / torch.sqrt(var_unbiased_false + eps)
print(f"  Range: [{norm_false.min():.4f}, {norm_false.max():.4f}]")

print("\n4. Normalized with unbiased=True:")
norm_true = (input_data - mean) / torch.sqrt(var_unbiased_true + eps)
print(f"  Range: [{norm_true.min():.4f}, {norm_true.max():.4f}]")

# Check what PyTorch LayerNorm actually uses
print("\n5. PyTorch LayerNorm (to see which it matches):")
ln = torch.nn.LayerNorm(256, eps=1e-5)
ln.weight.data = torch.ones(256)
ln.bias.data = torch.zeros(256)
with torch.no_grad():
    ln_output = ln(input_data)
print(f"  Range: [{ln_output.min():.4f}, {ln_output.max():.4f}]")

if torch.allclose(ln_output, norm_false):
    print("  ✅ LayerNorm uses unbiased=False (divides by N)")
elif torch.allclose(ln_output, norm_true):
    print("  ✅ LayerNorm uses unbiased=True (divides by N-1)")
else:
    print("  ❓ LayerNorm uses neither?")

print("="*80)
