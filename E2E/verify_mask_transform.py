#!/usr/bin/env python3
"""Verify the mask transformation for attention."""

import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

# Load mask
mask = torch.from_numpy(np.load(ref_dir / "step7_cond_T.npy")[[0]])  # [1, 1, 696]
mask_2d = mask.squeeze(1)  # [1, 696]

print("="*80)
print("MASK TRANSFORMATION FOR ATTENTION")
print("="*80)

print(f"\nOriginal mask shape: {mask.shape}")  # [1, 1, 696]
print(f"mask_2d shape: {mask_2d.shape}")  # [1, 696]
print(f"mask_2d sum: {mask_2d.sum().item()} (number of valid positions)")
print(f"mask_2d range: [{mask_2d.min().item()}, {mask_2d.max().item()}]")

# Python attention code expands mask like this:
mask_expanded = mask_2d.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
print(f"\nmask_expanded shape: {mask_expanded.shape}")  # [1, 1, 1, 696]

# Create attention scores (dummy)
B, H, T = 1, 8, 696
scores = torch.randn(B, H, T, T)
print(f"\nScores shape: {scores.shape}")  # [1, 8, 696, 696]

# Apply mask (Python way)
scores_masked = scores.masked_fill(mask_expanded == 0, -1e9)
print(f"Scores before mask: range=[{scores.min():.4f}, {scores.max():.4f}]")
print(f"Scores after mask: range=[{scores_masked.min():.4f}, {scores_masked.max():.4f}]")

# For Swift additive bias approach, we need:
# - A bias mask of shape [B, 1, 1, T] or [B, 1, T, T]
# - Where mask_2d == 0, bias = -1e9
# - Where mask_2d == 1, bias = 0

# Option 1: [B, 1, 1, T] - broadcasts across query dimension
bias_mask_1 = torch.where(mask_expanded == 0, torch.tensor(-1e9), torch.tensor(0.0))
print(f"\nOption 1 - Bias mask shape: {bias_mask_1.shape}")  # [1, 1, 1, 696]
print(f"Bias mask range: [{bias_mask_1.min().item()}, {bias_mask_1.max().item()}]")

# Verify it produces same result
scores_with_bias = scores + bias_mask_1
print(f"Scores + bias: range=[{scores_with_bias.min():.4f}, {scores_with_bias.max():.4f}]")
print(f"Match: {torch.allclose(scores_masked, scores_with_bias)}")

# Save the bias mask for Swift
np.save(ref_dir / "tfmr_bias_mask.npy", bias_mask_1.numpy())
print(f"\n✅ Saved bias mask to tfmr_bias_mask.npy")

print("="*80)
