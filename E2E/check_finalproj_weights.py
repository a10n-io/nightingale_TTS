#!/usr/bin/env python3
"""Check Python finalProj weights."""
import torch
from pathlib import Path
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Load model
MODELS_DIR = Path("models/chatterbox")
device = "cpu"

print("Loading Chatterbox model...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

decoder = model.s3gen.flow.decoder.estimator

# Check finalProj weights
fp_weight = decoder.final_proj.weight  # Should be [80, 256, 1]
fp_bias = decoder.final_proj.bias  # Should be [80]

print(f"\nfinalProj.weight:")
print(f"  shape: {fp_weight.shape}")
print(f"  range: [{fp_weight.min().item():.6f}, {fp_weight.max().item():.6f}]")
print(f"  mean: {fp_weight.mean().item():.6f}")
print(f"  First 5 weights of channel 0: {fp_weight[0, :5, 0].tolist()}")

print(f"\nfinalProj.bias:")
print(f"  shape: {fp_bias.shape}")
print(f"  range: [{fp_bias.min().item():.6f}, {fp_bias.max().item():.6f}]")
print(f"  mean: {fp_bias.mean().item():.6f}")
print(f"  First 10: {fp_bias[:10].tolist()}")

print("\n✅ Done")
