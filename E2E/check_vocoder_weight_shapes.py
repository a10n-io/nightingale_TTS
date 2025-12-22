#!/usr/bin/env python3
"""Check vocoder weight shapes to verify transposition logic."""
import safetensors.torch as st
from pathlib import Path

# Load s3gen weights
s3gen_path = Path("models/chatterbox/s3gen.safetensors")
weights = st.load_file(s3gen_path)

print("="*80)
print("VOCODER WEIGHT SHAPES - Verify Transposition Logic")
print("="*80)

# Check conv_pre (Conv1d)
print("\n=== conv_pre (Conv1d) ===")
for key in sorted(weights.keys()):
    if "mel2wav.conv_pre" in key and "parametrizations" not in key:
        print(f"  {key}: {list(weights[key].shape)}")

# Check conv_post (Conv1d)
print("\n=== conv_post (Conv1d) ===")
for key in sorted(weights.keys()):
    if "mel2wav.conv_post" in key and "parametrizations" not in key:
        print(f"  {key}: {list(weights[key].shape)}")

# Check ups (ConvTranspose1d)
print("\n=== ups (ConvTranspose1d) ===")
for key in sorted(weights.keys()):
    if "mel2wav.ups" in key and "parametrizations" not in key:
        print(f"  {key}: {list(weights[key].shape)}")

# Check weight_norm parametrizations
print("\n=== weight_norm parametrizations ===")
print("conv_pre:")
for key in sorted(weights.keys()):
    if "mel2wav.conv_pre.parametrizations" in key:
        print(f"  {key}: {list(weights[key].shape)}")

print("\nconv_post:")
for key in sorted(weights.keys()):
    if "mel2wav.conv_post.parametrizations" in key:
        print(f"  {key}: {list(weights[key].shape)}")

print("\nups.0:")
for key in sorted(weights.keys()):
    if "mel2wav.ups.0.parametrizations" in key:
        print(f"  {key}: {list(weights[key].shape)}")

print("\nresblocks.0.convs1.0:")
for key in sorted(weights.keys()):
    if "mel2wav.resblocks.0.convs1.0.parametrizations" in key:
        print(f"  {key}: {list(weights[key].shape)}")

# Check f0_predictor
print("\n=== f0_predictor ===")
print("Linear (classifier):")
for key in sorted(weights.keys()):
    if "f0_predictor.classifier" in key and "parametrizations" not in key:
        print(f"  {key}: {list(weights[key].shape)}")

print("\nConv1d (condnet - with weight_norm):")
for key in sorted(weights.keys()):
    if "f0_predictor.condnet.0.parametrizations" in key:
        print(f"  {key}: {list(weights[key].shape)}")

# Check source components
print("\n=== source components ===")
print("sourceDowns.0 (Conv1d - with weight_norm):")
for key in sorted(weights.keys()):
    if "source_downs.0" in key:
        print(f"  {key}: {list(weights[key].shape)}")

print("\nmSource.linear (Linear):")
for key in sorted(weights.keys()):
    if "m_source.l_linear" in key:
        print(f"  {key}: {list(weights[key].shape)}")

print("\n" + "="*80)
print("EXPECTED TRANSPOSITIONS:")
print("="*80)
print("Conv1d: PyTorch [Out, In, Kernel] -> MLX [Out, Kernel, In]")
print("  Example: [512, 80, 7] -> [512, 7, 80]")
print("\nConvTranspose1d: PyTorch [In, Out, Kernel] -> MLX [Out, Kernel, In]")
print("  Example: [512, 256, 16] -> [256, 16, 512]")
print("\nLinear: PyTorch [Out, In] -> MLX [In, Out]")
print("  Example: [1, 512] -> [512, 1]")
print("\nweight_norm: combine original0 * (original1 / ||original1||)")
print("  original0: [Out, 1, 1] (magnitude)")
print("  original1: [Out, In, Kernel] (direction)")
print("  combined: [Out, In, Kernel] then transpose as Conv1d/ConvT")
