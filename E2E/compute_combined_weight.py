"""
Manually compute combined weight_norm weight to compare with Swift.
"""
import torch
from safetensors.torch import load_file

state_dict = load_file("/Users/a10n/Projects/nightingale_TTS/models/chatterbox/s3gen.safetensors")

print("=" * 80)
print("COMPUTE COMBINED WEIGHT_NORM WEIGHT")
print("=" * 80)

# Get original0 and original1
key0 = "mel2wav.source_resblocks.0.convs1.0.parametrizations.weight.original0"
key1 = "mel2wav.source_resblocks.0.convs1.0.parametrizations.weight.original1"

original0 = state_dict[key0]  # [Out, 1, 1] magnitude
original1 = state_dict[key1]  # [Out, In, Kernel] direction

print(f"\noriginal0 shape: {original0.shape}")
print(f"original1 shape: {original1.shape}")

# Combine: weight = original0 * (original1 / ||original1||)
norm = torch.sqrt((original1 * original1).sum(dim=[1, 2], keepdim=True))
normalized = original1 / (norm + 1e-8)
combined = original0 * normalized

print(f"\ncombined shape: {combined.shape}")
print(f"combined [0,0,:5]: {combined[0, 0, :5].tolist()}")

# Transpose to MLX format
transposed = combined.permute(0, 2, 1)  # [Out, In, Kernel] -> [Out, Kernel, In]
print(f"\ntransposed shape: {transposed.shape}")
print(f"transposed [0,0,:5]: {transposed[0, 0, :5].tolist()}")

print("\n" + "=" * 80)
