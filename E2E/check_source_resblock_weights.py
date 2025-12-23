"""
Check source_resblocks weights in Python to compare with Swift.
"""
from safetensors.torch import load_file

state_dict = load_file("/Users/a10n/Projects/nightingale_TTS/models/chatterbox/s3gen.safetensors")

print("=" * 80)
print("SOURCE RESBLOCKS WEIGHT CHECK")
print("=" * 80)

# Check source_resblocks[0].convs1[0] weight
key = "mel2wav.source_resblocks.0.convs1.0.parametrizations.weight.original1"
if key in state_dict:
    w = state_dict[key]
    print(f"\n{key}:")
    print(f"   Shape (PyTorch): {w.shape}")  # Should be [out, in, kernel]
    print(f"   [0,0,:5]: {w[0, 0, :5].tolist()}")
    print(f"\n   After transpose to MLX [out, kernel, in]:")
    wt = w.permute(0, 2, 1)
    print(f"   Shape: {wt.shape}")
    print(f"   [0,0,:5]: {wt[0, 0, :5].tolist()}")
else:
    print(f"Key not found: {key}")

# Also check original0 (magnitude)
key0 = "mel2wav.source_resblocks.0.convs1.0.parametrizations.weight.original0"
if key0 in state_dict:
    w0 = state_dict[key0]
    print(f"\n{key0}:")
    print(f"   Shape: {w0.shape}")
    print(f"   First 5 values: {w0.flatten()[:5].tolist()}")

print("\n" + "=" * 80)
