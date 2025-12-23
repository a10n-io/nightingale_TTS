"""
Check finalProj weights in Python to compare with Swift.
"""
from safetensors.torch import load_file

state_dict = load_file("/Users/a10n/Projects/nightingale_TTS/models/chatterbox/s3gen.safetensors")

print("=" * 80)
print("FINALPROJ WEIGHT CHECK")
print("=" * 80)

# Check finalProj weight
key_w = "flow.decoder.final_proj.weight"
key_b = "flow.decoder.final_proj.bias"

if key_w in state_dict:
    w = state_dict[key_w]
    print(f"\n{key_w}:")
    print(f"   Shape (PyTorch): {w.shape}")  # Should be [out, in, kernel]
    print(f"   Range: [{w.min().item():.6f}, {w.max().item():.6f}]")
    print(f"   Mean: {w.mean().item():.6f}")
    print(f"   Std: {w.std().item():.6f}")

    # Sample a few values
    print(f"   w[0,0,0]: {w[0, 0, 0].item():.6f}")
    print(f"   w[0,:5,0]: {w[0, :5, 0].tolist()}")
    print(f"   w[79,:5,0]: {w[79, :5, 0].tolist()}")
else:
    print(f"Key not found: {key_w}")

if key_b in state_dict:
    b = state_dict[key_b]
    print(f"\n{key_b}:")
    print(f"   Shape: {b.shape}")
    print(f"   Range: [{b.min().item():.6f}, {b.max().item():.6f}]")
    print(f"   Mean: {b.mean().item():.6f}")
    print(f"   Std: {b.std().item():.6f}")

    # Check if bias is very negative (could explain darkness)
    print(f"\n   Bias contribution to mel:")
    print(f"   If applied to all 80 mel bins, mean shift = {b.mean().item():.6f}")

    # Sample some bias values
    print(f"   b[:10]: {b[:10].tolist()}")
    print(f"   b[-10:]: {b[-10:].tolist()}")
else:
    print(f"Key not found: {key_b}")

print("\n" + "=" * 80)
