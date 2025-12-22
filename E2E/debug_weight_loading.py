"""Debug weight loading for mSource."""
from safetensors.torch import load_file
from pathlib import Path

weights = load_file('models/mlx/vocoder_weights.safetensors')

# Check what keys would match for m_source
print("Looking for m_source keys:")
for key, value in weights.items():
    if 'm_source' in key:
        print(f"  Found: {key}")
        print(f"    Shape: {value.shape}")
        print(f"    Values: {value.flatten().tolist()}")
        
# Check Swift remapping logic
def remap_key(key):
    k = key
    if k.contains("m_source."): 
        k = k.replace("m_source.l_linear.", "vocoder.mSource.linear.")
        return k
    return None

# Simulate what Swift would do
print("\nSimulating Swift key remapping:")
for key in weights.keys():
    if "m_source" in key:
        remapped = key.replace("m_source.l_linear.", "vocoder.mSource.linear.")
        print(f"  {key} -> {remapped}")
