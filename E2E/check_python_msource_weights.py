"""
Check Python's m_source.l_linear weights for comparison with Swift
"""
import torch
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"

# Load vocoder weights
vocoder_weights = load_file(str(MODELS_DIR / "s3gen.safetensors"))

# Find m_source.l_linear weights
msource_weight_key = None
msource_bias_key = None

for key in vocoder_weights.keys():
    if "m_source" in key and "linear" in key:
        if "weight" in key:
            msource_weight_key = key
        elif "bias" in key:
            msource_bias_key = key

print("=" * 80)
print("PYTHON m_source.l_linear WEIGHTS CHECK")
print("=" * 80)

if msource_weight_key:
    weight = vocoder_weights[msource_weight_key]
    print(f"\nFound key: {msource_weight_key}")
    print(f"  Shape: {weight.shape}")
    print(f"  Mean (absolute): {weight.abs().mean().item():.6f}")
    print(f"  Std: {weight.std().item():.6f}")
    print(f"  First 5 weights: {weight.flatten()[:5].tolist()}")
else:
    print("\n❌ m_source linear weight not found!")
    print("Available keys with 'm_source':")
    for key in sorted(vocoder_weights.keys()):
        if "m_source" in key or "source" in key:
            print(f"  {key}")

if msource_bias_key:
    bias = vocoder_weights[msource_bias_key]
    print(f"\nFound key: {msource_bias_key}")
    print(f"  Shape: {bias.shape}")
    print(f"  Values: {bias.tolist()}")
else:
    print("\n❌ m_source linear bias not found!")

print("\n" + "=" * 80)
print("EXPECTED SWIFT VALUES (if loading correctly):")
print("=" * 80)
if msource_weight_key:
    weight = vocoder_weights[msource_weight_key]
    print(f"  Shape: {list(reversed(weight.shape))} (transposed for MLX)")
    print(f"  Mean (absolute): {weight.abs().mean().item():.6f}")
    print(f"  Std: {weight.std().item():.6f}")
print("=" * 80)
