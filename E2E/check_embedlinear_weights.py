"""
Compare embedLinear weight VALUES between Python file and what Swift loaded
"""
import torch
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"

# Load Python weights from file
flow_weights = load_file(str(MODELS_DIR / "s3gen.safetensors"))
python_weight = flow_weights["flow.encoder.embed.out.0.weight"]  # [512, 512]

print("=" * 80)
print("PYTHON EMBEDLINEAR WEIGHT (from file)")
print("=" * 80)
print(f"Shape: {python_weight.shape}")
print(f"PyTorch Linear format: [out_features, in_features] = [512, 512]")
print(f"Sum: {python_weight.sum().item():.6f}")
print(f"Mean: {python_weight.mean().item():.6f}")
print(f"Std: {python_weight.std().item():.6f}")
print(f"Range: [{python_weight.min().item():.6f}, {python_weight.max().item():.6f}]")
print(f"\nFirst 5 values [0, :5]: {python_weight[0, :5].tolist()}")
print(f"First 5 values [:5, 0]: {python_weight[:5, 0].tolist()}")

print("\n" + "=" * 80)
print("WHAT SWIFT SHOULD HAVE AFTER TRANSPOSE")
print("=" * 80)
print("MLX Linear format: [in_features, out_features]")
print("So we TRANSPOSE: [512, 512] -> [512, 512]")
transposed = python_weight.T
print(f"Transposed sum: {transposed.sum().item():.6f} (should match Python: 6.151423)")
print(f"\nTransposed first 5 values [0, :5]: {transposed[0, :5].tolist()}")
print(f"Transposed first 5 values [:5, 0]: {transposed[:5, 0].tolist()}")
