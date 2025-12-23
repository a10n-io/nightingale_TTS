"""
Check what the actual Python w1 weight sum should be
"""
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"

# Load Python weights
flow_weights = load_file(str(MODELS_DIR / "s3gen.safetensors"))

# Get encoder.encoders.0.feed_forward.w_1.weight
w1_weight = flow_weights["flow.encoder.encoders.0.feed_forward.w_1.weight"]

print("=" * 80)
print("PYTHON encoder.encoders.0.feed_forward.w_1.weight")
print("=" * 80)
print(f"Shape: {w1_weight.shape}")
print(f"Sum: {w1_weight.sum().item():.6f}")
print(f"Mean: {w1_weight.mean().item():.6f}")
print(f"Std: {w1_weight.std().item():.6f}")
print(f"Range: [{w1_weight.min().item():.6f}, {w1_weight.max().item():.6f}]")
