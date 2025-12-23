"""
Check Conv1d weight shapes in encoder
"""
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"

# Load Python weights
flow_weights = load_file(str(MODELS_DIR / "s3gen.safetensors"))

print("=" * 80)
print("ENCODER CONV1D WEIGHT SHAPES")
print("=" * 80)

# Check pre_lookahead_layer Conv1d
if "flow.encoder.pre_lookahead_layer.conv1.weight" in flow_weights:
    w = flow_weights["flow.encoder.pre_lookahead_layer.conv1.weight"]
    print(f"\nflow.encoder.pre_lookahead_layer.conv1.weight")
    print(f"  Shape: {w.shape}")
    print(f"  PyTorch Conv1d format: [out_channels, in_channels, kernel_size]")

if "flow.encoder.pre_lookahead_layer.conv2.weight" in flow_weights:
    w = flow_weights["flow.encoder.pre_lookahead_layer.conv2.weight"]
    print(f"\nflow.encoder.pre_lookahead_layer.conv2.weight")
    print(f"  Shape: {w.shape}")

# Check up_layer Conv1d
if "flow.encoder.up_layer.conv.weight" in flow_weights:
    w = flow_weights["flow.encoder.up_layer.conv.weight"]
    print(f"\nflow.encoder.up_layer.conv.weight")
    print(f"  Shape: {w.shape}")
    print(f"  PyTorch Conv1d format: [out_channels, in_channels, kernel_size]")

print("\n" + "=" * 80)
print("WHAT SWIFT EXPECTS")
print("=" * 80)
print("MLX Conv1d format: [out_channels, kernel_size, in_channels]")
print("So we need to transpose from [out, in, kernel] -> [out, kernel, in]")
print("This is: .transposed(0, 2, 1)")
