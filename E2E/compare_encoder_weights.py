"""
Systematically compare encoder weights between Python and Swift.
Goal: Identify why Swift encoder produces outputs with half the variance and no correlation.
"""
import torch
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

print("=" * 80)
print("FORENSIC ENCODER WEIGHT COMPARISON")
print("=" * 80)

# Load model files
models_dir = PROJECT_ROOT / "models" / "chatterbox"
s3gen_path = models_dir / "s3gen.safetensors"

if not s3gen_path.exists():
    print(f"\n❌ Model file not found: {s3gen_path}")
    exit(1)

flow_data = load_file(str(s3gen_path))
print(f"\n✅ Loaded s3gen.safetensors ({len(flow_data)} keys)")

# Find encoder-related keys
print(f"\n🔍 Searching for encoder weights...")
encoder_keys = [k for k in flow_data.keys() if 'encoder' in k.lower() and 'decoder' not in k.lower()]

# Group by prefix
encoder_groups = {}
for key in encoder_keys:
    # Get the first component (e.g., "flow.encoder" or "flow.encoder_proj")
    parts = key.split('.')
    if len(parts) >= 2:
        prefix = '.'.join(parts[:2])
        if prefix not in encoder_groups:
            encoder_groups[prefix] = []
        encoder_groups[prefix].append(key)

print(f"\n📊 Encoder weight groups found:")
for prefix, keys in sorted(encoder_groups.items()):
    print(f"\n   {prefix}: {len(keys)} parameters")
    # Show first few keys
    for key in keys[:5]:
        weight = flow_data[key]
        print(f"     {key}")
        # Skip non-float tensors
        if weight.dtype in [torch.float32, torch.float16, torch.float64]:
            print(f"       Shape: {weight.shape}, Range: [{weight.min().item():.6f}, {weight.max().item():.6f}], Mean: {weight.mean().item():.6f}")
        else:
            print(f"       Shape: {weight.shape}, dtype: {weight.dtype}")
    if len(keys) > 5:
        print(f"     ... and {len(keys) - 5} more")

# Focus on encoder_proj (this is what we hooked in Python)
print(f"\n" + "=" * 80)
print("ENCODER_PROJ ANALYSIS")
print("=" * 80)

encoder_proj_keys = [k for k in flow_data.keys() if 'encoder_proj' in k.lower()]
print(f"\n📊 encoder_proj parameters ({len(encoder_proj_keys)} total):")

for key in sorted(encoder_proj_keys):
    weight = flow_data[key]
    print(f"\n   {key}")
    print(f"     Shape: {weight.shape}")
    print(f"     Range: [{weight.min().item():.8f}, {weight.max().item():.8f}]")
    print(f"     Mean: {weight.mean().item():.8f}")
    print(f"     Std: {weight.std().item():.8f}")

# Check the main encoder (UpsampleConformerEncoder or similar)
print(f"\n" + "=" * 80)
print("MAIN ENCODER ANALYSIS")
print("=" * 80)

main_encoder_keys = [k for k in flow_data.keys() if k.startswith('flow.encoder.') and 'proj' not in k]
print(f"\n📊 Main encoder parameters ({len(main_encoder_keys)} total)")

# Group by layer type
layer_types = {}
for key in main_encoder_keys:
    # Extract layer type (e.g., "embed", "encoders.0", etc.)
    parts = key.split('.')
    if len(parts) >= 3:
        layer = parts[2]
        if layer not in layer_types:
            layer_types[layer] = []
        layer_types[layer].append(key)

print(f"\n   Found {len(layer_types)} layer types:")
for layer_type, keys in sorted(layer_types.items()):
    print(f"     {layer_type}: {len(keys)} parameters")

# Check for potential issues
print(f"\n" + "=" * 80)
print("POTENTIAL ISSUES")
print("=" * 80)

# 1. Check for norm layers (could cause variance suppression)
norm_keys = [k for k in main_encoder_keys if 'norm' in k.lower()]
print(f"\n1️⃣ Normalization layers ({len(norm_keys)} found):")
for key in sorted(norm_keys)[:10]:
    weight = flow_data[key]
    print(f"   {key}: shape={weight.shape}, mean={weight.mean().item():.6f}, std={weight.std().item():.6f}")
if len(norm_keys) > 10:
    print(f"   ... and {len(norm_keys) - 10} more")

# 2. Check for scaling factors
scale_keys = [k for k in main_encoder_keys if 'scale' in k.lower() or 'gamma' in k.lower()]
if scale_keys:
    print(f"\n2️⃣ Scaling factors ({len(scale_keys)} found):")
    for key in sorted(scale_keys):
        weight = flow_data[key]
        print(f"   {key}: {weight.shape}, range=[{weight.min().item():.6f}, {weight.max().item():.6f}]")

# 3. Check if there's an after_norm layer (could suppress output)
after_norm_keys = [k for k in flow_data.keys() if 'after_norm' in k.lower()]
if after_norm_keys:
    print(f"\n3️⃣ After-norm layer found (POTENTIAL ISSUE):")
    for key in sorted(after_norm_keys):
        weight = flow_data[key]
        print(f"   {key}")
        print(f"     Shape: {weight.shape}")
        print(f"     Mean: {weight.mean().item():.8f}")
        print(f"     Std: {weight.std().item():.8f}")

        # Check if it's initialized as identity-like
        if 'weight' in key:
            identity_check = (weight - 1.0).abs().mean().item()
            print(f"     Distance from 1.0 (identity): {identity_check:.8f}")
            if identity_check > 0.1:
                print(f"     ⚠️  NOT identity-like!")

print("\n" + "=" * 80)
print("NEXT: Check encoder INPUT (speech_emb_matrix) for differences")
print("=" * 80)
