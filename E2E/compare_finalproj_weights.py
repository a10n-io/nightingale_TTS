"""
Compare finalProj weights between Python and Swift to find weight loading issues.
"""
import torch
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"

print("=" * 80)
print("COMPARE FINALPROJ WEIGHTS - Python vs Swift")
print("=" * 80)

# Load Python finalProj weights
python_path = FORENSIC_DIR / "python_finalproj_weights.safetensors"
if not python_path.exists():
    print(f"\n❌ Python weights not found: {python_path}")
    print(f"   Run: cd E2E && python trace_python_decoder.py")
    exit(1)

python_data = load_file(str(python_path))
python_weight = python_data["final_proj_weight"]
python_bias = python_data["final_proj_bias"]

print(f"\n📊 Python finalProj:")
print(f"   Weight shape: {python_weight.shape}")
print(f"   Weight range: [{python_weight.min().item():.8f}, {python_weight.max().item():.8f}]")
print(f"   Weight mean: {python_weight.mean().item():.8f}")
print(f"   Weight std: {python_weight.std().item():.8f}")
print(f"   Bias shape: {python_bias.shape}")
print(f"   Bias range: [{python_bias.min().item():.8f}, {python_bias.max().item():.8f}]")
print(f"   Bias mean: {python_bias.mean().item():.8f}")
print(f"   Bias std: {python_bias.std().item():.8f}")

# Now we need to extract Swift's finalProj weights from the model file
models_dir = PROJECT_ROOT / "models" / "chatterbox"
s3gen_path = models_dir / "s3gen.safetensors"

if not s3gen_path.exists():
    print(f"\n❌ Swift model not found: {s3gen_path}")
    exit(1)

flow_data = load_file(str(s3gen_path))
print(f"\n🔍 Looking for finalProj in Swift model...")
print(f"   Total keys in s3gen.safetensors: {len(flow_data)}")

# Find finalProj keys
finalproj_keys = [k for k in flow_data.keys() if 'final_proj' in k.lower()]
print(f"   Keys with 'final_proj': {finalproj_keys}")

if not finalproj_keys:
    print("\n   ⚠️  No finalProj keys found, checking for similar patterns...")
    proj_keys = [k for k in flow_data.keys() if 'proj' in k.lower() and 'final' in k.lower()]
    print(f"   Keys with 'proj' and 'final': {proj_keys}")

    # Try alternative names
    for key in ['decoder.final_proj.weight', 'decoder.final_proj.bias',
                'estimator.final_proj.weight', 'estimator.final_proj.bias']:
        if key in flow_data:
            print(f"   Found: {key}")
            finalproj_keys.append(key)

if finalproj_keys:
    # Extract weight and bias
    weight_key = [k for k in finalproj_keys if 'weight' in k.lower()][0]
    bias_keys = [k for k in finalproj_keys if 'bias' in k.lower()]

    swift_weight = flow_data[weight_key]
    swift_bias = flow_data[bias_keys[0]] if bias_keys else None

    print(f"\n📊 Swift finalProj (from flow.safetensors):")
    print(f"   Weight key: {weight_key}")
    print(f"   Weight shape: {swift_weight.shape}")
    print(f"   Weight range: [{swift_weight.min().item():.8f}, {swift_weight.max().item():.8f}]")
    print(f"   Weight mean: {swift_weight.mean().item():.8f}")
    print(f"   Weight std: {swift_weight.std().item():.8f}")

    if swift_bias is not None:
        print(f"   Bias key: {bias_keys[0]}")
        print(f"   Bias shape: {swift_bias.shape}")
        print(f"   Bias range: [{swift_bias.min().item():.8f}, {swift_bias.max().item():.8f}]")
        print(f"   Bias mean: {swift_bias.mean().item():.8f}")
        print(f"   Bias std: {swift_bias.std().item():.8f}")

    # Compare
    print(f"\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    # Check if shapes match
    if python_weight.shape != swift_weight.shape:
        print(f"\n❌ SHAPE MISMATCH!")
        print(f"   Python: {python_weight.shape}")
        print(f"   Swift: {swift_weight.shape}")
        print(f"\n   This could be due to transpose differences (Conv1d format)")
    else:
        print(f"\n✅ Shapes match: {python_weight.shape}")

    # Compare values
    weight_diff = (python_weight - swift_weight).abs()
    print(f"\n📊 Weight difference:")
    print(f"   Mean absolute diff: {weight_diff.mean().item():.8f}")
    print(f"   Max absolute diff: {weight_diff.max().item():.8f}")

    if weight_diff.mean().item() < 1e-5:
        print(f"   ✅ Weights match (diff < 1e-5)")
    elif weight_diff.mean().item() < 1e-3:
        print(f"   ⚠️  Small difference (diff < 1e-3)")
    else:
        print(f"   ❌ LARGE DIFFERENCE (diff > 1e-3)")

    if swift_bias is not None and python_bias.shape == swift_bias.shape:
        bias_diff = (python_bias - swift_bias).abs()
        print(f"\n📊 Bias difference:")
        print(f"   Mean absolute diff: {bias_diff.mean().item():.8f}")
        print(f"   Max absolute diff: {bias_diff.max().item():.8f}")

        if bias_diff.mean().item() < 1e-5:
            print(f"   ✅ Biases match (diff < 1e-5)")
        elif bias_diff.mean().item() < 1e-3:
            print(f"   ⚠️  Small difference (diff < 1e-3)")
        else:
            print(f"   ❌ LARGE DIFFERENCE (diff > 1e-3)")

            # Check if bias could explain the 0.91 dB difference
            bias_delta = (swift_bias - python_bias).mean().item()
            print(f"\n🔍 Bias delta (Swift - Python): {bias_delta:.8f}")
            print(f"   If this is the cause, Swift mel should be {bias_delta:.3f} dB different")
            print(f"   Observed mel difference: 0.91 dB (Swift brighter)")

            if abs(bias_delta - 0.91) < 0.1:
                print(f"   ✅ BIAS MISMATCH EXPLAINS THE DIFFERENCE!")
else:
    print(f"\n❌ Could not find finalProj weights in Swift model")
    print(f"\n   Available keys (first 50):")
    for i, key in enumerate(list(flow_data.keys())[:50]):
        print(f"     {key}")

print("\n" + "=" * 80)
