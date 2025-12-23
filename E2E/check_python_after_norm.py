"""
Check Python's encoder after_norm layer to see if it's supposed to be identity.
"""
import torch
from safetensors.torch import load_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("CHECK PYTHON ENCODER AFTER_NORM")
print("=" * 80)

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
device = "cpu"

print(f"\nLoading Python model...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

encoder = model.s3gen.flow.encoder

print(f"\n📊 Encoder type: {type(encoder).__name__}")
print(f"\n🔍 Checking for after_norm...")

if hasattr(encoder, 'after_norm'):
    after_norm = encoder.after_norm
    print(f"   ✅ Found after_norm: {type(after_norm).__name__}")

    if hasattr(after_norm, 'weight'):
        weight = after_norm.weight
        print(f"\n   after_norm.weight:")
        print(f"     Shape: {weight.shape}")
        print(f"     Range: [{weight.min().item():.8f}, {weight.max().item():.8f}]")
        print(f"     Mean: {weight.mean().item():.8f}")
        print(f"     Std: {weight.std().item():.8f}")

        # Check if it's identity-like
        identity_dist = (weight - 1.0).abs().mean().item()
        print(f"     Distance from 1.0: {identity_dist:.8f}")

        if identity_dist < 0.1:
            print(f"     ✅ Close to identity (dist < 0.1)")
        else:
            print(f"     ⚠️  NOT identity-like (dist = {identity_dist:.3f})")
            print(f"\n   💡 This explains the variance suppression!")
            print(f"      Mean 0.341 means outputs are scaled by ~0.34")
            print(f"      This would reduce variance by ~(0.34)² = 0.12")
            print(f"      Observed: Swift std=0.227, Python std=0.455")
            print(f"      Expected if suppressed: 0.455 * 0.34 = 0.155")
            print(f"      Actual Swift: 0.227")
            print(f"\n   🔍 But Python still produces std=0.455...")
            print(f"      This means Python is NOT applying after_norm the same way!")

    if hasattr(after_norm, 'bias'):
        bias = after_norm.bias
        print(f"\n   after_norm.bias:")
        print(f"     Shape: {bias.shape}")
        print(f"     Range: [{bias.min().item():.8f}, {bias.max().item():.8f}]")
        print(f"     Mean: {bias.mean().item():.8f}")
        print(f"     Std: {bias.std().item():.8f}")
else:
    print(f"   ❌ No after_norm found!")
    print(f"\n   Encoder attributes:")
    print(f"   {[attr for attr in dir(encoder) if not attr.startswith('_')][:30]}")

# Check if Python applies after_norm
print(f"\n" + "=" * 80)
print("CHECKING IF PYTHON APPLIES AFTER_NORM")
print("=" * 80)

import inspect

# Try to get encoder source
try:
    source = inspect.getsource(encoder.forward)
    print(f"\n📝 Encoder forward method:")

    # Check if after_norm is called
    if 'after_norm' in source:
        print(f"   ✅ 'after_norm' found in forward method")
        # Show the relevant lines
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'after_norm' in line.lower():
                print(f"   Line {i}: {line.strip()}")
    else:
        print(f"   ⚠️  'after_norm' NOT found in forward method!")
        print(f"   🎯 THIS IS THE ISSUE!")
        print(f"\n   Python encoder might not be using after_norm,")
        print(f"   but Swift is applying it, causing the suppression!")
except:
    print(f"   ⚠️  Could not get source (compiled module)")

print("\n" + "=" * 80)
print("HYPOTHESIS")
print("=" * 80)
print(f"\nIf Python doesn't apply after_norm but Swift does:")
print(f"  - Python outputs have full variance (std=0.455)")
print(f"  - Swift outputs are suppressed by 0.341 (std=0.227)")
print(f"  - Solution: Don't apply after_norm in Swift, or initialize as identity")
print("\n" + "=" * 80)
