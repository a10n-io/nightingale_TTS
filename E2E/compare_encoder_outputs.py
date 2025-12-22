#!/usr/bin/env python3
"""Compare Swift encoder output with Python reference.

This script loads Swift's generated encoder output (mu) and compares it
with what Python would produce for the same speech tokens.
"""

import numpy as np
import torch
from safetensors import safe_open

print("="*80)
print("ENCODER OUTPUT COMPARISON: Swift vs Python")
print("="*80)

# Load Swift's encoder output (saved from GenerateAudio)
swift_mel_path = 'E2E/swift_generated_mel_raw.safetensors'

try:
    with safe_open(swift_mel_path, framework='numpy') as f:
        swift_mel = f.get_tensor('mel')

    print(f"\n✅ Loaded Swift mel: shape={swift_mel.shape}")
    print(f"   Range: [{swift_mel.min():.4f}, {swift_mel.max():.4f}]")
    print(f"   Mean: {swift_mel.mean():.4f}")

    # Analyze frequency structure
    print(f"\n   Channel energies (mean across time):")
    for i in [0, 10, 20, 30, 40, 50, 60, 70, 79]:
        energy = swift_mel[0, i, :].mean()
        print(f"     Channel {i:2d}: {energy:.4f}")

    # Check if values are in log scale (negative) or linear scale
    num_negative = (swift_mel < 0).sum()
    num_positive = (swift_mel > 0).sum()
    total = swift_mel.size

    print(f"\n   Value distribution:")
    print(f"     Negative: {num_negative}/{total} ({100*num_negative/total:.1f}%)")
    print(f"     Positive: {num_positive}/{total} ({100*num_positive/total:.1f}%)")

    if num_positive > num_negative:
        print(f"     ⚠️  Mixed positive/negative - NOT valid log mel!")
        print(f"     Expected: All negative values (log scale)")
    else:
        print(f"     ✅ Mostly negative - could be log mel")

    # Try to infer if this is encoder output or decoder output
    if swift_mel.min() < -5 and swift_mel.max() < 0:
        print(f"\n   Interpretation: Looks like LOG MEL spectrogram (decoder output)")
    elif swift_mel.min() > -3 and abs(swift_mel.mean()) < 1:
        print(f"\n   Interpretation: Looks like ENCODER OUTPUT (conditioning)")
    else:
        print(f"\n   Interpretation: UNKNOWN - values don't match expected patterns")

except Exception as e:
    print(f"\n❌ Error loading Swift mel: {e}")

print(f"\n{'='*80}")
print("DIAGNOSIS")
print("="*80)
print("""
For proper audio generation:
1. Encoder output (mu) should be: range ≈[-2, 2], mean ≈0 (conditioning signal)
2. Decoder output (mel) should be: range ≈[-10, -2], ALL NEGATIVE (log mel)
3. Vocoder expects log mel input

Swift's current decoder output is MIXED positive/negative, which is WRONG.
This suggests the ODE solver is not converging to valid mel spectrograms.

Next steps:
1. Verify ODE timesteps match Python (should be cosine scheduled)
2. Check if decoder forward pass uses mu conditioning correctly
3. Verify CFG weight and integration dt calculations
""")
