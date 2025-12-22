#!/usr/bin/env python3
"""
Diagnose why Swift ODE diverges by comparing decoder outputs.

This script:
1. Checks if Python reference outputs exist
2. Loads Swift's generated mel
3. Identifies the specific issue pattern
4. Suggests targeted fixes
"""

import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
SWIFT_MEL_PATH = PROJECT_ROOT / "E2E" / "swift_generated_mel_raw.safetensors"

print("="*80)
print("ODE DIVERGENCE DIAGNOSIS")
print("="*80)

# Load Swift's output
try:
    from safetensors import safe_open
    with safe_open(str(SWIFT_MEL_PATH), framework='numpy') as f:
        swift_mel = f.get_tensor('mel')

    print(f"\n✅ Loaded Swift mel: {swift_mel.shape}")
    print(f"   Range: [{swift_mel.min():.4f}, {swift_mel.max():.4f}]")
    print(f"   Mean: {swift_mel.mean():.4f}")

    # Analyze patterns
    print(f"\n📊 Pattern Analysis:")

    # Check if values are symmetric around zero
    pos_mean = swift_mel[swift_mel > 0].mean() if (swift_mel > 0).any() else 0
    neg_mean = swift_mel[swift_mel < 0].mean() if (swift_mel < 0).any() else 0
    print(f"   Positive values: mean={pos_mean:.4f}")
    print(f"   Negative values: mean={neg_mean:.4f}")
    print(f"   Symmetry: {abs(pos_mean + neg_mean):.4f} (0=perfect symmetry)")

    # Check channel-wise patterns
    channel_means = swift_mel[0, :, :].mean(axis=1)
    channel_trend = np.polyfit(range(80), channel_means, 1)[0]
    print(f"\n   Channel trend: {channel_trend:.6f}/channel")
    if abs(channel_trend) < 0.01:
        print(f"     → Channels have similar energy (NO frequency structure) ⚠️")
    else:
        print(f"     → Channels have varying energy (some frequency structure) ✓")

    # Check temporal patterns
    time_variance = swift_mel[0, :, :].var(axis=0).mean()
    print(f"\n   Temporal variance: {time_variance:.4f}")
    if time_variance < 0.5:
        print(f"     → Low temporal variation (static/stuck) ⚠️")
    else:
        print(f"     → Good temporal variation ✓")

    # Specific bug patterns
    print(f"\n🔍 Bug Pattern Detection:")

    if abs(swift_mel.mean()) < 0.1 and swift_mel.std() > 1.0:
        print(f"   ⚠️  PATTERN: Zero-mean high-variance")
        print(f"       Likely cause: Decoder initialized but weights not applied")
        print(f"       Fix: Verify decoder.update() is called correctly")

    if (swift_mel > 0).sum() > (swift_mel < 0).sum() * 0.9:
        print(f"   ⚠️  PATTERN: Mostly positive values")
        print(f"       Likely cause: Missing log() operation or wrong activation")
        print(f"       Fix: Check if decoder applies log() to output")

    if abs(channel_trend) < 0.001:
        print(f"   ⚠️  PATTERN: Flat frequency response")
        print(f"       Likely cause: Encoder not conditioning properly")
        print(f"       Fix: Verify mu is used correctly in decoder")

    if time_variance < 0.2:
        print(f"   ⚠️  PATTERN: Temporal stagnation")
        print(f"       Likely cause: ODE not integrating (dt too small or v=0)")
        print(f"       Fix: Check dt values and decoder velocity output")

    # Check if looks like initialized weights
    if abs(swift_mel.mean()) < 0.01 and 1.5 < swift_mel.std() < 2.5:
        print(f"   ⚠️  PATTERN: Gaussian noise (like initialization)")
        print(f"       Likely cause: Decoder forward pass not executing")
        print(f"       Fix: Add debug prints in decoder forward")

    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print("="*80)
    print("""
Based on the analysis, the most likely issues are:

1. **Decoder not using mu conditioning** (if flat frequency response)
   - Check that mu is concatenated with x before downBlocks
   - Verify concatenation is on channel dim: concat([x, mu, spk, cond], dim=1)

2. **ODE velocity wrong sign** (if diverging instead of converging)
   - Verify: xt = xt + v * dt (not xt = xt - v * dt)
   - Check v is not being negated somewhere

3. **Attention mask wrong format** (if random-looking output)
   - Full attention: all zeros (0 = attend)
   - Causal: upper triangle -inf (block future)
   - Verify mask is applied correctly in attention

4. **Layer not evaluating** (if Gaussian noise pattern)
   - Add eval() calls after each major operation
   - Check for MLX graph caching issues

Next step: Add comprehensive tracing to decoder first downBlock and compare
with Python to find exact divergence point.
    """)

except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"\nMake sure to run GenerateAudio first to create swift_generated_mel_raw.safetensors")

print(f"\n{'='*80}")
