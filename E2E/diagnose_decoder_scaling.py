#!/usr/bin/env python3
"""Diagnose decoder scaling issue by analyzing input/output magnitudes.

The decoder takes:
- x (noise): ~[-4, 4]
- mu (encoder): ~[-2, 2]
- spk (speaker): ~[-0.3, 0.3]
- cond (prompt mel): ~[-11.5, 0.6]

And should output velocity that pushes x toward target mel (~[-10, -2]).
But Swift outputs velocity ~[-1.3, 1.3] which is 26x too small.
"""

import numpy as np
from pathlib import Path
from safetensors import safe_open

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
E2E_DIR = PROJECT_ROOT / "E2E"

print("="*80)
print("DECODER SCALING DIAGNOSIS")
print("="*80)

# Load decoder inputs
with safe_open(str(E2E_DIR / "swift_ode_init.safetensors"), framework='numpy') as f:
    mu_T = f.get_tensor('mu_T')
    cond_T = f.get_tensor('cond_T')
    spk_emb = f.get_tensor('spk_emb')
    x = f.get_tensor('initial_noise')

print("\n1. DECODER INPUTS")
print("-"*80)
print(f"x (noise): shape={x.shape}, range=[{x.min():.2f}, {x.max():.2f}], std={x.std():.2f}")
print(f"mu: shape={mu_T.shape}, range=[{mu_T.min():.2f}, {mu_T.max():.2f}], std={mu_T.std():.2f}")
print(f"spk: shape={spk_emb.shape}, range=[{spk_emb.min():.2f}, {spk_emb.max():.2f}], std={spk_emb.std():.2f}")
print(f"cond: shape={cond_T.shape}, range=[{cond_T.min():.2f}, {cond_T.max():.2f}], std={cond_T.std():.2f}")

# Analyze cond more carefully - it's the prompt mel
print("\n2. CONDITION ANALYSIS (cond = prompt mel)")
print("-"*80)
print("cond is the prompt mel spectrogram (log scale)")
prompt_len = 500  # Approximate based on shape
gen_len = cond_T.shape[2] - prompt_len

cond_prompt = cond_T[:, :, :prompt_len]
cond_gen = cond_T[:, :, prompt_len:]

print(f"\nPrompt part (first ~{prompt_len} frames):")
print(f"  Range: [{cond_prompt.min():.2f}, {cond_prompt.max():.2f}]")
print(f"  Mean: {cond_prompt.mean():.2f}")
print(f"  % Negative: {100*(cond_prompt < 0).sum()/cond_prompt.size:.1f}%")

print(f"\nGenerated part (last ~{gen_len} frames):")
print(f"  Range: [{cond_gen.min():.2f}, {cond_gen.max():.2f}]")
print(f"  Mean: {cond_gen.mean():.2f}")
print(f"  % Zero: {100*(cond_gen == 0).sum()/cond_gen.size:.1f}%")

# The generated part should be all zeros
if abs(cond_gen.mean()) < 0.01:
    print("  ✓ Generated part is zeros (correct)")
else:
    print("  ⚠️  Generated part is NOT zeros!")

# Now check decoder output
with safe_open(str(E2E_DIR / "swift_ode_step1.safetensors"), framework='numpy') as f:
    v_cond = f.get_tensor('v_cond')

print("\n3. DECODER OUTPUT")
print("-"*80)
print(f"v_cond: shape={v_cond.shape}, range=[{v_cond.min():.2f}, {v_cond.max():.2f}], std={v_cond.std():.2f}")

# Compute concatenated input
# Need to expand spk to match spatial dimensions
spk_expanded = np.tile(spk_emb[:, :, np.newaxis], (1, 1, x.shape[2]))
h_input = np.concatenate([x, mu_T, spk_expanded, cond_T], axis=1)

print("\n4. CONCATENATED INPUT (h)")
print("-"*80)
print(f"h = [x, mu, spk, cond]")
print(f"  Shape: {h_input.shape}")
print(f"  Range: [{h_input.min():.2f}, {h_input.max():.2f}]")
print(f"  Mean: {h_input.mean():.2f}")
print(f"  Std: {h_input.std():.2f}")

# The input h has 4 * 80 = 320 channels
# Channels 0-79: x (noise)
# Channels 80-159: mu (encoder)
# Channels 160-239: spk (speaker)
# Channels 240-319: cond (prompt mel)

print("\nPer-input statistics:")
print(f"  x (ch 0-79):       mean={h_input[:, 0:80, :].mean():.4f}, std={h_input[:, 0:80, :].std():.4f}")
print(f"  mu (ch 80-159):    mean={h_input[:, 80:160, :].mean():.4f}, std={h_input[:, 80:160, :].std():.4f}")
print(f"  spk (ch 160-239):  mean={h_input[:, 160:240, :].mean():.4f}, std={h_input[:, 160:240, :].std():.4f}")
print(f"  cond (ch 240-319): mean={h_input[:, 240:320, :].mean():.4f}, std={h_input[:, 240:320, :].std():.4f}")

# Check if there's a scaling issue with cond
print("\n5. POTENTIAL SCALING ISSUE")
print("-"*80)

# cond has large negative values (log mel) while x, mu, spk are small
# If decoder weights are initialized assuming all inputs have similar scale,
# this could cause issues

print("\nInput magnitude ratios:")
x_mag = np.abs(h_input[:, 0:80, :]).mean()
mu_mag = np.abs(h_input[:, 80:160, :]).mean()
spk_mag = np.abs(h_input[:, 160:240, :]).mean()
cond_mag = np.abs(h_input[:, 240:320, :]).mean()

print(f"  |x|:    {x_mag:.4f}")
print(f"  |mu|:   {mu_mag:.4f}")
print(f"  |spk|:  {spk_mag:.4f}")
print(f"  |cond|: {cond_mag:.4f}")

if cond_mag > 2 * x_mag:
    print(f"\n⚠️  ISSUE: cond magnitude ({cond_mag:.2f}) is {cond_mag/x_mag:.1f}x larger than x!")
    print("   This could cause the decoder to be dominated by cond input,")
    print("   making it insensitive to x (the state we're trying to evolve).")

# Analyze what the velocity SHOULD be
print("\n6. EXPECTED VELOCITY MAGNITUDE")
print("-"*80)

# At t=0, we're at x ~ [-4, 4]
# Target mel is ~ [-10, -2] (based on log mel statistics)
# Distance to travel: from [−4,4] center to [−10,−2] center
# Initial: mean ~ 0, target: mean ~ −6
# So we need to move by ~−6 on average

target_mel_range = (-10, -2)
initial_x_mean = x.mean()
target_mel_mean = (target_mel_range[0] + target_mel_range[1]) / 2

distance = target_mel_mean - initial_x_mean
n_steps = 10
avg_dt = 1.0 / n_steps

expected_velocity_magnitude = abs(distance) / (n_steps * avg_dt)

print(f"Initial x mean: {initial_x_mean:.2f}")
print(f"Target mel mean: {target_mel_mean:.2f}")
print(f"Distance: {distance:.2f}")
print(f"Expected velocity: ~{distance:.2f} / ({n_steps} steps × {avg_dt:.2f} dt) = {expected_velocity_magnitude:.2f}")

actual_velocity_magnitude = np.abs(v_cond).mean()
print(f"\nActual velocity magnitude: {actual_velocity_magnitude:.2f}")
print(f"Ratio: {actual_velocity_magnitude / expected_velocity_magnitude:.2f}x")

if actual_velocity_magnitude < expected_velocity_magnitude * 0.3:
    print(f"\n🔴 CRITICAL: Velocity is {expected_velocity_magnitude/actual_velocity_magnitude:.1f}x too small!")
    print("   The decoder is not producing strong enough corrections.")

print("\n" + "="*80)
print("SUMMARY & NEXT STEPS")
print("="*80)

print("\nFindings:")
print("  1. Decoder inputs look correct (proper concatenation)")
print("  2. Velocity magnitude is 26x too small")
print("  3. cond (prompt mel) has much larger magnitude than x/mu/spk")

print("\nMost likely causes:")
print("  A. Weight loading issue: Decoder weights may not be applied correctly")
print("  B. Scaling issue: Decoder may need input normalization or output scaling")
print("  C. Initialization: Decoder may be stuck at initialization values")

print("\nRecommended debugging steps:")
print("  1. Check if decoder weights are actually being used (not initialization)")
print("  2. Add tracing to first decoder layer to verify computation")
print("  3. Compare first layer output with Python reference")
