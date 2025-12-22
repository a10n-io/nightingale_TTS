#!/usr/bin/env python3
"""Analyze Swift decoder outputs to diagnose ODE divergence.

This script loads Swift's encoder and ODE outputs and analyzes them to find
the root cause of why the ODE is diverging instead of converging.
"""

import numpy as np
from pathlib import Path
from safetensors import safe_open

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
E2E_DIR = PROJECT_ROOT / "E2E"

print("="*80)
print("SWIFT DECODER ANALYSIS")
print("="*80)

# Load Swift encoder outputs
print("\n1. ENCODER OUTPUTS")
print("-"*80)
with safe_open(str(E2E_DIR / "swift_encoder_outputs.safetensors"), framework='numpy') as f:
    encoder_out = f.get_tensor('encoder_out')
    mu = f.get_tensor('mu')

print(f"encoder_out: {encoder_out.shape}")
print(f"  Range: [{encoder_out.min():.4f}, {encoder_out.max():.4f}]")
print(f"  Mean: {encoder_out.mean():.4f}")
print(f"\nmu (encoder projection): {mu.shape}")
print(f"  Range: [{mu.min():.4f}, {mu.max():.4f}]")
print(f"  Mean: {mu.mean():.4f}")

# mu is the conditioning signal that tells the decoder what mel to generate
# It should have meaningful values that guide the ODE to the target mel
print(f"\n  Analysis: mu statistics by channel (first 10 channels):")
for i in range(10):
    channel_mean = mu[0, i, :].mean()
    channel_std = mu[0, i, :].std()
    print(f"    Channel {i:2d}: mean={channel_mean:7.4f}, std={channel_std:.4f}")

# Load ODE initialization
print("\n2. ODE INITIALIZATION")
print("-"*80)
with safe_open(str(E2E_DIR / "swift_ode_init.safetensors"), framework='numpy') as f:
    mu_T = f.get_tensor('mu_T')
    cond_T = f.get_tensor('cond_T')
    spk_emb = f.get_tensor('spk_emb')
    initial_noise = f.get_tensor('initial_noise')
    t_span = f.get_tensor('t_span')

print(f"mu_T: {mu_T.shape}, range=[{mu_T.min():.4f}, {mu_T.max():.4f}]")
print(f"cond_T: {cond_T.shape}, range=[{cond_T.min():.4f}, {cond_T.max():.4f}]")
print(f"spk_emb: {spk_emb.shape}, range=[{spk_emb.min():.4f}, {spk_emb.max():.4f}]")
print(f"initial_noise: {initial_noise.shape}, range=[{initial_noise.min():.4f}, {initial_noise.max():.4f}]")
print(f"t_span: {t_span}")

# Check if mu_T matches mu (should be transposed)
mu_transposed = mu.transpose(0, 2, 1)  # [1, 636, 80] -> [1, 80, 636]
print(f"\nVerification:")
print(f"  mu shape: {mu.shape}")
print(f"  mu_T shape: {mu_T.shape}")
print(f"  Expected mu_T shape (transposed mu): {mu_transposed.shape}")
if mu_T.shape == mu_transposed.shape:
    diff = np.abs(mu_T - mu_transposed).max()
    print(f"  Max difference: {diff:.2e} {'✓' if diff < 1e-6 else '✗'}")
else:
    print(f"  ⚠️  SHAPE MISMATCH!")

# Load ODE step 1 outputs
print("\n3. ODE STEP 1 ANALYSIS")
print("-"*80)
with safe_open(str(E2E_DIR / "swift_ode_step1.safetensors"), framework='numpy') as f:
    x_before = f.get_tensor('x_before')
    v_cond = f.get_tensor('v_cond')
    v_uncond = f.get_tensor('v_uncond')
    v_cfg = f.get_tensor('v_cfg')
    x_after = f.get_tensor('x_after')

print(f"x_before (initial noise): {x_before.shape}")
print(f"  Range: [{x_before.min():.4f}, {x_before.max():.4f}]")
print(f"  Mean: {x_before.mean():.4f}, Std: {x_before.std():.4f}")

print(f"\nv_cond (conditional velocity): {v_cond.shape}")
print(f"  Range: [{v_cond.min():.4f}, {v_cond.max():.4f}]")
print(f"  Mean: {v_cond.mean():.4f}")

print(f"\nv_uncond (unconditional velocity): {v_uncond.shape}")
print(f"  Range: [{v_uncond.min():.4f}, {v_uncond.max():.4f}]")
print(f"  Mean: {v_uncond.mean():.4f}")

print(f"\nv_cfg (after CFG): {v_cfg.shape}")
print(f"  Range: [{v_cfg.min():.4f}, {v_cfg.max():.4f}]")
print(f"  Mean: {v_cfg.mean():.4f}")

print(f"\nx_after (after Euler step): {x_after.shape}")
print(f"  Range: [{x_after.min():.4f}, {x_after.max():.4f}]")
print(f"  Mean: {x_after.mean():.4f}")

# Analyze the velocity field
print("\n4. VELOCITY FIELD ANALYSIS")
print("-"*80)

# The velocity should point from current state (noise ~[−4,4]) toward target mel (~[−10,−2])
# If initial x is around [−4,4] and target is around [−10,−2], velocity should:
#   - Have large negative component (to push x down from 4 toward −2)
#   - Be asymmetric (more negative than positive)

print("Expected behavior for convergence:")
print("  Initial x: ~[−4, 4] (Gaussian noise)")
print("  Target mel: ~[−10, −2] (log mel spectrogram, all negative)")
print("  Expected velocity: Should push x DOWN (toward −10 to −2)")
print("    → Should be MOSTLY NEGATIVE (large negative values, small positive)")
print("")

v_neg = (v_cond < 0).sum()
v_pos = (v_cond > 0).sum()
v_total = v_cond.size

print(f"Actual velocity distribution (v_cond):")
print(f"  Negative values: {v_neg}/{v_total} ({100*v_neg/v_total:.1f}%)")
print(f"  Positive values: {v_pos}/{v_total} ({100*v_pos/v_total:.1f}%)")
print(f"  Mean: {v_cond.mean():.4f}")
print(f"  Median: {np.median(v_cond):.4f}")

if abs(v_cond.mean()) < 0.5:
    print("\n⚠️  ISSUE DETECTED: Velocity is nearly SYMMETRIC around zero!")
    print("   This means the decoder is not pushing toward the target mel.")
    print("   The velocity should be strongly NEGATIVE to reach log mel range.")
else:
    print("\n✓ Velocity has clear directional bias")

# Check if velocity magnitude is reasonable
v_magnitude = np.abs(v_cond).mean()
print(f"\nVelocity magnitude:")
print(f"  Mean |v|: {v_magnitude:.4f}")

# For convergence over 10 steps with dt ~ 0.1, we need |v| ~ distance / (10 * 0.1)
# Distance from [−4,4] to [−10,−2]: roughly 8 units
# Expected |v| ~ 8 / 1.0 = 8
expected_v = 8.0
print(f"  Expected for convergence: ~{expected_v:.1f}")

if v_magnitude < expected_v * 0.2:
    print(f"  ⚠️  Velocity is TOO SMALL! ({v_magnitude:.2f} << {expected_v:.1f})")
    print("     The decoder is not producing strong enough corrections.")
elif v_magnitude > expected_v * 5:
    print(f"  ⚠️  Velocity is TOO LARGE! ({v_magnitude:.2f} >> {expected_v:.1f})")
    print("     This could cause instability or overshoot.")
else:
    print(f"  ✓ Velocity magnitude is reasonable")

# Analyze channel-wise velocity
print("\n5. CHANNEL-WISE VELOCITY ANALYSIS")
print("-"*80)
print("Checking if different mel channels get different velocities...")

channel_velocities = v_cond[0, :, :].mean(axis=1)  # [80]
print(f"\nFirst 10 channel mean velocities:")
for i in range(10):
    print(f"  Channel {i:2d}: {channel_velocities[i]:7.4f}")

v_channel_std = channel_velocities.std()
print(f"\nStd dev of channel means: {v_channel_std:.4f}")

if v_channel_std < 0.1:
    print("  ⚠️  All channels have similar velocities!")
    print("     The decoder is not responding to frequency structure in mu.")
else:
    print("  ✓ Channels have varying velocities")

# Final diagnosis
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

issues = []

if abs(v_cond.mean()) < 0.5:
    issues.append("Velocity is symmetric around zero (should be negative)")
if v_magnitude < 2.0:
    issues.append("Velocity magnitude is too small for convergence")
if v_channel_std < 0.1:
    issues.append("All channels have similar velocities (not using mu conditioning)")

if issues:
    print("\n🔴 CRITICAL ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    print("\nMost likely root causes:")
    print("  1. Decoder is not using mu properly in forward pass")
    print("     → Check concatenation: should be [x, mu, spk, cond] on channel dim")
    print("  2. Decoder weights may be scaled incorrectly")
    print("     → Check if weight loading transposed correctly")
    print("  3. Final projection may have wrong activation or scaling")
    print("     → Should output raw velocity values, no activation")
else:
    print("\n✓ No obvious issues detected")

# Compare with what we'd expect from Python
print("\n" + "="*80)
print("EXPECTED vs ACTUAL")
print("="*80)
print("\nFor proper convergence from noise to log mel:")
print(f"  Expected v_cond range: ~[−15, −5] (strongly negative)")
print(f"  Actual v_cond range: [{v_cond.min():.2f}, {v_cond.max():.2f}]")
print(f"\n  Expected mean: ~−10 (push down to log mel)")
print(f"  Actual mean: {v_cond.mean():.2f}")
print(f"\n  Expected channel variation: high (different freqs need different corrections)")
print(f"  Actual channel std: {v_channel_std:.2f}")

print("\n" + "="*80)
