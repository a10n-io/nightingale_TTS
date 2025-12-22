#!/usr/bin/env python3
"""Compare Swift and Python outputs to find where they diverge."""

import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
E2E_DIR = PROJECT_ROOT / "E2E"
REF_DIR = E2E_DIR / "reference_outputs" / "samantha" / "expressive_surprise_en"

print("="*80)
print("COMPARING SWIFT vs PYTHON OUTPUTS")
print("="*80)

# Compare Step 6 outputs (encoder)
print("\n[Step 6: Encoder]")
print("-"*80)

try:
    swift_encoder = np.load(E2E_DIR / "swift_encoder_outputs.safetensors")
    print("ERROR: Cannot load safetensors with numpy")
except Exception as e:
    print(f"  Note: Need to extract tensors from safetensors")

# Instead, let me load Python reference step6 outputs
try:
    py_encoder = np.load(REF_DIR / "step6_encoder_out.npy")
    py_mu = np.load(REF_DIR / "step6_mu.npy")

    print(f"Python encoder_out: {py_encoder.shape}")
    print(f"  Range: [{py_encoder.min():.4f}, {py_encoder.max():.4f}]")
    print(f"  Mean: {py_encoder.mean():.4f}")

    print(f"\nPython mu: {py_mu.shape}")
    print(f"  Range: [{py_mu.min():.4f}, {py_mu.max():.4f}]")
    print(f"  Mean: {py_mu.mean():.4f}")

except Exception as e:
    print(f"  Could not load Python reference: {e}")

# Check decoder step 7
print("\n[Step 7: ODE Step 1]")
print("-"*80)

try:
    py_step1_v_cond = np.load(REF_DIR / "step7_step1_dxdt_cond.npy")
    py_step1_x_before = np.load(REF_DIR / "step7_step1_x_before.npy")
    py_step1_x_after = np.load(REF_DIR / "step7_step1_x_after.npy")

    print(f"Python step1 x_before: {py_step1_x_before.shape}")
    print(f"  Range: [{py_step1_x_before.min():.4f}, {py_step1_x_before.max():.4f}]")

    print(f"\nPython step1 v_cond: {py_step1_v_cond.shape}")
    print(f"  Range: [{py_step1_v_cond.min():.4f}, {py_step1_v_cond.max():.4f}]")
    print(f"  Mean: {py_step1_v_cond.mean():.4f}")

    print(f"\nPython step1 x_after: {py_step1_x_after.shape}")
    print(f"  Range: [{py_step1_x_after.min():.4f}, {py_step1_x_after.max():.4f}]")

except Exception as e:
    print(f"  Python reference not found: {e}")
    print(f"  Need to generate Python reference outputs first")

# Check final mel
print("\n[Step 7: Final Mel]")
print("-"*80)

try:
    py_final_mel = np.load(REF_DIR / "step7_final_mel.npy")

    print(f"Python final mel: {py_final_mel.shape}")
    print(f"  Range: [{py_final_mel.min():.4f}, {py_final_mel.max():.4f}]")
    print(f"  Mean: {py_final_mel.mean():.4f}")
    print(f"  Negative %: {100*(py_final_mel < 0).sum()/py_final_mel.size:.1f}%")

except Exception as e:
    print(f"  Python reference not found: {e}")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Generate Python reference outputs with generate_decoder_reference.py")
print("2. Run Swift and save intermediate outputs")
print("3. Compare tensors to find divergence point")
print("="*80)
