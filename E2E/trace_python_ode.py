#!/usr/bin/env python3
"""Trace Python ODE to see intermediate values."""

import sys
sys.path.insert(0, '/Users/a10n/Projects/nightingale_TTS/python')

import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

# Load model
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals

model_dir = PROJECT_ROOT / "models" / "chatterbox"
device = "cpu"

model = ChatterboxMultilingualTTS.from_local(str(model_dir), device=device)
voice_path = PROJECT_ROOT / "baked_voices" / "samantha" / "baked_voice.pt"
model.conds = Conditionals.load(str(voice_path), map_location=device)

# Load Python tokens
tokens_path = PROJECT_ROOT / "E2E" / "python_generated_tokens.npy"
python_tokens = np.load(tokens_path)
speech_tokens = torch.from_numpy(python_tokens).unsqueeze(0).long().to(device)

print("=" * 80)
print("PYTHON ODE TRACE")
print("=" * 80)

# Monkey-patch the solve_euler to print intermediate values
from chatterbox.models.s3gen.flow import CausalMaskedDiffWithXvec

original_solve_euler = CausalMaskedDiffWithXvec.solve_euler

def traced_solve_euler(self, x, t_span, mu, mask, spks, cond):
    print(f"\nInitial noise: range=[{x.min().item():.4f}, {x.max().item():.4f}], mean={x.mean().item():.4f}")

    sol = [x]
    for step in range(1, len(t_span)):
        t = t_span[step - 1]
        dt = t_span[step] - t

        # Estimate velocity
        dphi_dt = self.estimator(x, mask, mu, t, spks, cond)

        print(f"\nStep {step}/{len(t_span)-1}: t={t.item():.6f}, dt={dt.item():.6f}")
        print(f"  x_before: [{x.min().item():.4f}, {x.max().item():.4f}]")
        print(f"  velocity: [{dphi_dt.min().item():.4f}, {dphi_dt.max().item():.4f}]")

        # Euler step
        x = x + dt * dphi_dt
        sol.append(x)

        print(f"  x_after:  [{x.min().item():.4f}, {x.max().item():.4f}]")

    return sol[-1]

CausalMaskedDiffWithXvec.solve_euler = traced_solve_euler

with torch.inference_mode():
    print("\nGenerating mel with traced ODE...")
    output_mels = model.s3gen.flow_inference(
        speech_tokens=speech_tokens,
        ref_wav=None,
        ref_sr=None,
        ref_dict=model.conds.gen,
        finalize=True,
        n_cfm_timesteps=10
    )

print(f"\n" + "=" * 80)
print("FINAL MEL")
print("=" * 80)
print(f"Shape: {output_mels.shape}")
print(f"Range: [{output_mels.min().item():.4f}, {output_mels.max().item():.4f}]")
print(f"Mean: {output_mels.mean().item():.4f}")
print("=" * 80)
