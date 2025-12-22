#!/usr/bin/env python3
"""Save Python mel for Swift vocoder test."""

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

print("Generating Python mel...")

with torch.inference_mode():
    output_mels = model.s3gen.flow_inference(
        speech_tokens=speech_tokens,
        ref_wav=None,
        ref_sr=None,
        ref_dict=model.conds.gen,
        finalize=True,
        n_cfm_timesteps=10
    )

print(f"Python mel: {output_mels.shape}, range=[{output_mels.min().item():.4f}, {output_mels.max().item():.4f}]")

# Save as numpy
mel_np = output_mels.cpu().numpy()
output_path = PROJECT_ROOT / "E2E" / "python_mel_for_swift_vocoder.npy"
np.save(output_path, mel_np)

print(f"✅ Saved: {output_path}")
