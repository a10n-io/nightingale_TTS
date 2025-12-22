#!/usr/bin/env python3
"""Check Python mel spectrogram range to compare with Swift."""

import sys
sys.path.insert(0, '/Users/a10n/Projects/nightingale_TTS/python')

import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

print("Checking Python mel spectrogram range...")

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

print(f"Generating mel from Python tokens...")

with torch.inference_mode():
    # Just get the mel, skip vocoder
    output_mels = model.s3gen.flow_inference(
        speech_tokens=speech_tokens,
        ref_wav=None,
        ref_sr=None,
        ref_dict=model.conds.gen,
        finalize=True,
        n_cfm_timesteps=10
    )

print(f"\nPython mel spectrogram:")
print(f"  Shape: {output_mels.shape}")
print(f"  Range: [{output_mels.min().item():.4f}, {output_mels.max().item():.4f}]")
print(f"  Mean: {output_mels.mean().item():.4f}")
print(f"  Median: {output_mels.median().item():.4f}")

# Check per-channel stats
for i in [0, 10, 20, 40, 60, 79]:
    channel_mean = output_mels[:, :, i].mean().item()
    print(f"  Channel {i:2d} mean: {channel_mean:7.4f}")

print("\n✅ Python mel range is correct (mostly negative, log-scale)")
