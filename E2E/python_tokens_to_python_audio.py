#!/usr/bin/env python3
"""Generate audio from Python tokens using Python pipeline (reference)."""

import sys
sys.path.insert(0, '/Users/a10n/Projects/nightingale_TTS/python')

import torch
import numpy as np
from pathlib import Path
import scipy.io.wavfile as wavfile

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

print("=" * 80)
print("PYTHON TOKENS → PYTHON AUDIO (Reference)")
print("=" * 80)

# Load tokens
tokens_path = PROJECT_ROOT / "E2E" / "python_generated_tokens.npy"
python_tokens = np.load(tokens_path)
print(f"\nLoaded Python tokens: {len(python_tokens)} tokens")
print(f"First 10: {python_tokens[:10].tolist()}")

# Load model
print("\nLoading Python model...")
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals

model_dir = PROJECT_ROOT / "models" / "chatterbox"
device = "cpu"

model = ChatterboxMultilingualTTS.from_local(str(model_dir), device=device)
voice_path = PROJECT_ROOT / "baked_voices" / "samantha" / "baked_voice.pt"
model.conds = Conditionals.load(str(voice_path), map_location=device)

# Convert to tensor
speech_tokens = torch.from_numpy(python_tokens).unsqueeze(0).long().to(device)
print(f"Token tensor shape: {speech_tokens.shape}")

# Generate audio using Python S3Gen
print("\nGenerating audio from Python tokens...")
with torch.inference_mode():
    wav = model.s3gen(
        speech_tokens=speech_tokens,
        ref_wav=None,
        ref_sr=None,
        ref_dict=model.conds.gen,
        finalize=True,
        n_cfm_timesteps=10
    )

    wav_np = wav.squeeze(0).detach().cpu().numpy()
    print(f"Generated audio: {len(wav_np)} samples ({len(wav_np)/24000:.2f}s)")
    print(f"   Range: [{wav_np.min():.4f}, {wav_np.max():.4f}]")

    # Apply watermark
    watermarked = model.watermarker.apply_watermark(wav_np, sample_rate=24000)

# Save
output_path = PROJECT_ROOT / "test_audio" / "python_tokens_python_audio.wav"
wavfile.write(str(output_path), 24000, watermarked)

print(f"\n✅ Saved: {output_path}")
print(f"   Audio range: [{watermarked.min():.4f}, {watermarked.max():.4f}]")
print("\n" + "=" * 80)
