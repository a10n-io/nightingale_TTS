#!/usr/bin/env python3
"""Generate audio from Swift tokens using Python pipeline."""

import sys
sys.path.insert(0, '/Users/a10n/Projects/nightingale_TTS/python')

import torch
import numpy as np
from pathlib import Path
import scipy.io.wavfile as wavfile

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

print("=" * 80)
print("SWIFT TOKENS → PYTHON AUDIO")
print("=" * 80)

# Swift tokens from console output (after filtering 6561, 6562)
swift_tokens = [3677, 4218, 1951, 3890, 6077, 6086, 5357, 3170, 3161, 3404, 2675, 2124, 4314, 2125, 2003, 4299, 6486, 6405, 4137, 1950, 1946, 4133, 6320, 5591, 5672, 5429, 5186, 5189, 4544, 4571, 4850, 4849, 1034, 2032, 1978, 2834, 2753, 566, 647, 404, 2591, 4778, 4859, 4858, 1763, 2031, 1733, 3647, 2918, 731, 974, 245, 2432, 2513, 2648, 2181, 2175, 4671, 6374, 6377, 6373, 74, 380, 623, 632, 641, 1370, 2099, 2018, 2020, 2017, 317, 4609, 4608, 1896, 4512, 2295, 4501, 4771, 5663, 3475, 4439, 4526, 4769, 5021, 5102, 5101, 4857, 1277, 2059, 2112, 168, 2274, 2933, 8, 1560, 1484, 1483, 1563, 752]

print(f"\nSwift tokens: {len(swift_tokens)} tokens")
print(f"First 10: {swift_tokens[:10]}")

# Load model
print("\nLoading Python model...")
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals

model_dir = PROJECT_ROOT / "models" / "chatterbox"
device = "cpu"

model = ChatterboxMultilingualTTS.from_local(str(model_dir), device=device)
voice_path = PROJECT_ROOT / "baked_voices" / "samantha" / "baked_voice.pt"
model.conds = Conditionals.load(str(voice_path), map_location=device)

# Convert to tensor
speech_tokens = torch.tensor([swift_tokens], dtype=torch.long, device=device)
print(f"Token tensor shape: {speech_tokens.shape}")

# Generate audio using Python S3Gen
print("\nGenerating audio from Swift tokens using Python S3Gen...")
with torch.inference_mode():
    # Use s3gen.forward() to generate audio from tokens
    wav = model.s3gen(
        speech_tokens=speech_tokens,
        ref_wav=None,
        ref_sr=None,
        ref_dict=model.conds.gen,
        finalize=True,
        n_cfm_timesteps=10  # Match Swift ODE timesteps
    )

    wav_np = wav.squeeze(0).detach().cpu().numpy()
    print(f"Generated audio: {len(wav_np)} samples ({len(wav_np)/24000:.2f}s)")
    print(f"   Range: [{wav_np.min():.4f}, {wav_np.max():.4f}]")

    # Apply watermark
    watermarked = model.watermarker.apply_watermark(wav_np, sample_rate=24000)

# Save
output_path = PROJECT_ROOT / "test_audio" / "swift_tokens_python_audio.wav"
wavfile.write(str(output_path), 24000, watermarked)

print(f"\n✅ Saved: {output_path}")
print(f"   Audio range: [{watermarked.min():.4f}, {watermarked.max():.4f}]")
print("\n" + "=" * 80)
