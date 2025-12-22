#!/usr/bin/env python3
"""Generate Python reference audio for comparison with Swift."""

import sys
sys.path.insert(0, '/Users/a10n/Projects/nightingale_TTS/python')

import torch
import numpy as np
from pathlib import Path
import scipy.io.wavfile as wavfile

# Set deterministic
torch.manual_seed(42)
np.random.seed(42)
if hasattr(torch.mps, 'manual_seed'):
    torch.mps.manual_seed(42)

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

print("="*80)
print("GENERATE PYTHON REFERENCE AUDIO")
print("="*80)

# Load chatterbox model
print("\nLoading model...")
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals

model_dir = PROJECT_ROOT / "models" / "chatterbox"
device = "cpu"

model = ChatterboxMultilingualTTS.from_local(str(model_dir), device=device)

# Load voice
voice_path = PROJECT_ROOT / "baked_voices" / "samantha" / "baked_voice.pt"
model.conds = Conditionals.load(str(voice_path), map_location=device)
print(f"  Model loaded: {device}")
print(f"  Voice: samantha")

# Generate audio
text = "Wow! I absolutely cannot believe that it worked on the first try!"
print(f"\nGenerating audio...")
print(f"  Text: '{text}'")
print(f"  Language: en")
print(f"  CFG weight: 0.5")
print(f"  Temperature: 0.001")

audio_tensor = model.generate(
    text=text,
    language_id="en",
    cfg_weight=0.5,
    temperature=0.001
)

# Convert to numpy and save
audio_np = audio_tensor.cpu().numpy()
if audio_np.ndim == 2:
    audio_np = audio_np.squeeze(0)  # Remove batch dimension
print(f"\nAudio generated:")
print(f"  Shape: {audio_np.shape}")
print(f"  Range: [{audio_np.min():.4f}, {audio_np.max():.4f}]")
print(f"  Mean: {audio_np.mean():.4f}")
print(f"  Std: {audio_np.std():.4f}")
print(f"  Duration: {len(audio_np)/24000:.2f}s")

# Save as WAV
output_path = PROJECT_ROOT / "test_audio" / "python_reference.wav"
wavfile.write(str(output_path), 24000, audio_np)

print(f"\n✅ Saved: {output_path}")
print(f"   File size: {output_path.stat().st_size} bytes")

print("\n" + "="*80)
