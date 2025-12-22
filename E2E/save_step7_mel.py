"""Save Python reference audio for Swift comparison."""
import sys
sys.path.insert(0, "/Users/a10n/Projects/nightingale_TTS/python/chatterbox/src")

import numpy as np
import torch
import scipy.io.wavfile as wav
from pathlib import Path

OUTPUT_DIR = Path("/Users/a10n/Projects/nightingale_TTS/E2E/reference_outputs/samantha/expressive_surprise_en")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Chatterbox model...")
from chatterbox.tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained(device="mps")

# Prepare conditionals from reference audio
voice_path = "/Users/a10n/Projects/nightingale_TTS/baked_voices/samantha/ref_audio.wav"
exaggeration = 0.5
model.prepare_conditionals(voice_path, exaggeration=exaggeration)

# Use same text as Swift test
text = "Wow! I absolutely cannot believe that it worked on the first try!"

print(f"Generating for: {text}")

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate audio 
audio = model.generate(text, temperature=0.7, top_p=0.95, cfg_weight=0.5, repetition_penalty=1.2)

print(f"Generated audio shape: {audio.shape}")
print(f"Audio range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")

# Save audio
audio_np = audio.squeeze().cpu().numpy()
audio_int16 = (np.clip(audio_np, -1, 1) * 32767).astype(np.int16)
wav.write(str(OUTPUT_DIR / "step8_python_audio.wav"), 24000, audio_int16)
print(f"Saved: {OUTPUT_DIR / 'step8_python_audio.wav'}")

# Save raw audio numpy
np.save(OUTPUT_DIR / "step8_audio.npy", audio_np)
print(f"Saved: {OUTPUT_DIR / 'step8_audio.npy'}")

print("\nDone!")
