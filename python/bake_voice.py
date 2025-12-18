#!/usr/bin/env python3
"""
Bake voice embeddings from reference audio for Chatterbox Multilingual TTS.
This pre-computes all voice embeddings at full precision for faster inference.
"""

import torch
import librosa
from pathlib import Path
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Paths
REF_AUDIO = "/Users/a10n/Projects/nightingale_TTS/baked_voices/ref_audio.wav"
OUTPUT_DIR = Path("/Users/a10n/Projects/nightingale_TTS/baked_voices")
OUTPUT_FILE = OUTPUT_DIR / "baked_voice.pt"

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Check reference audio
ref_path = Path(REF_AUDIO)
if not ref_path.exists():
    raise FileNotFoundError(f"Reference audio not found: {REF_AUDIO}")

# Get audio duration
wav, sr = librosa.load(REF_AUDIO, sr=None)
duration = len(wav) / sr
print(f"Reference audio: {REF_AUDIO}")
print(f"Duration: {duration:.2f}s")
print(f"Sample rate: {sr}Hz")

# Load model
print("\nLoading Chatterbox Multilingual TTS...")
model = ChatterboxMultilingualTTS.from_pretrained(device=device)
print("Model loaded successfully")

# Bake voice embeddings at full precision (exaggeration=0.5 is default)
print(f"\nBaking voice embeddings from reference audio...")
print("This will use full length and full precision")
model.prepare_conditionals(REF_AUDIO, exaggeration=0.5)

# Save the baked voice conditionals
print(f"\nSaving baked voice to: {OUTPUT_FILE}")
model.conds.save(OUTPUT_FILE)

# Verify the saved file
file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
print(f"✓ Baked voice saved successfully ({file_size_mb:.2f} MB)")

# Test loading
print("\nVerifying saved voice...")
from chatterbox.mtl_tts import Conditionals
loaded_conds = Conditionals.load(OUTPUT_FILE, map_location=device)
print("✓ Baked voice can be loaded successfully")

print("\n" + "="*60)
print("Voice baking complete!")
print("="*60)
print(f"\nTo use this baked voice in your code:")
print(f"```python")
print(f"from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals")
print(f"")
print(f"model = ChatterboxMultilingualTTS.from_pretrained(device='mps')")
print(f"model.conds = Conditionals.load('{OUTPUT_FILE}', map_location='mps')")
print(f"")
print(f"# Now generate without audio_prompt_path")
print(f"wav = model.generate('Hello world', language_id='en')")
print(f"```")
